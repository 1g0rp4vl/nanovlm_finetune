"""Shared environment, model, generation, and file helpers."""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image

from src.config import ACTION_IDS, ENV_ID


def make_env(env_id: str = ENV_ID):
    """Create a MiniGrid environment that returns partial RGB observations."""
    env = gym.make(env_id, render_mode="rgb_array")
    return ImgObsWrapper(RGBImgPartialObsWrapper(env))


def parse_action(text: str) -> str | None:
    """Read an action from the final word of a generated response."""
    words = text.strip().lower().split()
    if not words:
        return None
    action = re.sub(r"^[^a-z]+|[^a-z]+$", "", words[-1])
    return action if action in ACTION_IDS else None


def set_trainable(model, mode: str) -> None:
    """Select which NanoVLM modules receive gradient updates."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    modules = {
        "mp": [model.MP],
        "decoder": [model.decoder],
        "decoder_mp": [model.decoder, model.MP],
        "all": [model],
    }[mode]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def get_amp_dtype(name: str, device: torch.device):
    """Resolve the requested mixed-precision dtype for the current device."""
    if name == "off" or device.type != "cuda":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def as_rgb_image(image) -> Image.Image:
    """Convert a path, array, or PIL image to RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return Image.fromarray(image).convert("RGB")


def prepare_vlm_batch(
    model,
    tokenizer,
    image_processor,
    images,
    prompts,
    device,
    padding_side: str,
):
    """Build NanoVLM prompt tensors and processed image packs."""
    from data.processors import get_image_string

    image_packs = []
    messages = []
    for image, prompt in zip(images, prompts, strict=True):
        image_parts, image_grid = image_processor(as_rgb_image(image))
        if (
            not hasattr(tokenizer, "global_image_token")
            and image_grid[0] * image_grid[1] == len(image_parts) - 1
        ):
            image_parts = image_parts[1:]
        image_packs.append(image_parts)

        image_string = get_image_string(
            tokenizer,
            [image_grid],
            model.cfg.mp_image_token_length,
        )
        messages.append([{"role": "user", "content": image_string + prompt}])

    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = previous_padding

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        image_packs,
    )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    image_processor,
    images,
    prompts,
    device,
    *,
    max_new_tokens: int,
    batch_size: int,
    greedy: bool,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    dtype=None,
):
    """Generate token tensors, decoded texts, and parsed actions in batches."""
    model.eval()
    token_rows = []
    texts = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        input_ids, attention_mask, image_packs = prepare_vlm_batch(
            model,
            tokenizer,
            image_processor,
            batch_images,
            batch_prompts,
            device,
            padding_side="left",
        )
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=dtype is not None,
        ):
            generated = model.generate(
                input_ids,
                image_packs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        token_rows.extend(row.detach().cpu() for row in generated)
        texts.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return token_rows, texts, [parse_action(text) for text in texts]


@torch.no_grad()
def evaluate_policy(
    model,
    tokenizer,
    image_processor,
    prompt: str,
    device,
    *,
    episodes: int,
    seed_start: int,
    max_steps: int,
    max_new_tokens: int,
    batch_size: int,
    dtype=None,
):
    """Evaluate a greedy policy in parallel MiniGrid environments."""
    envs = [make_env() for _ in range(episodes)]
    observations = [
        env.reset(seed=seed_start + index)[0] for index, env in enumerate(envs)
    ]
    active = list(range(episodes))
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=bool)
    truncated = np.zeros(episodes, dtype=bool)
    actions = Counter()

    try:
        for _ in range(max_steps):
            if not active:
                break
            images = [observations[index] for index in active]
            _, _, batch_actions = generate(
                model,
                tokenizer,
                image_processor,
                images,
                [prompt] * len(images),
                device,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                greedy=True,
                dtype=dtype,
            )

            next_active = []
            for index, action in zip(active, batch_actions, strict=True):
                lengths[index] += 1
                actions[action or "invalid"] += 1
                if action is None:
                    continue
                observation, reward, terminated, was_truncated, _ = envs[index].step(
                    ACTION_IDS[action]
                )
                observations[index] = observation
                returns[index] += reward
                successes[index] = terminated and not was_truncated
                truncated[index] = was_truncated
                if not terminated and not was_truncated:
                    next_active.append(index)
            active = next_active
    finally:
        for env in envs:
            env.close()

    total_actions = max(sum(actions.values()), 1)
    return {
        "success_rate": float(successes.mean()),
        "mean_return": float(returns.mean()),
        "episode_length": float(lengths.mean()),
        "truncated_rate": float(truncated.mean()),
        "invalid_action_rate": actions["invalid"] / total_actions,
        "action_distribution": {
            action: actions[action] / total_actions
            for action in ("left", "right", "forward", "invalid")
        },
    }


def reset_outputs(output_dir: Path) -> None:
    """Remove a previous run and recreate its output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def write_json(path: Path, data) -> None:
    """Write formatted JSON and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_checkpoint(source: str) -> str:
    """Return a local checkpoint path or a Hugging Face model identifier."""
    path = Path(source)
    if path.exists():
        return str(path)
    if path.is_absolute() or source.startswith((".", "checkpoints/")):
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    return source
