"""Batched GRPO fine-tuning of NanoVLM in MiniGrid EmptyEnv."""

import argparse
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.config import ACTION_IDS, PROMPTS
from src.utils import (
    generate,
    get_amp_dtype,
    make_env,
    prepare_vlm_batch,
    reset_outputs,
    resolve_checkpoint,
    set_trainable,
    write_json,
)


class GRPOTrainer:
    """Generate grouped trajectories and optimize the clipped GRPO objective."""

    def __init__(self, args, policy, reference, tokenizer, image_processor, device):
        self.args = args
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.dtype = get_amp_dtype(args.amp_dtype, device)
        self.prompt = PROMPTS[args.prompt]

    def generate_actions(self, observations, greedy=False):
        """Generate token sequences and actions for a batch of observations."""
        tokens, _, actions = generate(
            self.policy,
            self.tokenizer,
            self.image_processor,
            observations,
            [self.prompt] * len(observations),
            self.device,
            max_new_tokens=self.args.max_new_tokens,
            batch_size=self.args.generation_batch_size,
            greedy=greedy,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            dtype=self.dtype,
        )
        return tokens, actions

    def sequence_logprobs(self, model, samples):
        """Return summed generated-token log-probabilities for each sample."""
        prompt_ids, prompt_mask, image_packs = prepare_vlm_batch(
            model,
            self.tokenizer,
            self.image_processor,
            [sample["observation"] for sample in samples],
            [self.prompt] * len(samples),
            self.device,
            padding_side="right",
        )
        prompt_lengths = prompt_mask.sum(dim=1).tolist()
        generated = [sample["tokens"].to(self.device) for sample in samples]
        sequences = [
            torch.cat([prompt_ids[index, :length], generated[index]])
            for index, length in enumerate(prompt_lengths)
        ]

        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            batch = self.tokenizer.pad(
                {"input_ids": sequences},
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = previous_padding

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = torch.full_like(input_ids, -100)
        for index, (length, tokens) in enumerate(
            zip(prompt_lengths, generated, strict=True)
        ):
            labels[index, length : length + len(tokens)] = tokens

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype is not None,
        ):
            hidden, _ = model(input_ids, image_packs, attention_mask=attention_mask)
            logits = model.decoder.head(hidden[:, :-1])
            targets = labels[:, 1:]
            mask = targets.ne(-100)
            token_logprobs = (
                F.log_softmax(logits, dim=-1)
                .gather(
                    2,
                    targets.masked_fill(~mask, 0).unsqueeze(-1),
                )
                .squeeze(-1)
            )
        return (token_logprobs * mask).sum(dim=1)

    @torch.no_grad()
    def score(self, model, samples):
        """Score samples in bounded GPU batches."""
        scores = []
        for start in range(0, len(samples), self.args.score_batch_size):
            batch = samples[start : start + self.args.score_batch_size]
            scores.extend(self.sequence_logprobs(model, batch).cpu().tolist())
        return scores

    @torch.no_grad()
    def run_episodes(self, seeds, greedy=False):
        """Run many MiniGrid environments with batched model generation."""
        envs = [make_env() for _ in seeds]
        observations = [
            env.reset(seed=seed)[0] for env, seed in zip(envs, seeds, strict=True)
        ]
        trajectories = [
            {"samples": [], "return": 0.0, "success": False, "truncated": False}
            for _ in seeds
        ]
        active = list(range(len(seeds)))

        try:
            for _ in range(self.args.rollout_max_steps):
                if not active:
                    break
                current = [np.asarray(observations[index]).copy() for index in active]
                token_rows, actions = self.generate_actions(current, greedy)
                next_active = []

                for index, observation, tokens, action in zip(
                    active,
                    current,
                    token_rows,
                    actions,
                    strict=True,
                ):
                    invalid = action is None
                    trajectories[index]["samples"].append(
                        {
                            "observation": observation,
                            "tokens": tokens,
                            "action": action or "invalid",
                            "invalid": invalid,
                        }
                    )
                    if invalid:
                        trajectories[index]["return"] += self.args.invalid_reward
                        continue

                    next_observation, reward, terminated, truncated, _ = envs[
                        index
                    ].step(ACTION_IDS[action])
                    observations[index] = next_observation
                    trajectories[index]["return"] += float(reward)
                    trajectories[index]["success"] = bool(terminated and not truncated)
                    trajectories[index]["truncated"] = bool(truncated)
                    if not terminated and not truncated:
                        next_active.append(index)
                active = next_active
        finally:
            for env in envs:
                env.close()

        for trajectory in trajectories:
            trajectory["length"] = len(trajectory["samples"])
        return trajectories

    @staticmethod
    def behavior_metrics(trajectories, prefix):
        """Summarize rewards, success, lengths, and generated actions."""
        samples = [
            sample for trajectory in trajectories for sample in trajectory["samples"]
        ]
        counts = Counter(sample["action"] for sample in samples)
        total = max(len(samples), 1)
        metrics = {
            f"{prefix}/success_rate": float(
                np.mean([trajectory["success"] for trajectory in trajectories])
            ),
            f"{prefix}/mean_return": float(
                np.mean([trajectory["return"] for trajectory in trajectories])
            ),
            f"{prefix}/episode_length": float(
                np.mean([trajectory["length"] for trajectory in trajectories])
            ),
            f"{prefix}/invalid_action_rate": counts["invalid"] / total,
            f"{prefix}/truncated_rate": float(
                np.mean([trajectory["truncated"] for trajectory in trajectories])
            ),
        }
        for action in ("left", "right", "forward", "invalid"):
            metrics[f"{prefix}/action_{action}_frac"] = counts[action] / total
        return metrics

    @torch.no_grad()
    def rollout(self, update):
        """Generate one grouped rollout batch and attach GRPO statistics."""
        seeds = []
        for index in range(self.args.rollout_batch_size):
            seed = (
                self.args.rollout_seed_start
                + update * self.args.rollout_batch_size
                + index
            )
            seeds.extend([seed] * self.args.group_size)

        trajectories = self.run_episodes(seeds)
        returns = torch.tensor(
            [trajectory["return"] for trajectory in trajectories]
        ).view(self.args.rollout_batch_size, self.args.group_size)
        group_std = returns.std(dim=1, keepdim=True, unbiased=False)
        advantages = (returns - returns.mean(dim=1, keepdim=True)) / (group_std + 1e-6)

        samples = []
        for trajectory, advantage in zip(
            trajectories,
            advantages.flatten().tolist(),
            strict=True,
        ):
            for sample in trajectory["samples"]:
                sample["advantage"] = advantage
                samples.append(sample)

        old_scores = self.score(self.policy, samples)
        reference_scores = self.score(self.reference, samples)
        for sample, old_score, reference_score in zip(
            samples,
            old_scores,
            reference_scores,
            strict=True,
        ):
            sample["old_logprob"] = old_score
            sample["reference_logprob"] = reference_score

        metrics = {
            "train/reward_mean": float(returns.mean()),
            "train/group_reward_std": float(group_std.mean()),
            **self.behavior_metrics(trajectories, "train"),
        }
        return samples, metrics

    def loss(self, samples):
        """Compute clipped policy loss and KL penalty for one minibatch."""
        current = self.sequence_logprobs(self.policy, samples)
        old = torch.tensor(
            [sample["old_logprob"] for sample in samples],
            device=self.device,
        )
        reference = torch.tensor(
            [sample["reference_logprob"] for sample in samples],
            device=self.device,
        )
        advantage = torch.tensor(
            [sample["advantage"] for sample in samples],
            device=self.device,
        )

        ratio = torch.exp(current - old)
        clipped_ratio = ratio.clamp(
            1 - self.args.clip_range,
            1 + self.args.clip_range,
        )
        policy_loss = -torch.minimum(
            ratio * advantage,
            clipped_ratio * advantage,
        ).mean()
        kl = (torch.exp(reference - current) - (reference - current) - 1).mean()
        loss = policy_loss + self.args.beta * kl
        return loss, {
            "train/grpo_loss": float(loss.detach()),
            "train/policy_loss": float(policy_loss.detach()),
            "train/kl": float(kl.detach()),
        }

    @torch.no_grad()
    def evaluate(self, episodes=None):
        """Evaluate greedily on a fixed sequence of seeds."""
        self.policy.eval()
        count = episodes or self.args.eval_episodes
        seeds = [self.args.eval_seed_start + index for index in range(count)]
        return self.behavior_metrics(self.run_episodes(seeds, greedy=True), "eval")


def mean_metrics(rows):
    """Average numeric metrics across minibatches or a moving window."""
    keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    }
    return {
        key: float(np.mean([row[key] for row in rows if key in row])) for key in keys
    }


def append_jsonl(path: Path, row) -> None:
    """Append one JSON object to a metrics file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def parse_args():
    """Parse batched GRPO options."""
    parser = argparse.ArgumentParser(
        description="Batched GRPO fine-tuning for NanoVLM MiniGrid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--prompt",
        choices=PROMPTS,
        default="policy",
        help="Generation format.",
    )
    parser.add_argument(
        "--nanovlm-dir",
        type=Path,
        default=Path("external/nanoVLM"),
        help="Cloned nanoVLM repository.",
    )
    parser.add_argument("--updates", type=int, default=100, help="GRPO updates.")
    parser.add_argument("--lr", type=float, default=1e-6, help="AdamW learning rate.")
    parser.add_argument(
        "--trainable",
        choices=["mp", "decoder", "decoder_mp", "all"],
        default="mp",
        help="NanoVLM modules to update.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["auto", "bfloat16", "float16", "off"],
        default="auto",
        help="Mixed-precision dtype.",
    )
    parser.add_argument(
        "--rollout-batch-size", type=int, default=4, help="Start states per update."
    )
    parser.add_argument(
        "--group-size", type=int, default=4, help="Trajectories per start state."
    )
    parser.add_argument(
        "--generation-batch-size", type=int, default=32, help="Generation batch size."
    )
    parser.add_argument(
        "--score-batch-size", type=int, default=32, help="Log-probability batch size."
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=32, help="Optimization minibatch size."
    )
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient."
    )
    parser.add_argument(
        "--clip-range", type=float, default=0.2, help="Probability-ratio clipping."
    )
    parser.add_argument(
        "--invalid-reward", type=float, default=-1.0, help="Invalid-output penalty."
    )
    parser.add_argument(
        "--rollout-max-steps", type=int, default=40, help="Episode step limit."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generated tokens; inferred as 1 or 48 when omitted.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling.")
    parser.add_argument(
        "--eval-every", type=int, default=10, help="Updates between evaluations."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50, help="Episodes per intermediate eval."
    )
    parser.add_argument(
        "--final-episodes", type=int, default=200, help="Episodes in final eval."
    )
    parser.add_argument(
        "--save-every", type=int, default=50, help="Checkpoint interval."
    )
    parser.add_argument(
        "--metric-window", type=int, default=10, help="Moving-average window."
    )
    parser.add_argument(
        "--rollout-seed-start", type=int, default=20_000, help="First training seed."
    )
    parser.add_argument(
        "--eval-seed-start", type=int, default=30_000, help="First evaluation seed."
    )
    return parser.parse_args()


def main():
    """Run batched GRPO training, evaluation, logging, and checkpointing."""
    args = parse_args()
    args.max_new_tokens = args.max_new_tokens or (
        48 if args.prompt == "plan_action" else 1
    )
    sys.path.insert(0, str(args.nanovlm_dir.resolve()))
    from data.processors import get_image_processor, get_tokenizer
    from models.vision_language_model import VisionLanguageModel

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = VisionLanguageModel.from_pretrained(checkpoint).to(device)
    reference = VisionLanguageModel.from_pretrained(checkpoint).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    set_trainable(policy, args.trainable)

    tokenizer = get_tokenizer(
        policy.cfg.lm_tokenizer,
        policy.cfg.vlm_extra_tokens,
        policy.cfg.lm_chat_template,
    )
    image_processor = get_image_processor(
        policy.cfg.max_img_size,
        policy.cfg.vit_img_size,
        False,
    )
    trainer = GRPOTrainer(
        args,
        policy,
        reference,
        tokenizer,
        image_processor,
        device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and trainer.dtype == torch.float16,
    )
    reset_outputs(args.output_dir)
    metrics_path = args.output_dir / "metrics.jsonl"
    window = deque(maxlen=args.metric_window)

    progress = trange(1, args.updates + 1, desc="GRPO")
    for update in progress:
        policy.eval()
        samples, rollout_metrics = trainer.rollout(update)
        random.shuffle(samples)

        minibatch_metrics = []
        policy.train()
        for start in range(0, len(samples), args.minibatch_size):
            loss, metrics = trainer.loss(samples[start : start + args.minibatch_size])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            minibatch_metrics.append(metrics)

        row = {
            "update": update,
            **rollout_metrics,
            **mean_metrics(minibatch_metrics),
        }
        window.append(row)
        row.update(
            {
                f"window/{key}": value
                for key, value in mean_metrics(window).items()
                if key != "update"
            }
        )
        if update % args.eval_every == 0:
            row.update(trainer.evaluate())
            progress.write(
                f"eval update={update} success={row['eval/success_rate']:.3f} "
                f"return={row['eval/mean_return']:.3f}"
            )
        append_jsonl(metrics_path, row)
        progress.set_postfix(
            reward=f"{row['train/reward_mean']:.3f}",
            success=f"{row['train/success_rate']:.2f}",
            loss=f"{row['train/grpo_loss']:.3f}",
            kl=f"{row['train/kl']:.3f}",
        )

        if args.save_every and update % args.save_every == 0:
            policy.save_pretrained(str(args.output_dir / f"update_{update}"))

    final_dir = args.output_dir / "final"
    policy.save_pretrained(str(final_dir))
    summary = {
        "checkpoint": str(final_dir),
        "source_checkpoint": checkpoint,
        "prompt": args.prompt,
        "updates": args.updates,
        "rollout_episodes": (args.updates * args.rollout_batch_size * args.group_size),
        **trainer.evaluate(args.final_episodes),
    }
    write_json(args.output_dir / "final.json", summary)
    write_json(final_dir / "summary.json", summary)
    print(f"Saved final checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
