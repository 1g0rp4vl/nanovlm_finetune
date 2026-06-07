"""Shared MiniGrid actions, environments, and prompts."""

ENV_ID = "MiniGrid-Empty-Random-6x6-v0"

MIXED_ENV_IDS = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-Random-5x5-v0",
    "MiniGrid-Empty-6x6-v0",
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
]

ACTIONS = {0: "left", 1: "right", 2: "forward"}
ACTION_IDS = {name: action_id for action_id, name in ACTIONS.items()}

ACTION_PROMPT = """Task: MiniGrid Empty navigation.

You receive the agent's current partial RGB observation.
Choose the next action.

Valid actions: left, right, forward.
Return exactly one action word: left, right, or forward."""

POLICY_PROMPT = """Task: MiniGrid Empty navigation.

You receive the agent's current partial RGB observation.
The goal is the green square.

Expert policy:
- If the green goal is not visible, turn right.
- If the green goal is visible, move toward it using the shortest path.

Valid actions: left, right, forward.
Return exactly one action word: left, right, or forward."""

PLAN_ACTION_PROMPT = """Task: MiniGrid Empty navigation.

You receive the agent's current partial RGB observation.
The goal is the green square.

Write two short sentences: first describe the visible state, then state your plan.
After those sentences, output the next action as the final word.

Valid final actions: left, right, forward."""

PROMPTS = {
    "action": ACTION_PROMPT,
    "policy": POLICY_PROMPT,
    "plan_action": PLAN_ACTION_PROMPT,
}
