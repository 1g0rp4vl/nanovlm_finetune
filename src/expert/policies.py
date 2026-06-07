"""Deterministic expert policies for MiniGrid EmptyEnv."""

from src.config import ACTION_IDS

LEFT = ACTION_IDS["left"]
RIGHT = ACTION_IDS["right"]
FORWARD = ACTION_IDS["forward"]
AGENT_X = 3
AGENT_Y = 6


def _goal_position(env):
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return x, y
    raise RuntimeError("Goal not found")


def _turns(current_direction, target_direction):
    difference = (target_direction - current_direction) % 4
    return min(difference, 4 - difference)


def _first_action(current_direction, target_direction):
    difference = (target_direction - current_direction) % 4
    if difference == 0:
        return FORWARD
    if difference == 1:
        return RIGHT
    if difference == 3:
        return LEFT
    return RIGHT


def shortest_path_action(env):
    """Return the first action of the shortest empty-room path to the goal."""
    agent_x, agent_y = env.unwrapped.agent_pos
    direction = env.unwrapped.agent_dir
    goal_x, goal_y = _goal_position(env)
    dx, dy = goal_x - agent_x, goal_y - agent_y
    if dx == dy == 0:
        return None

    candidates = []
    if dx:
        x_direction = 0 if dx > 0 else 2
        cost = abs(dx) + _turns(direction, x_direction)
        if dy:
            y_direction = 1 if dy > 0 else 3
            cost += abs(dy) + _turns(x_direction, y_direction)
        candidates.append((cost, x_direction))
    if dy:
        y_direction = 1 if dy > 0 else 3
        cost = abs(dy) + _turns(direction, y_direction)
        if dx:
            x_direction = 0 if dx > 0 else 2
            cost += abs(dx) + _turns(y_direction, x_direction)
        candidates.append((cost, y_direction))

    return _first_action(direction, min(candidates)[1])


def visible_goal_position(env):
    """Return whether the goal appears left, right, or ahead in partial view."""
    grid, _ = env.unwrapped.gen_obs_grid()
    agent_x = grid.width // 2
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                if x < agent_x:
                    return "left"
                if x > agent_x:
                    return "right"
                return "ahead"
    return None


def visible_goal_clockwise_action(env):
    """Follow a shortest path to a visible goal; otherwise turn right."""
    if visible_goal_position(env) is not None:
        return shortest_path_action(env)
    return RIGHT


def _cell_type(grid, x, y):
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return None
    cell = grid.get(x, y)
    return cell.type if cell is not None else None


def _is_wall(grid, x, y):
    return _cell_type(grid, x, y) == "wall"


def _visible_goal(grid, visible):
    for x in range(grid.width):
        for y in range(grid.height):
            if visible[x, y] and _cell_type(grid, x, y) == "goal":
                return x, y
    return None


def _goal_action(grid, goal):
    goal_x, goal_y = goal
    dx, dy = goal_x - AGENT_X, goal_y - AGENT_Y
    if dx == 0:
        return FORWARD
    if dy < 0 and not _is_wall(grid, AGENT_X, AGENT_Y - 1):
        return FORWARD
    return LEFT if dx < 0 else RIGHT


def _sees_both_side_walls(grid, visible):
    for y in range(grid.height):
        open_cells = [
            x for x in range(grid.width) if visible[x, y] and not _is_wall(grid, x, y)
        ]
        if not open_cells:
            continue
        left, right = min(open_cells), max(open_cells)
        if (
            left > 0
            and right + 1 < grid.width
            and visible[left - 1, y]
            and visible[right + 1, y]
            and _is_wall(grid, left - 1, y)
            and _is_wall(grid, right + 1, y)
        ):
            return True
    return False


def _open_cells(grid, visible, x_range):
    return sum(
        visible[x, y] and not _is_wall(grid, x, y)
        for x in x_range
        for y in range(grid.height)
    )


def partial_observation_action(env):
    """Navigate using only the visible grid and a wall-following heuristic."""
    grid, visible = env.unwrapped.gen_obs_grid()
    goal = _visible_goal(grid, visible)
    if goal is not None:
        return _goal_action(grid, goal)
    if _sees_both_side_walls(grid, visible):
        return RIGHT

    front = _is_wall(grid, AGENT_X, AGENT_Y - 1)
    left = _is_wall(grid, AGENT_X - 1, AGENT_Y)
    right = _is_wall(grid, AGENT_X + 1, AGENT_Y)
    if front and left:
        return RIGHT
    if front and right:
        return LEFT
    if front:
        left_open = _open_cells(grid, visible, range(AGENT_X))
        right_open = _open_cells(grid, visible, range(AGENT_X + 1, grid.width))
        return LEFT if left_open < right_open else RIGHT
    return FORWARD


EXPERTS = {
    "shortest_path": shortest_path_action,
    "visible_goal_clockwise": visible_goal_clockwise_action,
    "partial_observation": partial_observation_action,
}
