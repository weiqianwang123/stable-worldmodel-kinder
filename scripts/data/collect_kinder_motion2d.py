"""Collect and visualize KinDER Motion2D episodes.

This is a smoke-data script for the stable-worldmodel KinDER adapter. It
creates one ``swm/KinderMotion2D-pN-v0`` environment, rolls out a small policy,
saves the episode arrays to ``episode.npz``, and writes quick visualizations.
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import shutil
import sys
import types
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-passages', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='*', default=None)
    parser.add_argument(
        '--start-seed',
        type=int,
        default=0,
        help='First seed used with --num-episodes.',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Collect consecutive seeds [start-seed, start-seed + N).',
    )
    parser.add_argument(
        '--policy',
        choices=('astar', 'greedy', 'random'),
        default='astar',
    )
    parser.add_argument('--grid-resolution', type=float, default=0.025)
    parser.add_argument('--clearance', type=float, default=0.005)
    parser.add_argument('--waypoint-tolerance', type=float, default=0.015)
    parser.add_argument('--heading-tolerance', type=float, default=0.18)
    parser.add_argument('--rotate-in-place-threshold', type=float, default=0.45)
    parser.add_argument('--passage-approach-margin', type=float, default=0.14)
    parser.add_argument(
        '--disable-passage-waypoints',
        action='store_true',
        help='Use only grid A* instead of Motion2D passage-center waypoints.',
    )
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument(
        '--artifact-limit',
        type=int,
        default=None,
        help=(
            'Save per-episode npz/gif/trajectory only for the first N '
            'episodes. Defaults to all episodes; use 0 for dataset only.'
        ),
    )
    parser.add_argument(
        '--no-gif',
        action='store_true',
        help='Skip episode.gif files when saving per-episode artifacts.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('outputs/kinder_motion2d_episode'),
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=None,
        help='Folder-format dataset destination. Defaults under out-dir.',
    )
    parser.add_argument(
        '--kindergarden-home',
        type=str,
        default=None,
        help='Optional local kindergarden checkout root.',
    )
    args = parser.parse_args()
    if args.seeds is not None and args.num_episodes is not None:
        parser.error('use either --seeds or --num-episodes, not both')
    if args.num_episodes is not None and args.num_episodes <= 0:
        parser.error('--num-episodes must be positive')
    if args.artifact_limit is not None and args.artifact_limit < 0:
        parser.error('--artifact-limit must be non-negative')
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.seeds is not None:
        seeds = args.seeds
    elif args.num_episodes is not None:
        seeds = list(
            range(args.start_seed, args.start_seed + args.num_episodes)
        )
    else:
        seeds = [args.seed]

    ensure_motion2d_registered()

    env_id = f'swm/KinderMotion2D-p{args.num_passages}-v0'
    make_kwargs = {
        'max_episode_steps': args.max_steps,
        'render_mode': 'rgb_array',
    }
    if args.kindergarden_home is not None:
        make_kwargs['kindergarden_home'] = args.kindergarden_home

    dataset_dir = args.dataset_dir or args.out_dir / 'dataset_folder'
    dataset_writer = FolderDatasetBuilder(dataset_dir)
    episode_meta: list[dict] = []
    for ep_idx, seed in enumerate(seeds):
        env = gym.make(env_id, **make_kwargs)
        env.action_space.seed(seed)
        try:
            episode = collect_episode(env, args, seed=seed)
        finally:
            env.close()

        dataset_writer.add_episode(episode)
        paths: dict[str, Path] = {}
        save_artifacts = (
            args.artifact_limit is None or ep_idx < args.artifact_limit
        )
        if save_artifacts:
            seed_dir = (
                args.out_dir
                if len(seeds) == 1
                else args.out_dir / f'seed_{seed:02d}'
            )
            seed_dir.mkdir(parents=True, exist_ok=True)
            paths = save_episode_artifacts(
                episode,
                seed_dir,
                fps=args.fps,
                save_gif_artifact=not args.no_gif,
            )
        episode_meta.append(
            {
                'seed': int(seed),
                'num_steps': int(episode['pixels'].shape[0]),
                'terminated': bool(episode['terminated'][-1]),
                'truncated': bool(episode['truncated'][-1]),
                **{k: str(v) for k, v in paths.items()},
            }
        )
        print(
            f'[{ep_idx + 1}/{len(seeds)}] seed={seed} '
            f"steps={episode_meta[-1]['num_steps']} "
            f"terminated={episode_meta[-1]['terminated']} "
            f"truncated={episode_meta[-1]['truncated']}",
            flush=True,
        )

    dataset_writer.close()
    summary_path = args.out_dir / 'trajectories.png'
    save_trajectory_contact_sheet(episode_meta, summary_path)

    metadata = {
        'env_id': env_id,
        'seeds': [int(seed) for seed in seeds],
        'policy': args.policy,
        'num_episodes': len(seeds),
        'dataset_dir': str(dataset_dir),
        'summary_png': str(summary_path),
        'episodes': episode_meta,
    }
    metadata_path = args.out_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')

    print(json.dumps(metadata, indent=2))


def ensure_motion2d_registered() -> None:
    try:
        import stable_worldmodel.envs  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if exc.name in {'gymnasium', 'numpy', 'PIL'}:
            raise
        print(
            'Warning: full stable_worldmodel env registration failed because '
            f'{exc.name!r} is missing; registering only KinDER Motion2D for '
            'this smoke script.',
            file=sys.stderr,
        )

    for name in list(sys.modules):
        if name == 'stable_worldmodel' or name.startswith(
            'stable_worldmodel.'
        ):
            del sys.modules[name]

    package = types.ModuleType('stable_worldmodel')
    package.__path__ = [str(ROOT / 'stable_worldmodel')]
    sys.modules['stable_worldmodel'] = package

    envs_package = types.ModuleType('stable_worldmodel.envs')
    envs_package.__path__ = [str(ROOT / 'stable_worldmodel' / 'envs')]
    sys.modules['stable_worldmodel.envs'] = envs_package

    mod = importlib.import_module('stable_worldmodel.envs.kinder.motion2d')
    cls = mod.KinderMotion2D

    safe_register(
        id='swm/KinderMotion2D-v0',
        entry_point=cls,
    )
    for num_passages in range(6):
        safe_register(
            id=f'swm/KinderMotion2D-p{num_passages}-v0',
            entry_point=cls,
            kwargs={'num_passages': num_passages},
        )


def safe_register(id: str, **kwargs) -> None:
    if id not in registry:
        register(id=id, **kwargs)


def collect_episode(
    env: gym.Env,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    obs, info = env.reset(seed=seed)
    del obs
    planner = (
        AStarMotion2DPolicy(env, args)
        if args.policy == 'astar'
        else None
    )
    if planner is not None:
        planner.reset(info)

    buffers: dict[str, list[np.ndarray]] = {
        'pixels': [],
        'state': [],
        'proprio': [],
        'goal_state': [],
        'goal_proprio': [],
        'action': [],
        'reward': [],
        'terminated': [],
        'truncated': [],
        'step_idx': [],
    }

    dummy_action = np.full(env.action_space.shape, np.nan, dtype=np.float32)
    append_step(
        buffers,
        frame=env.render(),
        info=info,
        action=dummy_action,
        reward=np.nan,
        terminated=False,
        truncated=False,
        step_idx=0,
        image_size=args.image_size,
    )

    for step_idx in range(1, args.max_steps + 1):
        action = (
            planner.action(info)
            if planner is not None
            else choose_action(env, info, args.policy)
        )
        _, reward, terminated, truncated, info = env.step(action)
        append_step(
            buffers,
            frame=env.render(),
            info=info,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            step_idx=step_idx,
            image_size=args.image_size,
        )
        if terminated or truncated:
            break

    episode = {k: np.asarray(v) for k, v in buffers.items()}
    if planner is not None and planner.path is not None:
        episode['planned_path'] = np.asarray(planner.path, dtype=np.float32)
    if 'goal' in info:
        episode['goal_pixels'] = resize_image(info['goal'], args.image_size)
    return episode


def save_episode_artifacts(
    episode: dict[str, np.ndarray],
    out_dir: Path,
    *,
    fps: int,
    save_gif_file: bool = True,
    save_gif_artifact: bool | None = None,
) -> dict[str, Path]:
    if save_gif_artifact is not None:
        save_gif_file = save_gif_artifact

    npz_path = out_dir / 'episode.npz'
    np.savez_compressed(npz_path, **episode)

    trajectory_path = out_dir / 'trajectory.png'
    save_trajectory_overlay(episode, trajectory_path)

    paths = {
        'episode_npz': npz_path,
        'trajectory_png': trajectory_path,
    }
    if save_gif_file:
        gif_path = out_dir / 'episode.gif'
        save_gif(episode['pixels'], gif_path, fps=fps)
        paths['episode_gif'] = gif_path
    return paths


def write_folder_dataset(
    episodes: list[dict[str, np.ndarray]],
    dataset_dir: Path,
) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pixels_dir = dataset_dir / 'pixels'
    pixels_dir.mkdir()

    lengths = np.asarray(
        [len(ep['pixels']) for ep in episodes],
        dtype=np.int32,
    )
    offsets = np.concatenate(
        ([0], np.cumsum(lengths[:-1], dtype=np.int64))
    ).astype(np.int64)
    np.savez(dataset_dir / 'ep_len.npz', lengths)
    np.savez(dataset_dir / 'ep_offset.npz', offsets)

    tabular_keys = [
        'action',
        'reward',
        'terminated',
        'truncated',
        'step_idx',
        'state',
        'proprio',
        'goal_state',
        'goal_proprio',
    ]
    for key in tabular_keys:
        values = np.concatenate([ep[key] for ep in episodes])
        np.savez(dataset_dir / f'{key}.npz', values)

    for ep_idx, episode in enumerate(episodes):
        for step, frame in enumerate(episode['pixels']):
            Image.fromarray(frame).save(
                pixels_dir / f'ep_{ep_idx}_step_{step}.jpeg'
            )


class FolderDatasetBuilder:
    """Streaming folder dataset writer for larger collection jobs."""

    tabular_keys = (
        'action',
        'reward',
        'terminated',
        'truncated',
        'step_idx',
        'state',
        'proprio',
        'goal_state',
        'goal_proprio',
    )

    def __init__(self, dataset_dir: Path) -> None:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = dataset_dir
        self.pixels_dir = dataset_dir / 'pixels'
        self.pixels_dir.mkdir()
        self.lengths: list[int] = []
        self.offsets: list[int] = []
        self._global_offset = 0
        self._tabular = {key: [] for key in self.tabular_keys}

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        ep_idx = len(self.lengths)
        length = int(len(episode['pixels']))
        self.lengths.append(length)
        self.offsets.append(self._global_offset)
        self._global_offset += length

        for key in self.tabular_keys:
            self._tabular[key].append(episode[key])

        for step, frame in enumerate(episode['pixels']):
            Image.fromarray(frame).save(
                self.pixels_dir / f'ep_{ep_idx}_step_{step}.jpeg'
            )
        self.flush()

    def flush(self) -> None:
        np.savez(
            self.dataset_dir / 'ep_len.npz',
            np.asarray(self.lengths, dtype=np.int32),
        )
        np.savez(
            self.dataset_dir / 'ep_offset.npz',
            np.asarray(self.offsets, dtype=np.int64),
        )
        for key, chunks in self._tabular.items():
            values = np.concatenate(chunks)
            np.savez(self.dataset_dir / f'{key}.npz', values)

    def close(self) -> None:
        self.flush()


class AStarMotion2DPolicy:
    """Grid A* over Motion2D rectangles, then waypoint following."""

    world_min = 0.0
    world_max = 2.5

    def __init__(self, env: gym.Env, args: argparse.Namespace) -> None:
        self.env = env
        self.resolution = float(args.grid_resolution)
        self.clearance = float(args.clearance)
        self.waypoint_tolerance = float(args.waypoint_tolerance)
        self.heading_tolerance = float(args.heading_tolerance)
        self.rotate_in_place_threshold = float(args.rotate_in_place_threshold)
        self.passage_approach_margin = float(args.passage_approach_margin)
        self.use_passage_waypoints = not args.disable_passage_waypoints
        self.path: list[np.ndarray] | None = None
        self._waypoint_idx = 1

    def reset(self, info: dict) -> None:
        obj_state = self._devectorize(info['state'])
        robot = self._object_by_name(obj_state, 'robot')
        target = self._object_by_name(obj_state, 'target_region')

        start = self._xy(obj_state, robot)
        goal = self._rect_center(obj_state, target)
        robot_radius = float(obj_state.get(robot, 'base_radius'))
        obstacles = self._obstacle_rects(obj_state)

        if self.use_passage_waypoints:
            path = self._passage_path(start, goal, obstacles, robot_radius)
            if path is not None:
                self.path = path
                self._waypoint_idx = 1 if len(path) > 1 else 0
                return

        margin_candidates = [
            max(0.0, robot_radius - 0.5 * self.clearance),
            robot_radius,
            0.75 * robot_radius,
            robot_radius + self.clearance,
        ]
        for margin in margin_candidates:
            path = self._plan(start, goal, obstacles, margin)
            if path is not None:
                self.path = path
                self._waypoint_idx = 1 if len(path) > 1 else 0
                return
        raise RuntimeError('A* failed to find a Motion2D path.')

    def action(self, info: dict) -> np.ndarray:
        if self.path is None:
            self.reset(info)

        assert self.path is not None
        robot_xy = np.asarray(info['proprio'][:2], dtype=np.float32)
        while self._waypoint_idx < len(self.path) - 1:
            target = self.path[self._waypoint_idx]
            if np.linalg.norm(target - robot_xy) > self.waypoint_tolerance:
                break
            self._waypoint_idx += 1

        target = self.path[self._waypoint_idx]
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        delta = target - robot_xy
        low = np.asarray(self.env.action_space.low, dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        if np.linalg.norm(delta) > 1e-6:
            theta = float(info['proprio'][2])
            desired_theta = float(np.arctan2(delta[1], delta[0]))
            dtheta = self._angle_diff(desired_theta, theta)
            action[2] = dtheta
            if abs(dtheta) <= self.rotate_in_place_threshold:
                scale = 1.0 if abs(dtheta) <= self.heading_tolerance else 0.35
                action[:2] = scale * delta
        action[3] = low[3]
        action[4] = 0.0
        return np.clip(action, low, high).astype(self.env.action_space.dtype)

    def _devectorize(self, state: np.ndarray):
        kinder_env = self.env.unwrapped.kinder_env
        return kinder_env.observation_space.devectorize(
            np.asarray(state, dtype=np.float32)
        )

    @staticmethod
    def _object_by_name(obj_state, name: str):
        for obj in obj_state:
            if obj.name == name:
                return obj
        raise KeyError(name)

    @staticmethod
    def _xy(obj_state, obj) -> np.ndarray:
        return np.asarray(
            [obj_state.get(obj, 'x'), obj_state.get(obj, 'y')],
            dtype=np.float32,
        )

    def _rect_center(self, obj_state, obj) -> np.ndarray:
        xy = self._xy(obj_state, obj)
        return xy + np.asarray(
            [obj_state.get(obj, 'width'), obj_state.get(obj, 'height')],
            dtype=np.float32,
        ) / 2.0

    def _obstacle_rects(
        self,
        obj_state,
    ) -> list[tuple[float, float, float, float]]:
        rects = []
        for obj in obj_state:
            if not obj.name.startswith('obstacle'):
                continue
            rects.append(
                (
                    float(obj_state.get(obj, 'x')),
                    float(obj_state.get(obj, 'y')),
                    float(obj_state.get(obj, 'width')),
                    float(obj_state.get(obj, 'height')),
                )
            )
        return rects

    def _passage_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[tuple[float, float, float, float]],
        robot_radius: float,
    ) -> list[np.ndarray] | None:
        passages = self._passage_centers(obstacles)
        if not passages:
            return [start.astype(np.float32), goal.astype(np.float32)]

        x_dir = 1.0 if goal[0] >= start[0] else -1.0
        ordered = sorted(passages, key=lambda item: item[0], reverse=x_dir < 0)
        offset = max(self.passage_approach_margin, robot_radius + self.clearance)
        path = [start.astype(np.float32)]
        for x0, width, gap_low, gap_high in ordered:
            x_center = x0 + 0.5 * width
            if x_dir > 0 and not (start[0] < x_center < goal[0]):
                continue
            if x_dir < 0 and not (goal[0] < x_center < start[0]):
                continue

            gap_height = gap_high - gap_low
            if gap_height <= 2.0 * robot_radius:
                return None

            gap_y = 0.5 * (gap_low + gap_high)
            before_x = x0 - offset if x_dir > 0 else x0 + width + offset
            after_x = x0 + width + offset if x_dir > 0 else x0 - offset
            for xy in (
                np.asarray([before_x, gap_y], dtype=np.float32),
                np.asarray([x_center, gap_y], dtype=np.float32),
                np.asarray([after_x, gap_y], dtype=np.float32),
            ):
                xy = np.clip(xy, self.world_min + 1e-3, self.world_max - 1e-3)
                if np.linalg.norm(xy - path[-1]) > self.waypoint_tolerance:
                    path.append(xy.astype(np.float32))

        if np.linalg.norm(goal - path[-1]) > self.waypoint_tolerance:
            path.append(goal.astype(np.float32))

        margin = max(0.0, robot_radius - self.clearance)
        for p0, p1 in zip(path[:-1], path[1:]):
            if not self._line_free(p0, p1, obstacles, margin):
                return None
        return path

    @staticmethod
    def _passage_centers(
        obstacles: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        groups: dict[tuple[float, float], list[tuple[float, float, float, float]]] = {}
        for rect in obstacles:
            x, _, width, _ = rect
            groups.setdefault((round(x, 4), round(width, 4)), []).append(rect)

        passages = []
        for rects in groups.values():
            if len(rects) < 2:
                continue
            rects = sorted(rects, key=lambda rect: rect[1])
            for lower, upper in zip(rects[:-1], rects[1:]):
                x0, _, width, _ = lower
                gap_low = lower[1] + lower[3]
                gap_high = upper[1]
                if gap_high > gap_low:
                    passages.append((x0, width, gap_low, gap_high))
        return passages

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        return float(
            np.arctan2(np.sin(target - current), np.cos(target - current))
        )

    def _plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> list[np.ndarray] | None:
        n = int(round((self.world_max - self.world_min) / self.resolution)) + 1
        start_cell = self._nearest_free(
            self._to_cell(start, n), n, obstacles, margin
        )
        goal_cell = self._nearest_free(
            self._to_cell(goal, n), n, obstacles, margin
        )
        if start_cell is None or goal_cell is None:
            return None

        path_cells = self._astar(start_cell, goal_cell, n, obstacles, margin)
        if path_cells is None:
            return None

        path = [self._from_cell(cell) for cell in path_cells]
        path[0] = start.astype(np.float32)
        path[-1] = goal.astype(np.float32)
        return self._shortcut_path(path, obstacles, margin)

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        n: int,
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> list[tuple[int, int]] | None:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        frontier = [(self._heuristic(start, goal), 0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {
            start: None
        }
        cost_so_far = {start: 0.0}

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct_path(came_from, goal)
            if current_cost > cost_so_far[current]:
                continue
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < n and 0 <= nxt[1] < n):
                    continue
                if self._occupied(nxt, obstacles, margin):
                    continue
                step_cost = float(np.hypot(dx, dy))
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(nxt, float('inf')):
                    continue
                cost_so_far[nxt] = new_cost
                priority = new_cost + self._heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, new_cost, nxt))
                came_from[nxt] = current
        return None

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        cells = [goal]
        cur = goal
        while came_from[cur] is not None:
            cur = came_from[cur]
            cells.append(cur)
        cells.reverse()
        return cells

    def _shortcut_path(
        self,
        path: list[np.ndarray],
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> list[np.ndarray]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_free(path[i], path[j], obstacles, margin):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _line_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> bool:
        dist = float(np.linalg.norm(end - start))
        steps = max(2, int(np.ceil(dist / (0.5 * self.resolution))))
        for alpha in np.linspace(0.0, 1.0, steps):
            xy = start * (1.0 - alpha) + end * alpha
            if self._point_occupied(xy, obstacles, margin):
                return False
        return True

    def _nearest_free(
        self,
        cell: tuple[int, int],
        n: int,
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> tuple[int, int] | None:
        if not self._occupied(cell, obstacles, margin):
            return cell
        queue = deque([cell])
        seen = {cell}
        while queue:
            cur = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (cur[0] + dx, cur[1] + dy)
                    in_bounds = 0 <= nxt[0] < n and 0 <= nxt[1] < n
                    if nxt in seen or not in_bounds:
                        continue
                    if not self._occupied(nxt, obstacles, margin):
                        return nxt
                    seen.add(nxt)
                    queue.append(nxt)
        return None

    def _occupied(
        self,
        cell: tuple[int, int],
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> bool:
        return self._point_occupied(self._from_cell(cell), obstacles, margin)

    def _point_occupied(
        self,
        xy: np.ndarray,
        obstacles: list[tuple[float, float, float, float]],
        margin: float,
    ) -> bool:
        x, y = float(xy[0]), float(xy[1])
        if (
            x < self.world_min + margin
            or x > self.world_max - margin
            or y < self.world_min + margin
            or y > self.world_max - margin
        ):
            return True
        for ox, oy, width, height in obstacles:
            if (
                ox - margin <= x <= ox + width + margin
                and oy - margin <= y <= oy + height + margin
            ):
                return True
        return False

    def _to_cell(self, xy: np.ndarray, n: int) -> tuple[int, int]:
        clipped = np.clip(xy, self.world_min, self.world_max)
        idx = np.rint((clipped - self.world_min) / self.resolution).astype(int)
        idx = np.clip(idx, 0, n - 1)
        return int(idx[0]), int(idx[1])

    def _from_cell(self, cell: tuple[int, int]) -> np.ndarray:
        return np.asarray(
            [
                self.world_min + cell[0] * self.resolution,
                self.world_min + cell[1] * self.resolution,
            ],
            dtype=np.float32,
        )


def append_step(
    buffers: dict[str, list[np.ndarray]],
    *,
    frame: np.ndarray,
    info: dict,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_idx: int,
    image_size: int,
) -> None:
    buffers['pixels'].append(resize_image(frame, image_size))
    for key in ('state', 'proprio', 'goal_state', 'goal_proprio'):
        buffers[key].append(np.asarray(info[key], dtype=np.float32))
    buffers['action'].append(np.asarray(action, dtype=np.float32))
    buffers['reward'].append(np.asarray(reward, dtype=np.float32))
    buffers['terminated'].append(np.asarray(terminated, dtype=bool))
    buffers['truncated'].append(np.asarray(truncated, dtype=bool))
    buffers['step_idx'].append(np.asarray(step_idx, dtype=np.int32))


def choose_action(env: gym.Env, info: dict, policy: str) -> np.ndarray:
    if policy == 'random':
        return env.action_space.sample()

    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    robot_xy = np.asarray(info['proprio'][:2], dtype=np.float32)
    target_xy = np.asarray(info['goal_proprio'][:2], dtype=np.float32)
    action[:2] = np.clip(target_xy - robot_xy, low[:2], high[:2])
    return np.clip(action, low, high).astype(env.action_space.dtype)


def resize_image(frame: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image)


def save_gif(frames: np.ndarray, path: Path, fps: int) -> None:
    pil_frames = [Image.fromarray(frame) for frame in frames]
    stride = max(1, len(pil_frames) // 120)
    pil_frames = pil_frames[::stride]
    duration_ms = max(1, int(1000 / fps))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_trajectory_overlay(
    episode: dict[str, np.ndarray],
    path: Path,
) -> None:
    frame = Image.fromarray(episode['pixels'][-1]).convert('RGB')
    draw = ImageDraw.Draw(frame)
    width, height = frame.size

    def project(xy: np.ndarray) -> tuple[int, int]:
        x = float(np.clip(xy[0], 0.0, 2.5))
        y = float(np.clip(xy[1], 0.0, 2.5))
        px = int(round(x / 2.5 * (width - 1)))
        py = int(round((1.0 - y / 2.5) * (height - 1)))
        return px, py

    points = [project(xy) for xy in episode['proprio'][:, :2]]
    planned_path = episode.get('planned_path')
    if planned_path is not None and len(planned_path) > 1:
        planned_points = [project(xy) for xy in planned_path]
        draw.line(planned_points, fill=(255, 180, 20), width=2)

    if len(points) > 1:
        draw.line(points, fill=(20, 140, 255), width=3)

    start = points[0]
    end = points[-1]
    target_xy = (
        planned_path[-1]
        if planned_path is not None and len(planned_path) > 0
        else episode['goal_proprio'][0, :2]
    )
    target = project(target_xy)
    draw_marker(draw, start, fill=(30, 180, 70), radius=5)
    draw_marker(draw, end, fill=(30, 90, 255), radius=5)
    draw_marker(draw, target, fill=(230, 45, 45), radius=6)
    draw.text(
        (8, 8),
        'green=start  blue=actual  orange=A*  red=goal',
        fill=(0, 0, 0),
    )
    frame.save(path)


def save_trajectory_contact_sheet(
    episode_meta: list[dict],
    path: Path,
) -> None:
    episode_meta = [meta for meta in episode_meta if 'trajectory_png' in meta]
    if not episode_meta:
        return
    thumbs = [
        Image.open(meta['trajectory_png']).convert('RGB')
        for meta in episode_meta
    ]
    thumb_w, thumb_h = thumbs[0].size
    cols = min(5, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    title_h = 18
    sheet = Image.new(
        'RGB',
        (cols * thumb_w, rows * (thumb_h + title_h)),
        'white',
    )
    draw = ImageDraw.Draw(sheet)
    for idx, (thumb, meta) in enumerate(zip(thumbs, episode_meta)):
        row, col = divmod(idx, cols)
        x = col * thumb_w
        y = row * (thumb_h + title_h)
        label = (
            f"seed={meta['seed']} "
            f"T={meta['num_steps']} "
            f"term={int(meta['terminated'])}"
        )
        draw.text((x + 4, y + 2), label, fill=(0, 0, 0))
        sheet.paste(thumb, (x, y + title_h))
    sheet.save(path)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    radius: int,
) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


if __name__ == '__main__':
    main()
