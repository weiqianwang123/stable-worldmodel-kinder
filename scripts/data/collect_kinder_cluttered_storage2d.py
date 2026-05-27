"""Collect and visualize KinDER ClutteredStorage2D episodes."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-blocks', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=400)
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
        choices=('scripted', 'random'),
        default='scripted',
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
        default=Path('outputs/kinder_cluttered_storage2d_b1_episode'),
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

    ensure_cluttered_storage2d_registered()
    load_planner_symbols(args.kindergarden_home)

    env_id = f'swm/KinderClutteredStorage2D-b{args.num_blocks}-v0'
    make_kwargs: dict[str, Any] = {
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
            episode, meta = collect_episode(
                env,
                args,
                seed=seed,
                env_id=env_id,
                make_kwargs=make_kwargs,
            )
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

        meta.update({k: str(v) for k, v in paths.items()})
        episode_meta.append(meta)
        print(
            f'[{ep_idx + 1}/{len(seeds)}] seed={seed} '
            f"steps={meta['num_steps']} success={meta['success']} "
            f"terminated={meta['terminated']} truncated={meta['truncated']} "
            f"failure={meta['failure_reason']}",
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


def ensure_cluttered_storage2d_registered() -> None:
    try:
        import stable_worldmodel.envs  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if exc.name in {'gymnasium', 'numpy', 'PIL'}:
            raise
        print(
            'Warning: full stable_worldmodel env registration failed because '
            f'{exc.name!r} is missing; registering only KinDER '
            'ClutteredStorage2D for this smoke script.',
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

    mod = importlib.import_module(
        'stable_worldmodel.envs.kinder.cluttered_storage2d'
    )
    cls = mod.KinderClutteredStorage2D

    safe_register(
        id='swm/KinderClutteredStorage2D-v0',
        entry_point=cls,
    )
    for num_blocks in (1, 3, 7, 15):
        safe_register(
            id=f'swm/KinderClutteredStorage2D-b{num_blocks}-v0',
            entry_point=cls,
            kwargs={'num_blocks': num_blocks},
        )


def safe_register(id: str, **kwargs) -> None:
    if id not in registry:
        register(id=id, **kwargs)


PLANNER_SYMBOLS: dict[str, Any] = {}


def load_planner_symbols(kindergarden_home: str | None) -> None:
    if PLANNER_SYMBOLS:
        return
    from stable_worldmodel.envs.kinder._utils import ensure_kindergarden_on_path

    ensure_kindergarden_on_path(kindergarden_home)
    from kinder.envs.kinematic2d.structs import SE2Pose
    from kinder.envs.kinematic2d.utils import (
        crv_pose_plan_to_action_plan,
        get_suctioned_objects,
        get_tool_tip_position,
        is_inside_shelf,
        run_motion_planning_for_crv_robot,
    )
    from kinder.envs.utils import get_se2_pose

    PLANNER_SYMBOLS.update(
        {
            'SE2Pose': SE2Pose,
            'crv_pose_plan_to_action_plan': crv_pose_plan_to_action_plan,
            'get_se2_pose': get_se2_pose,
            'get_suctioned_objects': get_suctioned_objects,
            'get_tool_tip_position': get_tool_tip_position,
            'is_inside_shelf': is_inside_shelf,
            'run_motion_planning_for_crv_robot': run_motion_planning_for_crv_robot,
        }
    )


def collect_episode(
    env: gym.Env,
    args: argparse.Namespace,
    *,
    seed: int,
    env_id: str,
    make_kwargs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict]:
    obs, info = env.reset(seed=seed)
    del obs
    planner = None
    if args.policy == 'scripted':
        planner = ScriptedClutteredStorage2DPolicy(
            env,
            env_id=env_id,
            make_kwargs=make_kwargs,
            seed=seed,
        )

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

    failure_reason = None
    for step_idx in range(1, args.max_steps + 1):
        if planner is not None:
            action = planner.action(info)
            failure_reason = planner.failure_reason
            if action is None:
                break
        else:
            action = env.action_space.sample()

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
    if 'goal' in info:
        episode['goal_pixels'] = resize_image(info['goal'], args.image_size)

    terminated = bool(episode['terminated'][-1])
    truncated = bool(episode['truncated'][-1])
    success = terminated and not truncated
    if not success and failure_reason is None and planner is not None:
        failure_reason = 'scripted_plan_exhausted_without_success'

    meta = {
        'seed': int(seed),
        'num_steps': int(episode['pixels'].shape[0]),
        'success': bool(success),
        'terminated': terminated,
        'truncated': truncated,
        'failure_reason': failure_reason,
    }
    return episode, meta


@dataclass(frozen=True)
class PickCandidate:
    pre_pose: Any
    pre_arm: float
    grasp_arm: float


@dataclass(frozen=True)
class CarryCandidate:
    arm: float
    final_block_pose: Any
    pre_insert_block_pose: Any
    insert_distance: float


class ScriptedClutteredStorage2DPolicy:
    """Scripted b1 policy that validates candidate plans in a simulator."""

    def __init__(
        self,
        env: gym.Env,
        *,
        env_id: str,
        make_kwargs: dict[str, Any],
        seed: int,
    ) -> None:
        self.env = env
        self.env_id = env_id
        self.make_kwargs = {
            k: v for k, v in make_kwargs.items() if k != 'max_episode_steps'
        }
        self.seed = seed
        self._queue: list[np.ndarray] | None = None
        self.failure_reason: str | None = None

    def action(self, info: dict) -> np.ndarray | None:
        if self._queue is None:
            self._queue = self._build_plan(info)
        if not self._queue:
            return None
        return self._queue.pop(0)

    def _build_plan(self, info: dict) -> list[np.ndarray]:
        if getattr(self.env.unwrapped, 'num_blocks', None) != 1:
            self.failure_reason = 'scripted_policy_only_supports_b1'
            return []

        start_state = np.asarray(info['state'], dtype=np.float32).copy()
        sim = gym.make(self.env_id, **self.make_kwargs)
        try:
            sim.reset(seed=self.seed)
            sim.unwrapped.set_state(start_state)
            plan = self._search_plan(sim, start_state)
        finally:
            sim.close()
        return plan

    def _search_plan(self, sim: gym.Env, start_state: np.ndarray) -> list[np.ndarray]:
        sim.unwrapped.set_state(start_state)
        obs = start_state.copy()
        full_actions: list[np.ndarray] = []

        retract = self._arm_actions(sim, obs, self._min_arm(sim, obs), vac=0.0)
        obs, _, terminated, truncated, _ = self._run_actions(sim, obs, retract)
        if terminated or truncated:
            return retract
        full_actions.extend(retract)

        found_pick = False
        for pick_actions, attached_state in self._iter_pick_plans(sim, obs):
            found_pick = True
            carry_result = self._find_carry_plan(sim, attached_state)
            if carry_result is None:
                continue
            self.failure_reason = None
            return full_actions + pick_actions + carry_result

        self.failure_reason = (
            'carry_plan_failed' if found_pick else 'pick_plan_failed'
        )
        return []

    def _find_pick_plan(
        self,
        sim: gym.Env,
        start_state: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray] | None:
        return next(self._iter_pick_plans(sim, start_state), None)

    def _iter_pick_plans(
        self,
        sim: gym.Env,
        start_state: np.ndarray,
    ):
        for idx, candidate in enumerate(
            self._iter_pick_candidates(sim, start_state)
        ):
            sim.unwrapped.set_state(start_state)
            move_actions = self._plan_to_pose(
                sim,
                start_state,
                candidate.pre_pose,
                seed=self.seed * 1000 + idx,
                vacuum_while_moving=False,
            )
            if move_actions is None:
                continue
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, start_state, move_actions
            )
            if terminated or truncated:
                continue
            arm_actions = self._arm_actions(sim, obs, candidate.pre_arm, vac=0.0)
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, obs, arm_actions
            )
            if terminated or truncated:
                continue
            grasp_actions = self._arm_actions(
                sim, obs, candidate.grasp_arm, vac=0.0
            )
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, obs, grasp_actions
            )
            if terminated or truncated:
                continue
            vac_on = self._zero_action(sim)
            vac_on[4] = 1.0
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, obs, [vac_on]
            )
            if terminated or truncated:
                continue
            if self._is_block_suctioned(sim, obs):
                yield (
                    move_actions + arm_actions + grasp_actions + [vac_on],
                    obs.copy(),
                )

    def _find_carry_plan(
        self,
        sim: gym.Env,
        attached_state: np.ndarray,
    ) -> list[np.ndarray] | None:
        cand_idx = 0
        for candidate in self._iter_carry_candidates(sim, attached_state):
            sim.unwrapped.set_state(attached_state)
            arm_actions = self._arm_actions(
                sim, attached_state, candidate.arm, vac=1.0
            )
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, attached_state, arm_actions
            )
            if terminated:
                return arm_actions
            if truncated or not self._is_block_suctioned(sim, obs):
                continue

            pose = self._robot_pose_for_desired_block_pose(
                sim,
                obs,
                candidate.pre_insert_block_pose,
            )
            if pose is None or not self._pose_in_world(sim, obs, pose):
                continue

            move_actions = self._plan_to_pose(
                sim,
                obs,
                pose,
                seed=self.seed * 1000 + 500 + cand_idx,
                vacuum_while_moving=True,
                num_attempts=20,
                num_iters=350,
            )
            cand_idx += 1
            if move_actions is None:
                continue
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, obs, move_actions
            )
            if terminated:
                return arm_actions + move_actions
            if truncated or not self._is_block_suctioned(sim, obs):
                continue

            insert_actions = self._insert_actions(
                sim,
                candidate.insert_distance,
            )
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, obs, insert_actions
            )
            if terminated:
                return arm_actions + move_actions + insert_actions
            if truncated:
                continue

            vac_off = self._zero_action(sim)
            vac_off[4] = 0.0
            obs, _, terminated, truncated, _ = self._run_actions(
                sim, obs, [vac_off]
            )
            if terminated and not truncated:
                return arm_actions + move_actions + insert_actions + [vac_off]
        return None

    def _iter_pick_candidates(
        self,
        env: gym.Env,
        state_vec: np.ndarray,
    ):
        obj_state = self._devectorize(env, state_vec)
        robot, _, block = self._robot_shelf_block(obj_state)
        min_arm = self._min_arm(env, state_vec)
        max_arm = float(obj_state.get(robot, 'arm_length'))
        pre_arm = min_arm
        grasp_arms = [
            min(max_arm, min_arm + offset)
            for offset in (0.14, 0.18, 0.22, 0.28, 0.36, 0.46)
        ]
        grasp_arms = sorted({float(a) for a in grasp_arms if a > pre_arm + 1e-3})

        block_x = float(obj_state.get(block, 'x'))
        block_y = float(obj_state.get(block, 'y'))
        block_theta = float(obj_state.get(block, 'theta'))
        block_w = float(obj_state.get(block, 'width'))
        block_h = float(obj_state.get(block, 'height'))
        origin = np.asarray([block_x, block_y], dtype=np.float32)
        tangent = np.asarray(
            [math.cos(block_theta), math.sin(block_theta)],
            dtype=np.float32,
        )
        local_y = np.asarray(
            [-math.sin(block_theta), math.cos(block_theta)],
            dtype=np.float32,
        )
        gripper_width = float(obj_state.get(robot, 'gripper_width'))
        suction_overlap = min(0.25 * block_h, 0.5 * gripper_width)
        tangent_offsets = [0.0, -0.2 * block_w, 0.2 * block_w]

        robot_xy = np.asarray(
            [obj_state.get(robot, 'x'), obj_state.get(robot, 'y')],
            dtype=np.float32,
        )
        candidates: list[tuple[float, PickCandidate]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for side in (1.0, -1.0):
            face_out = side * local_y
            # Approach the long face from outside, so the gripper points inward.
            theta = self._wrap_angle(math.atan2(-face_out[1], -face_out[0]))
            face_base = origin + (block_h if side > 0.0 else 0.0) * local_y
            for tangent_offset in tangent_offsets:
                suction_target = (
                    face_base
                    + (0.5 * block_w + tangent_offset) * tangent
                    - suction_overlap * face_out
                )
                for grasp_arm in grasp_arms:
                    base_xy = suction_target - self._suction_offset(
                        obj_state,
                        robot,
                        theta,
                        grasp_arm,
                    )
                    pose_key = (
                        int(round(float(base_xy[0]) * 100)),
                        int(round(float(base_xy[1]) * 100)),
                        int(round(float(theta) * 100)),
                        int(round(float(grasp_arm) * 100)),
                    )
                    if pose_key in seen:
                        continue
                    seen.add(pose_key)
                    pre_pose = self._se2_pose(base_xy[0], base_xy[1], theta)
                    if not self._pose_in_world(env, state_vec, pre_pose):
                        continue

                    test_state = obj_state.copy()
                    test_robot = self._object_by_name(test_state, 'robot')
                    test_state.set(test_robot, 'x', float(base_xy[0]))
                    test_state.set(test_robot, 'y', float(base_xy[1]))
                    test_state.set(test_robot, 'theta', float(theta))
                    test_state.set(test_robot, 'arm_joint', float(grasp_arm))
                    test_state.set(test_robot, 'vacuum', 1.0)
                    if not self._state_suctions_block(test_state):
                        continue
                    score = float(np.linalg.norm(base_xy - robot_xy))
                    candidates.append(
                        (
                            score,
                            PickCandidate(
                                pre_pose=pre_pose,
                                pre_arm=float(pre_arm),
                                grasp_arm=float(grasp_arm),
                            ),
                        )
                    )

        for _, candidate in sorted(candidates, key=lambda item: item[0]):
            yield candidate

    def _iter_carry_candidates(
        self,
        env: gym.Env,
        state_vec: np.ndarray,
    ):
        obj_state = self._devectorize(env, state_vec)
        robot, shelf, block = self._robot_shelf_block(obj_state)
        suctioned = self._suctioned_objects(obj_state, robot)
        block_items = [
            (obj, rel) for obj, rel in suctioned if obj.name == 'block0'
        ]
        if not block_items:
            return
        _, gripper_to_obj = block_items[0]
        min_arm = self._min_arm(env, state_vec)
        max_arm = float(obj_state.get(robot, 'arm_length'))
        current_arm = float(obj_state.get(robot, 'arm_joint'))
        arms = [current_arm, min_arm, 0.35, 0.45, 0.55, 0.65]
        arms = sorted({float(np.clip(a, min_arm, max_arm)) for a in arms})

        sx = float(obj_state.get(shelf, 'x1'))
        sy = float(obj_state.get(shelf, 'y1'))
        sw = float(obj_state.get(shelf, 'width1'))
        sh = float(obj_state.get(shelf, 'height1'))
        robot_theta = math.pi / 2
        final_block_theta = self._wrap_angle(robot_theta + gripper_to_obj.theta)
        block_w = float(obj_state.get(block, 'width'))
        block_h = float(obj_state.get(block, 'height'))
        center_xs = [sx + frac * sw for frac in (0.46, 0.5, 0.54)]
        center_ys = [
            sy + frac * sh
            for frac in (0.22, 0.32, 0.42, 0.52, 0.62, 0.72)
        ]
        insert_distances = [0.14, 0.18, 0.22, 0.26]
        insert_dir = np.asarray(
            [math.cos(robot_theta), math.sin(robot_theta)],
            dtype=np.float32,
        )

        seen: set[tuple[int, int, int, int, int]] = set()
        for arm in arms:
            for center_x in center_xs:
                for center_y in center_ys:
                    final_pose = self._block_pose_from_center(
                        np.asarray([center_x, center_y], dtype=np.float32),
                        final_block_theta,
                        block_w,
                        block_h,
                    )
                    if not self._block_inside_shelf(
                        obj_state, block, shelf, final_pose
                    ):
                        continue
                    for distance in insert_distances:
                        if arm + distance > max_arm + 1e-6:
                            continue
                        pre_center = (
                            np.asarray([center_x, center_y], dtype=np.float32)
                            - float(distance) * insert_dir
                        )
                        pre_pose = self._block_pose_from_center(
                            pre_center,
                            final_block_theta,
                            block_w,
                            block_h,
                        )
                        if not self._preinsert_below_shelf(
                            np.asarray(
                                [pre_pose.x, pre_pose.y],
                                dtype=np.float32,
                            ),
                            final_block_theta,
                            block_w,
                            block_h,
                            sy,
                        ):
                            continue
                        key = (
                            int(round(arm * 100)),
                            int(round(float(final_block_theta) * 100)),
                            int(round(center_x * 100)),
                            int(round(center_y * 100)),
                            int(round(float(distance) * 100)),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        yield CarryCandidate(
                            arm=arm,
                            final_block_pose=final_pose,
                            pre_insert_block_pose=pre_pose,
                            insert_distance=float(distance),
                        )

    def _preinsert_below_shelf(
        self,
        block_xy: np.ndarray,
        block_theta: float,
        block_w: float,
        block_h: float,
        shelf_y: float,
    ) -> bool:
        vertices = self._rect_vertices(block_xy, block_theta, block_w, block_h)
        return float(np.max(vertices[:, 1])) < shelf_y - 1e-3

    def _block_pose_from_center(
        self,
        center_xy: np.ndarray,
        theta: float,
        width: float,
        height: float,
    ):
        local_center = np.asarray([0.5 * width, 0.5 * height], dtype=np.float32)
        rotation = np.asarray(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float32,
        )
        xy = np.asarray(center_xy, dtype=np.float32) - rotation @ local_center
        return self._se2_pose(float(xy[0]), float(xy[1]), theta)

    def _rect_vertices(
        self,
        xy: np.ndarray,
        theta: float,
        width: float,
        height: float,
    ) -> np.ndarray:
        local = np.asarray(
            [
                [0.0, 0.0],
                [width, 0.0],
                [width, height],
                [0.0, height],
            ],
            dtype=np.float32,
        )
        rotation = np.asarray(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float32,
        )
        return np.asarray(xy, dtype=np.float32) + local @ rotation.T

    def _insert_actions(
        self,
        env: gym.Env,
        insert_distance: float,
    ) -> list[np.ndarray]:
        if insert_distance <= 1e-6:
            return []
        max_step = float(env.action_space.high[3])
        num_steps = max(1, int(np.ceil(insert_distance / max_step)))
        step = insert_distance / num_steps
        actions = []
        for _ in range(num_steps):
            action = self._zero_action(env)
            action[3] = step
            action[4] = 1.0
            actions.append(self._as_action(env, action))
        return actions

    def _block_inside_shelf(self, obj_state, block, shelf, pose) -> bool:
        test_state = obj_state.copy()
        test_block = self._object_by_name(test_state, block.name)
        test_shelf = self._object_by_name(test_state, shelf.name)
        test_state.set(test_block, 'x', float(pose.x))
        test_state.set(test_block, 'y', float(pose.y))
        test_state.set(test_block, 'theta', float(pose.theta))
        return bool(
            PLANNER_SYMBOLS['is_inside_shelf'](
                test_state,
                test_block,
                test_shelf,
                {},
            )
        )

    def _robot_pose_for_desired_block_pose(
        self,
        env: gym.Env,
        state_vec: np.ndarray,
        desired_block_pose,
    ):
        obj_state = self._devectorize(env, state_vec)
        robot, _, _ = self._robot_shelf_block(obj_state)
        suctioned = self._suctioned_objects(obj_state, robot)
        block_items = [(obj, rel) for obj, rel in suctioned if obj.name == 'block0']
        if not block_items:
            return None
        _, gripper_to_obj = block_items[0]
        desired_gripper = desired_block_pose * gripper_to_obj.inverse
        arm = float(obj_state.get(robot, 'arm_joint'))
        offset = self._tip_offset(
            obj_state,
            robot,
            desired_gripper.theta,
            arm,
        )
        base_xy = (
            np.asarray([desired_gripper.x, desired_gripper.y], dtype=np.float32)
            - offset
        )
        return self._se2_pose(
            float(base_xy[0]),
            float(base_xy[1]),
            float(desired_gripper.theta),
        )

    def _plan_to_pose(
        self,
        env: gym.Env,
        state_vec: np.ndarray,
        pose,
        *,
        seed: int,
        vacuum_while_moving: bool,
        num_attempts: int = 12,
        num_iters: int = 250,
    ) -> list[np.ndarray] | None:
        obj_state = self._devectorize(env, state_vec)
        robot, _, _ = self._robot_shelf_block(obj_state)
        plan = PLANNER_SYMBOLS['run_motion_planning_for_crv_robot'](
            obj_state,
            robot,
            pose,
            env.action_space,
            seed=seed,
            num_attempts=num_attempts,
            num_iters=num_iters,
            smooth_amt=30,
        )
        if plan is None:
            return None
        actions = PLANNER_SYMBOLS['crv_pose_plan_to_action_plan'](
            plan,
            env.action_space,
            vacuum_while_moving=vacuum_while_moving,
        )
        return [self._as_action(env, action) for action in actions]

    def _run_actions(
        self,
        env: gym.Env,
        obs: np.ndarray,
        actions: list[np.ndarray],
    ):
        info: dict = {}
        reward = 0.0
        terminated = False
        truncated = False
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(
                self._as_action(env, action)
            )
            if terminated or truncated:
                break
        return obs, reward, terminated, truncated, info

    def _arm_actions(
        self,
        env: gym.Env,
        state_vec: np.ndarray,
        target_arm: float,
        *,
        vac: float,
    ) -> list[np.ndarray]:
        obj_state = self._devectorize(env, state_vec)
        robot, _, _ = self._robot_shelf_block(obj_state)
        arm = float(obj_state.get(robot, 'arm_joint'))
        target_arm = float(
            np.clip(
                target_arm,
                float(obj_state.get(robot, 'base_radius')),
                float(obj_state.get(robot, 'arm_length')),
            )
        )
        actions = []
        for _ in range(20):
            diff = target_arm - arm
            if abs(diff) <= 1e-3:
                break
            action = self._zero_action(env)
            action[3] = np.clip(
                diff,
                float(env.action_space.low[3]),
                float(env.action_space.high[3]),
            )
            action[4] = vac
            actions.append(action)
            arm += float(action[3])
        if not actions and vac > 0.5:
            action = self._zero_action(env)
            action[4] = vac
            actions.append(action)
        return actions

    def _state_suctions_block(self, obj_state) -> bool:
        robot = self._object_by_name(obj_state, 'robot')
        return any(
            obj.name == 'block0'
            for obj, _ in self._suctioned_objects(obj_state, robot)
        )

    def _is_block_suctioned(self, env: gym.Env, state_vec: np.ndarray) -> bool:
        obj_state = self._devectorize(env, state_vec)
        return self._state_suctions_block(obj_state)

    def _suctioned_objects(self, obj_state, robot):
        return PLANNER_SYMBOLS['get_suctioned_objects'](obj_state, robot)

    def _tip_offset(self, obj_state, robot, theta: float, arm: float) -> np.ndarray:
        # The CRV gripper center is at ``arm`` along heading theta, and the
        # tool tip is the gripper's forward edge.
        gripper_width = float(obj_state.get(robot, 'gripper_width'))
        reach = float(arm) + 0.5 * gripper_width
        return np.asarray(
            [reach * math.cos(theta), reach * math.sin(theta)],
            dtype=np.float32,
        )

    def _suction_offset(
        self,
        obj_state,
        robot,
        theta: float,
        arm: float,
    ) -> np.ndarray:
        gripper_width = float(obj_state.get(robot, 'gripper_width'))
        reach = float(arm) + 1.5 * gripper_width
        return np.asarray(
            [reach * math.cos(theta), reach * math.sin(theta)],
            dtype=np.float32,
        )

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    def _block_target_points(self, obj_state, block) -> list[np.ndarray]:
        x = float(obj_state.get(block, 'x'))
        y = float(obj_state.get(block, 'y'))
        w = float(obj_state.get(block, 'width'))
        h = float(obj_state.get(block, 'height'))
        return [
            np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=np.float32),
            np.asarray([x + 0.25 * w, y + 0.5 * h], dtype=np.float32),
            np.asarray([x + 0.75 * w, y + 0.5 * h], dtype=np.float32),
            np.asarray([x + 0.5 * w, y + 0.25 * h], dtype=np.float32),
            np.asarray([x + 0.5 * w, y + 0.75 * h], dtype=np.float32),
        ]

    def _pose_in_world(self, env: gym.Env, state_vec: np.ndarray, pose) -> bool:
        obj_state = self._devectorize(env, state_vec)
        robot, shelf, _ = self._robot_shelf_block(obj_state)
        radius = float(obj_state.get(robot, 'base_radius'))
        cfg = env.unwrapped.kinder_env._object_centric_env.config
        min_x = float(cfg.world_min_x) + radius
        max_x = float(cfg.world_max_x) - radius
        min_y = float(cfg.world_min_y) + radius
        max_y = min(
            float(cfg.world_max_y) - radius,
            float(obj_state.get(shelf, 'y1')) - 0.25 * radius,
        )
        return min_x <= pose.x <= max_x and min_y <= pose.y <= max_y

    def _min_arm(self, env: gym.Env, state_vec: np.ndarray) -> float:
        obj_state = self._devectorize(env, state_vec)
        robot, _, _ = self._robot_shelf_block(obj_state)
        return float(obj_state.get(robot, 'base_radius'))

    def _devectorize(self, env: gym.Env, state_vec: np.ndarray):
        return env.unwrapped.kinder_env.observation_space.devectorize(
            np.asarray(state_vec, dtype=np.float32)
        )

    @staticmethod
    def _object_by_name(obj_state, name: str):
        for obj in obj_state:
            if obj.name == name:
                return obj
        raise KeyError(name)

    def _robot_shelf_block(self, obj_state):
        return (
            self._object_by_name(obj_state, 'robot'),
            self._object_by_name(obj_state, 'shelf'),
            self._object_by_name(obj_state, 'block0'),
        )

    def _se2_pose(self, x: float, y: float, theta: float):
        return PLANNER_SYMBOLS['SE2Pose'](
            float(x),
            float(y),
            float(theta),
        )

    def _zero_action(self, env: gym.Env) -> np.ndarray:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        return self._as_action(env, action)

    @staticmethod
    def _as_action(env: gym.Env, action) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32),
            np.asarray(env.action_space.low, dtype=np.float32),
            np.asarray(env.action_space.high, dtype=np.float32),
        ).astype(env.action_space.dtype)


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
        x = float(np.clip(xy[0], 0.0, 5.0))
        y = float(np.clip(xy[1], 0.0, 3.0))
        px = int(round(x / 5.0 * (width - 1)))
        py = int(round((1.0 - y / 3.0) * (height - 1)))
        return px, py

    robot_points = [project(xy) for xy in episode['proprio'][:, :2]]
    if len(robot_points) > 1:
        draw.line(robot_points, fill=(20, 140, 255), width=3)

    state = episode['state']
    if state.shape[1] >= 38:
        block_points = [project(step[28:30]) for step in state]
        if len(block_points) > 1:
            draw.line(block_points, fill=(255, 140, 20), width=2)
        draw_marker(draw, block_points[0], fill=(220, 120, 20), radius=5)
        draw_marker(draw, block_points[-1], fill=(230, 45, 45), radius=5)

    draw_marker(draw, robot_points[0], fill=(30, 180, 70), radius=5)
    draw_marker(draw, robot_points[-1], fill=(30, 90, 255), radius=5)
    draw.text(
        (8, 8),
        'green=robot start  blue=robot path  orange=block  red=block end',
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
            f"ok={int(meta['success'])}"
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
