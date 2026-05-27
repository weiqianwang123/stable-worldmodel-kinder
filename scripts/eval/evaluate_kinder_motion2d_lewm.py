"""Evaluate a trained LeWM on KinDER Motion2D seeds.

This script intentionally keeps the first world-model metric simple and
extensible: collect held-out Motion2D episodes, encode the true frames, then
measure latent prediction error when rolling the model forward with the true
actions. LeWM has no image decoder, so this tests dynamics in representation
space instead of reconstructed pixels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.collect_kinder_motion2d import (  # noqa: E402
    AStarMotion2DPolicy,
    FolderDatasetBuilder,
    collect_episode,
    ensure_motion2d_registered,
    save_episode_artifacts,
    save_trajectory_contact_sheet,
)
import stable_worldmodel as swm  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--num-passages', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--policy', choices=('astar', 'greedy', 'random'), default='astar')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--artifact-limit', type=int, default=None)
    parser.add_argument('--no-gif', action='store_true')
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('outputs/eval_kinder_motion2d_lewm_seeds_1000_1002'),
    )
    parser.add_argument('--dataset-dir', type=Path, default=None)
    parser.add_argument('--kindergarden-home', type=str, default=None)

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            'LeWM .pt checkpoint. If omitted, use latest weights_epoch_*.pt '
            'under the default checkpoint run.'
        ),
    )
    parser.add_argument(
        '--checkpoint-run',
        type=str,
        default='lewm_kinder_motion2d_500eps',
    )
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--skip-model', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--history-size', type=int, default=3)
    parser.add_argument('--rollout-horizon', type=int, default=32)
    parser.add_argument('--eval-stride', type=int, default=8)
    parser.add_argument('--skip-latent', action='store_true')

    parser.add_argument('--run-planning', action='store_true')
    parser.add_argument('--planning-start-step', type=int, default=0)
    parser.add_argument('--planning-goal-offset', type=int, default=None)
    parser.add_argument('--planning-eval-budget', type=int, default=100)
    parser.add_argument('--planning-horizon', type=int, default=5)
    parser.add_argument('--planning-receding-horizon', type=int, default=1)
    parser.add_argument('--planning-action-block', type=int, default=1)
    parser.add_argument('--no-planning-warm-start', action='store_true')
    parser.add_argument('--solver', choices=('cem', 'predictive_sampling'), default='cem')
    parser.add_argument('--planning-batch-size', type=int, default=1)
    parser.add_argument('--planning-num-samples', type=int, default=64)
    parser.add_argument('--planning-cem-steps', type=int, default=5)
    parser.add_argument('--planning-topk', type=int, default=8)
    parser.add_argument('--planning-var-scale', type=float, default=1.0)
    parser.add_argument('--planning-noise-scale', type=float, default=1.0)
    parser.add_argument('--no-planning-video', action='store_true')
    parser.add_argument(
        '--stats-dataset',
        type=Path,
        default=Path('outputs/kinder_motion2d_500eps/dataset_folder'),
        help='Dataset used to fit action normalization for planning.',
    )

    parser.add_argument('--grid-resolution', type=float, default=0.025)
    parser.add_argument('--clearance', type=float, default=0.005)
    parser.add_argument('--waypoint-tolerance', type=float, default=0.015)
    parser.add_argument('--heading-tolerance', type=float, default=0.18)
    parser.add_argument('--rotate-in-place-threshold', type=float, default=0.45)
    parser.add_argument('--passage-approach-margin', type=float, default=0.14)
    parser.add_argument('--disable-passage-waypoints', action='store_true')
    parser.add_argument('--plot-oracle-cost', action='store_true')
    parser.add_argument('--oracle-cost-num-seeds', type=int, default=20)
    parser.add_argument('--oracle-cost-start-seed', type=int, default=1000)
    parser.add_argument('--oracle-cost-goal-offset', type=int, default=None)
    args = parser.parse_args()
    if args.seeds is None:
        if args.plot_oracle_cost:
            args.seeds = list(
                range(
                    args.oracle_cost_start_seed,
                    args.oracle_cost_start_seed + args.oracle_cost_num_seeds,
                )
            )
        else:
            args.seeds = [1000, 1001, 1002]
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    episodes, episode_meta, dataset_dir = collect_eval_data(args)
    metadata: dict[str, Any] = {
        'env_id': f'swm/KinderMotion2D-p{args.num_passages}-v0',
        'seeds': [int(seed) for seed in args.seeds],
        'policy': args.policy,
        'num_episodes': len(episodes),
        'dataset_dir': str(dataset_dir),
        'summary_png': str(args.out_dir / 'trajectories.png'),
        'episodes': episode_meta,
    }

    if not args.skip_model and not args.skip_latent:
        checkpoint = resolve_checkpoint(
            args.checkpoint,
            run_name=args.checkpoint_run,
            epoch=args.epoch,
        )
        metrics = evaluate_latent_rollout(args, episodes, checkpoint)
        metadata['checkpoint'] = str(checkpoint)
        metadata['wm_eval'] = metrics
        save_metric_plot(metrics, args.out_dir / 'latent_rollout_mse.png')
        save_metric_csv(metrics, args.out_dir / 'latent_rollout_mse.csv')

    if args.run_planning:
        checkpoint = resolve_checkpoint(
            args.checkpoint,
            run_name=args.checkpoint_run,
            epoch=args.epoch,
        )
        planning = evaluate_planning(args, checkpoint, dataset_dir, episodes)
        metadata['checkpoint'] = str(checkpoint)
        metadata['planning_eval'] = planning

    if args.plot_oracle_cost:
        oracle_plot = save_oracle_cost_plot(args, episodes)
        metadata['oracle_cost_debug'] = oracle_plot

    metadata_path = args.out_dir / 'metrics.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


def collect_eval_data(
    args: argparse.Namespace,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]], Path]:
    ensure_motion2d_registered()
    env_id = f'swm/KinderMotion2D-p{args.num_passages}-v0'
    make_kwargs: dict[str, Any] = {
        'max_episode_steps': args.max_steps,
        'render_mode': 'rgb_array',
    }
    if args.kindergarden_home is not None:
        make_kwargs['kindergarden_home'] = args.kindergarden_home

    collect_args = SimpleNamespace(**vars(args))
    dataset_dir = args.dataset_dir or args.out_dir / 'dataset_folder'
    writer = FolderDatasetBuilder(dataset_dir)
    episodes: list[dict[str, np.ndarray]] = []
    episode_meta: list[dict[str, Any]] = []
    for ep_idx, seed in enumerate(args.seeds):
        env = gym.make(env_id, **make_kwargs)
        env.action_space.seed(seed)
        try:
            episode = collect_episode(env, collect_args, seed=seed)
        finally:
            env.close()

        writer.add_episode(episode)
        episodes.append(episode)

        save_artifacts = args.artifact_limit is None or ep_idx < args.artifact_limit
        paths: dict[str, Path] = {}
        if save_artifacts:
            seed_dir = (
                args.out_dir
                if len(args.seeds) == 1
                else args.out_dir / f'seed_{seed}'
            )
            seed_dir.mkdir(parents=True, exist_ok=True)
            paths = save_episode_artifacts(
                episode,
                seed_dir,
                fps=args.fps,
                save_gif_artifact=not args.no_gif,
            )

        meta = {
            'seed': int(seed),
            'num_steps': int(episode['pixels'].shape[0]),
            'terminated': bool(episode['terminated'][-1]),
            'truncated': bool(episode['truncated'][-1]),
            **{k: str(v) for k, v in paths.items()},
        }
        episode_meta.append(meta)
        print(
            f'[{ep_idx + 1}/{len(args.seeds)}] seed={seed} '
            f"steps={meta['num_steps']} terminated={meta['terminated']} "
            f"truncated={meta['truncated']}",
            flush=True,
        )

    writer.close()
    save_trajectory_contact_sheet(episode_meta, args.out_dir / 'trajectories.png')
    return episodes, episode_meta, dataset_dir


def resolve_checkpoint(
    checkpoint: str | None,
    *,
    run_name: str,
    epoch: int | None,
) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if path.exists():
            return path.resolve()
        cache_path = swm.data.utils.get_cache_dir(sub_folder='checkpoints') / checkpoint
        if cache_path.exists():
            return cache_path.resolve()
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    run_dir = find_checkpoint_run(run_name)
    weights = sorted(run_dir.glob('weights_epoch_*.pt'), key=epoch_from_path)
    if epoch is not None:
        weights = [path for path in weights if epoch_from_path(path) == epoch]
    if not weights:
        raise FileNotFoundError(f'No weights_epoch_*.pt found in {run_dir}')
    return weights[-1].resolve()


def find_checkpoint_run(run_name: str) -> Path:
    roots = [
        ROOT / 'outputs' / 'stablewm_cache' / 'checkpoints',
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        path = (root / run_name).resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            candidates.append(path)
    if not candidates:
        searched = ', '.join(str((root / run_name).resolve()) for root in roots)
        raise FileNotFoundError(f'Default checkpoint run not found. Searched: {searched}')
    return max(candidates, key=latest_epoch_in_dir)


def latest_epoch_in_dir(path: Path) -> int:
    weights = list(path.glob('weights_epoch_*.pt'))
    return max((epoch_from_path(weight) for weight in weights), default=-1)


def epoch_from_path(path: Path) -> int:
    match = re.search(r'weights_epoch_(\d+)\.pt$', path.name)
    return int(match.group(1)) if match else -1


def evaluate_planning(
    args: argparse.Namespace,
    checkpoint: Path,
    dataset_dir: Path,
    episodes: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    if args.skip_model:
        raise ValueError('--run-planning requires a model; remove --skip-model')

    import stable_worldmodel.envs  # noqa: F401

    device = choose_device(args.device)
    model = load_lewm_checkpoint(checkpoint).to(device).eval()
    model.requires_grad_(False)

    eval_dataset = swm.data.load_dataset(
        str(dataset_dir),
        frameskip=1,
        num_steps=1,
        keys_to_load=['pixels', 'action', 'proprio', 'state'],
        keys_to_cache=['action', 'proprio', 'state'],
    )
    stats_dataset = swm.data.load_dataset(
        str(args.stats_dataset),
        frameskip=1,
        num_steps=1,
        keys_to_load=['action'],
        keys_to_cache=['action'],
    )
    process = {'action': ColumnStandardizer.fit(stats_dataset.get_col_data('action'))}
    transform = {
        'pixels': make_policy_image_transform(args.image_size),
        'goal': make_policy_image_transform(args.image_size),
    }

    if args.solver == 'cem':
        solver = swm.solver.CEMSolver(
            model=model,
            batch_size=args.planning_batch_size,
            num_samples=args.planning_num_samples,
            var_scale=args.planning_var_scale,
            n_steps=args.planning_cem_steps,
            topk=args.planning_topk,
            device=device,
            seed=3072,
        )
    else:
        solver = swm.solver.PredictiveSamplingSolver(
            model=model,
            batch_size=args.planning_batch_size,
            num_samples=args.planning_num_samples,
            noise_scale=args.planning_noise_scale,
            device=device,
            seed=3072,
        )

    config = swm.PlanConfig(
        horizon=args.planning_horizon,
        receding_horizon=args.planning_receding_horizon,
        action_block=args.planning_action_block,
        history_len=1,
        warm_start=not args.no_planning_warm_start,
    )
    policy = ClippedWorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )

    goal_offset = resolve_planning_goal_offset(args, episodes)
    start_steps = [int(args.planning_start_step)] * len(episodes)
    episodes_idx = list(range(len(episodes)))
    video_dir = None if args.no_planning_video else args.out_dir / 'planning_videos'

    world = swm.World(
        f'swm/KinderMotion2D-p{args.num_passages}-v0',
        num_envs=len(episodes),
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=args.planning_eval_budget,
    )
    try:
        world.set_policy(policy)
        results = world.evaluate(
            dataset=eval_dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset=goal_offset,
            eval_budget=args.planning_eval_budget,
            callables=[
                {
                    'method': '_set_state',
                    'args': {'state': {'value': 'state'}},
                },
                {
                    'method': '_set_goal_state',
                    'args': {'goal_state': {'value': 'goal_state'}},
                },
            ],
            video=video_dir,
        )
        final_xy = np.asarray(world.infos['proprio'][:, 0, :2], dtype=np.float32)
        goal_xy = np.asarray(world.infos['goal_proprio'][:, 0, :2], dtype=np.float32)
        final_dist = np.linalg.norm(final_xy - goal_xy, axis=1)
    finally:
        world.close()

    planning = {
        'metric': 'mpc_success_from_eval_dataset',
        'solver': args.solver,
        'device': device,
        'goal_offset': int(goal_offset),
        'eval_budget': int(args.planning_eval_budget),
        'plan_config': {
            'horizon': int(args.planning_horizon),
            'receding_horizon': int(args.planning_receding_horizon),
            'action_block': int(args.planning_action_block),
            'warm_start': not args.no_planning_warm_start,
        },
        'solver_config': {
            'num_samples': int(args.planning_num_samples),
            'cem_steps': int(args.planning_cem_steps),
            'topk': int(args.planning_topk),
            'var_scale': float(args.planning_var_scale),
            'noise_scale': float(args.planning_noise_scale),
        },
        'success_rate': float(results['success_rate']),
        'episode_successes': [bool(x) for x in results['episode_successes']],
        'final_goal_distances': final_dist.astype(float).tolist(),
        'mean_final_goal_distance': float(final_dist.mean()),
        'video_dir': None if video_dir is None else str(video_dir),
    }
    (args.out_dir / 'planning_metrics.json').write_text(
        json.dumps(planning, indent=2) + '\n'
    )
    return planning


def resolve_planning_goal_offset(
    args: argparse.Namespace,
    episodes: list[dict[str, np.ndarray]],
) -> int:
    if args.planning_goal_offset is not None:
        return int(args.planning_goal_offset)
    min_len = min(int(ep['pixels'].shape[0]) for ep in episodes)
    max_offset = max(1, min_len - int(args.planning_start_step) - 2)
    return int(min(50, max_offset))


def save_oracle_cost_plot(
    args: argparse.Namespace,
    episodes: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    env_id = f'swm/KinderMotion2D-p{args.num_passages}-v0'
    make_kwargs: dict[str, Any] = {'render_mode': 'rgb_array'}
    if args.kindergarden_home is not None:
        make_kwargs['kindergarden_home'] = args.kindergarden_home

    goal_offset = resolve_oracle_cost_goal_offset(args, episodes)
    out_png = args.out_dir / 'oracle_cost_curves.png'
    out_csv = args.out_dir / 'oracle_cost_curves.csv'

    rows = ['seed,step,cost']
    curves: list[tuple[int, np.ndarray]] = []
    env = gym.make(env_id, **make_kwargs)
    try:
        for ep_idx, episode in enumerate(episodes):
            seed = int(args.seeds[ep_idx])
            goal_xy = oracle_goal_xy(args, episode, goal_offset)
            costs = []
            for step_idx, state in enumerate(episode['state']):
                cost = oracle_motion2d_cost(env, args, state, goal_xy)
                costs.append(cost)
                rows.append(f'{seed},{step_idx},{cost}')
            curves.append((seed, np.asarray(costs, dtype=np.float32)))
    finally:
        env.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('tab20')
    for idx, (seed, costs) in enumerate(curves):
        ax.plot(
            np.arange(len(costs)),
            costs,
            color=cmap(idx % 20),
            alpha=0.85,
            linewidth=1.8,
            label=str(seed),
        )
    ax.set_xlabel('step')
    ax.set_ylabel('oracle cost-to-go')
    title_goal = (
        'episode target' if goal_offset is None else f'goal_offset={goal_offset}'
    )
    ax.set_title(f'Motion2D oracle planner cost curves ({title_goal})')
    ax.grid(True, alpha=0.25)
    if len(curves) <= 20:
        ax.legend(title='seed', ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    out_csv.write_text('\n'.join(rows) + '\n')

    return {
        'metric': 'oracle_motion2d_planner_cost_to_goal',
        'goal_offset': goal_offset,
        'num_seeds': len(curves),
        'plot': str(out_png),
        'csv': str(out_csv),
    }


def resolve_oracle_cost_goal_offset(
    args: argparse.Namespace,
    episodes: list[dict[str, np.ndarray]],
) -> int | None:
    if args.oracle_cost_goal_offset is not None:
        return int(args.oracle_cost_goal_offset)
    if args.run_planning:
        return resolve_planning_goal_offset(args, episodes)
    if args.planning_goal_offset is not None:
        return int(args.planning_goal_offset)
    return None


def oracle_goal_xy(
    args: argparse.Namespace,
    episode: dict[str, np.ndarray],
    goal_offset: int | None,
) -> np.ndarray:
    if goal_offset is None:
        return np.asarray(episode['goal_proprio'][0, :2], dtype=np.float32)
    goal_step = min(
        int(args.planning_start_step) + int(goal_offset),
        int(episode['proprio'].shape[0]) - 1,
    )
    return np.asarray(episode['proprio'][goal_step, :2], dtype=np.float32)


def oracle_motion2d_cost(
    env: gym.Env,
    args: argparse.Namespace,
    state: np.ndarray,
    goal_xy: np.ndarray,
) -> float:
    state_with_goal = motion2d_state_with_target_xy(env, state, goal_xy)
    planner = AStarMotion2DPolicy(env, args)
    try:
        planner.reset({'state': state_with_goal})
    except RuntimeError:
        return float('nan')
    if planner.path is None or len(planner.path) == 0:
        return float('nan')
    points = np.asarray(planner.path, dtype=np.float32)
    if len(points) == 1:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def motion2d_state_with_target_xy(
    env: gym.Env,
    state: np.ndarray,
    goal_xy: np.ndarray,
) -> np.ndarray:
    obj_state = env.unwrapped.kinder_env.observation_space.devectorize(
        np.asarray(state, dtype=np.float32).copy()
    )
    target = object_by_name(obj_state, 'target_region')
    obj_state.set(target, 'x', float(goal_xy[0]))
    obj_state.set(target, 'y', float(goal_xy[1]))
    return np.asarray(
        env.unwrapped.kinder_env.observation_space.vectorize(obj_state),
        dtype=np.float32,
    )


def object_by_name(obj_state, name: str):
    for obj in obj_state:
        if obj.name == name:
            return obj
    raise KeyError(name)


class ColumnStandardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        arr = arr[~np.isnan(arr).any(axis=1)]
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean, scale)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (np.asarray(data, dtype=np.float32) - self.mean) / self.scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float32) * self.scale + self.mean


def make_policy_image_transform(image_size: int):
    def transform(image) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
        if tensor.ndim != 3:
            raise ValueError(f'Expected CHW image tensor, got shape {tensor.shape}')
        if tensor.shape[-2:] != (image_size, image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        mean = IMAGENET_MEAN.reshape(3, 1, 1).to(tensor)
        std = IMAGENET_STD.reshape(3, 1, 1).to(tensor)
        return (tensor - mean) / std

    return transform


class ClippedWorldModelPolicy(swm.policy.WorldModelPolicy):
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        action = super().get_action(info_dict, **kwargs)
        low = np.asarray(self.env.action_space.low, dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        return np.clip(action, low, high).astype(self.env.action_space.dtype)


@torch.no_grad()
def evaluate_latent_rollout(
    args: argparse.Namespace,
    episodes: list[dict[str, np.ndarray]],
    checkpoint: Path,
) -> dict[str, Any]:
    device = choose_device(args.device)
    model = load_lewm_checkpoint(checkpoint).to(device).eval()
    model.requires_grad_(False)

    history = int(args.history_size)
    horizon = int(args.rollout_horizon)
    stride = int(args.eval_stride)
    rollout_rows: list[np.ndarray] = []
    teacher_rows: list[np.ndarray] = []
    window_meta: list[dict[str, int]] = []

    for ep_idx, episode in enumerate(episodes):
        pixels = preprocess_pixels(episode['pixels'], args.image_size, device)
        actions = torch.as_tensor(
            np.nan_to_num(episode['action'], nan=0.0),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        encoded = model.encode({'pixels': pixels, 'action': actions})
        emb = encoded['emb']
        act_emb = encoded['act_emb']
        length = int(emb.shape[1])
        if length <= history:
            continue

        starts = range(0, length - history, max(1, stride))
        for start in starts:
            steps = min(horizon, length - history - start)
            if steps <= 0:
                continue
            rollout_mse = rollout_window_mse(model, emb, act_emb, start, history, steps)
            teacher_mse = teacher_forced_window_mse(
                model, emb, act_emb, start, history, steps
            )
            rollout_rows.append(rollout_mse)
            teacher_rows.append(teacher_mse)
            window_meta.append(
                {
                    'episode_index': int(ep_idx),
                    'seed': int(args.seeds[ep_idx]),
                    'start_step': int(start),
                    'num_pred_steps': int(steps),
                }
            )

    rollout = pad_and_stack(rollout_rows)
    teacher = pad_and_stack(teacher_rows)
    rollout_mean = np.nanmean(rollout, axis=0)
    teacher_mean = np.nanmean(teacher, axis=0)
    rollout_std = np.nanstd(rollout, axis=0)
    teacher_std = np.nanstd(teacher, axis=0)

    return {
        'metric': 'latent_embedding_mse',
        'device': device,
        'history_size': history,
        'rollout_horizon': horizon,
        'eval_stride': stride,
        'num_windows': int(len(window_meta)),
        'windows': window_meta,
        'teacher_forced_mse_mean': float(np.nanmean(teacher)),
        'rollout_mse_mean': float(np.nanmean(rollout)),
        'teacher_forced_mse_by_horizon': teacher_mean.tolist(),
        'rollout_mse_by_horizon': rollout_mean.tolist(),
        'teacher_forced_mse_std_by_horizon': teacher_std.tolist(),
        'rollout_mse_std_by_horizon': rollout_std.tolist(),
    }


def choose_device(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    return requested


def load_lewm_checkpoint(checkpoint: Path):
    config_path = checkpoint.parent / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found next to {checkpoint}')
    config = json.loads(config_path.read_text())
    model_cfg = config.get('model', config)
    model = instantiate(model_cfg)
    state_dict = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def preprocess_pixels(
    pixels: np.ndarray,
    image_size: int,
    device: str,
) -> torch.Tensor:
    x = torch.as_tensor(pixels, dtype=torch.float32, device=device)
    x = x.permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    if x.shape[-2:] != (image_size, image_size):
        b, t = x.shape[:2]
        x = F.interpolate(
            x.reshape(b * t, *x.shape[2:]),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False,
        ).reshape(b, t, 3, image_size, image_size)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (x - mean) / std


def rollout_window_mse(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    start: int,
    history: int,
    steps: int,
) -> np.ndarray:
    emb_list = [emb[:, start + idx] for idx in range(history)]
    mses = []
    for offset in range(steps):
        t = start + history + offset
        ctx_emb = torch.stack(emb_list[-history:], dim=1)
        ctx_act = act_emb[:, t - history : t]
        pred = model.predict(ctx_emb, ctx_act)[:, -1]
        target = emb[:, t]
        mses.append((pred - target).pow(2).mean().item())
        emb_list.append(pred)
    return np.asarray(mses, dtype=np.float32)


def teacher_forced_window_mse(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    start: int,
    history: int,
    steps: int,
) -> np.ndarray:
    mses = []
    for offset in range(steps):
        t = start + history + offset
        ctx_emb = emb[:, t - history : t]
        ctx_act = act_emb[:, t - history : t]
        pred = model.predict(ctx_emb, ctx_act)[:, -1]
        target = emb[:, t]
        mses.append((pred - target).pow(2).mean().item())
    return np.asarray(mses, dtype=np.float32)


def pad_and_stack(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float32)
    width = max(len(row) for row in rows)
    out = np.full((len(rows), width), np.nan, dtype=np.float32)
    for idx, row in enumerate(rows):
        out[idx, : len(row)] = row
    return out


def save_metric_plot(metrics: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional artifact
        print(f'[warn] skipping metric plot: {exc}', file=sys.stderr)
        return

    rollout = np.asarray(metrics['rollout_mse_by_horizon'], dtype=np.float32)
    teacher = np.asarray(metrics['teacher_forced_mse_by_horizon'], dtype=np.float32)
    xs = np.arange(1, len(rollout) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, rollout, label='autoregressive rollout')
    ax.plot(xs, teacher, label='teacher-forced one-step')
    ax.set_xlabel('prediction horizon')
    ax.set_ylabel('latent MSE')
    ax.set_title('KinDER Motion2D LeWM latent prediction error')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metric_csv(metrics: dict[str, Any], path: Path) -> None:
    rollout = metrics['rollout_mse_by_horizon']
    teacher = metrics['teacher_forced_mse_by_horizon']
    lines = ['horizon,rollout_mse,teacher_forced_mse']
    for idx, (rollout_mse, teacher_mse) in enumerate(zip(rollout, teacher), start=1):
        lines.append(f'{idx},{rollout_mse},{teacher_mse}')
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
