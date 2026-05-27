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
from stable_worldmodel.plot.video_utils import save_video  # noqa: E402
from stable_worldmodel.solver.callbacks import Callback  # noqa: E402


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
    parser.add_argument('--planning-horizon', type=int, default=20)
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
    parser.add_argument('--planner-debug-video', action='store_true')
    parser.add_argument('--planner-debug-topk', type=int, default=8)
    parser.add_argument(
        '--planner-debug-cem-iters',
        choices=('last', 'all'),
        default='last',
    )
    parser.add_argument('--planner-debug-every', type=int, default=1)
    parser.add_argument('--planner-debug-fps', type=int, default=6)
    parser.add_argument(
        '--planner-debug-dir',
        type=Path,
        default=Path('planner_debug'),
    )
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
    args.planner_debug_topk = max(1, int(args.planner_debug_topk))
    args.planner_debug_every = max(1, int(args.planner_debug_every))
    args.planner_debug_fps = max(1, int(args.planner_debug_fps))
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
    if args.planner_debug_video and args.solver != 'cem':
        raise ValueError('--planner-debug-video currently supports --solver cem only')

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
    env_id = f'swm/KinderMotion2D-p{args.num_passages}-v0'
    cem_debug_callback = (
        CEMRolloutDebugCallback(
            keep_iterations=args.planner_debug_cem_iters,
            max_topk=args.planner_debug_topk,
        )
        if args.planner_debug_video
        else None
    )
    planner_debug = (
        MPCPlannerDebugRecorder(
            args=args,
            env_id=env_id,
            seeds=args.seeds,
            process=process,
        )
        if args.planner_debug_video
        else None
    )

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
            callbacks=[] if cem_debug_callback is None else [cem_debug_callback],
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
        planner_debug=planner_debug,
    )

    goal_offset = resolve_planning_goal_offset(args, episodes)
    start_steps = [int(args.planning_start_step)] * len(episodes)
    episodes_idx = list(range(len(episodes)))
    video_dir = None if args.no_planning_video else args.out_dir / 'planning_videos'

    world = swm.World(
        env_id,
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

    debug_artifacts = None
    if planner_debug is not None:
        debug_dir = (
            args.planner_debug_dir
            if args.planner_debug_dir.is_absolute()
            else args.out_dir / args.planner_debug_dir
        )
        debug_artifacts = planner_debug.save(debug_dir)

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
        'planner_debug': debug_artifacts,
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


class CEMRolloutDebugCallback(Callback):
    """Capture compact CEM snapshots for later simulator rollout rendering."""

    name = 'planner_rollout_debug'

    def __init__(self, *, keep_iterations: str, max_topk: int) -> None:
        super().__init__(reduction='none')
        if keep_iterations not in ('last', 'all'):
            raise ValueError(
                f"keep_iterations must be 'last' or 'all', got {keep_iterations!r}"
            )
        self.keep_iterations = keep_iterations
        self.max_topk = max(1, int(max_topk))

    def __call__(self, **state: Any) -> None:
        value = self.compute(**state)
        if value is None:
            return
        if self.keep_iterations == 'last' and self._current:
            self._current[-1] = value
        else:
            self._current.append(value)

    def compute(self, **state: Any) -> dict[str, Any]:
        topk_candidates: torch.Tensor = state['topk_candidates']
        topk_vals: torch.Tensor = state['topk_vals']
        costs: torch.Tensor = state['costs']
        mean: torch.Tensor = state['mean']
        k = min(self.max_topk, int(topk_candidates.shape[1]))
        return {
            'step': int(state['step']),
            'topk_candidates': topk_candidates[:, :k].detach().cpu().float().numpy(),
            'topk_vals': topk_vals[:, :k].detach().cpu().float().numpy(),
            'mean': mean.detach().cpu().float().numpy(),
            'best_cost': costs.min(dim=1).values.detach().cpu().float().numpy(),
            'mean_cost': costs.mean(dim=1).detach().cpu().float().numpy(),
        }


class MPCPlannerDebugRecorder:
    """Stores CEM snapshots and writes Motion2D MPC rollout videos."""

    callback_key = CEMRolloutDebugCallback.name

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        env_id: str,
        seeds: list[int],
        process: dict[str, ColumnStandardizer],
    ) -> None:
        self.args = args
        self.env_id = env_id
        self.seeds = [int(seed) for seed in seeds]
        self.process = process
        self.records_by_env: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(len(self.seeds))
        }
        self._replan_counts = [0 for _ in self.seeds]

    def capture(
        self,
        *,
        raw_info_dict: dict[str, Any],
        replan_idx: list[int],
        outputs: dict[str, Any],
        solver: Any,
        config: swm.PlanConfig,
        action_space: gym.Space,
    ) -> None:
        callbacks = outputs.get('callbacks', {})
        history = callbacks.get(self.callback_key)
        if not history:
            return

        selected = tensor_to_numpy(outputs['actions'])
        replan_numbers = {}
        for row, env_i in enumerate(replan_idx):
            replan_numbers[row] = self._replan_counts[env_i]
            self._replan_counts[env_i] += 1

        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        batch_size = int(getattr(solver, 'batch_size', len(replan_idx)))
        for batch_i, batch_records in enumerate(history):
            row_start = batch_i * batch_size
            for cem_record in batch_records:
                mean = np.asarray(cem_record['mean'], dtype=np.float32)
                topk = np.asarray(
                    cem_record['topk_candidates'], dtype=np.float32
                )
                topk_vals = np.asarray(cem_record['topk_vals'], dtype=np.float32)
                best_cost = np.asarray(cem_record['best_cost'], dtype=np.float32)
                mean_cost = np.asarray(cem_record['mean_cost'], dtype=np.float32)
                for local_i in range(mean.shape[0]):
                    row = row_start + local_i
                    if row >= len(replan_idx):
                        continue
                    env_i = int(replan_idx[row])
                    replan_number = int(replan_numbers[row])
                    if replan_number % max(1, int(self.args.planner_debug_every)) != 0:
                        continue

                    selected_norm = (
                        selected[row]
                        if self.args.planner_debug_cem_iters == 'last'
                        else mean[local_i]
                    )
                    self.records_by_env[env_i].append(
                        {
                            'seed': self.seeds[env_i],
                            'env_index': env_i,
                            'replan_index': replan_number,
                            'cem_iteration': int(cem_record['step']),
                            'state': latest_info_value(
                                raw_info_dict, 'state', env_i
                            ).astype(np.float32),
                            'goal_state': latest_info_value(
                                raw_info_dict, 'goal_state', env_i
                            ).astype(np.float32),
                            'selected_action_plan': denormalize_action_plan(
                                selected_norm,
                                process=self.process,
                                low=low,
                                high=high,
                                action_block=config.action_block,
                            ),
                            'topk_action_plans': denormalize_action_plan(
                                topk[local_i],
                                process=self.process,
                                low=low,
                                high=high,
                                action_block=config.action_block,
                            ),
                            'topk_costs': topk_vals[local_i],
                            'best_cost': float(best_cost[local_i]),
                            'mean_cost': float(mean_cost[local_i]),
                        }
                    )

    def save(self, out_dir: Path) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            'dir': str(out_dir),
            'videos': [],
            'arrays': [],
            'json': [],
            'num_records': 0,
        }
        make_kwargs: dict[str, Any] = {'render_mode': 'rgb_array'}
        if self.args.kindergarden_home is not None:
            make_kwargs['kindergarden_home'] = self.args.kindergarden_home

        env = gym.make(self.env_id, **make_kwargs)
        try:
            env.reset(seed=0)
            for env_i, records in self.records_by_env.items():
                if not records:
                    continue
                seed = self.seeds[env_i]
                frames = [render_mpc_debug_frame(env, rec) for rec in records]
                video_path = out_dir / f'seed_{seed}_mpc_rollouts.mp4'
                save_video(video_path, frames, fps=int(self.args.planner_debug_fps))
                npz_path = out_dir / f'seed_{seed}_mpc_rollouts.npz'
                json_path = out_dir / f'seed_{seed}_mpc_rollouts.json'
                save_mpc_debug_npz(npz_path, records)
                json_path.write_text(
                    json.dumps(mpc_debug_json(records), indent=2) + '\n'
                )
                artifacts['videos'].append(str(video_path))
                artifacts['arrays'].append(str(npz_path))
                artifacts['json'].append(str(json_path))
                artifacts['num_records'] += len(records)
        finally:
            env.close()
        return artifacts


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def latest_info_value(
    info_dict: dict[str, Any],
    key: str,
    env_i: int,
) -> np.ndarray:
    if key not in info_dict:
        raise KeyError(f'Planner debug requires info[{key!r}]')
    arr = tensor_to_numpy(info_dict[key])
    value = np.asarray(arr[env_i])
    if value.ndim >= 1 and value.shape[0] == 1:
        value = value[-1]
    return np.asarray(value)


def denormalize_action_plan(
    plan: np.ndarray,
    *,
    process: dict[str, ColumnStandardizer],
    low: np.ndarray,
    high: np.ndarray,
    action_block: int,
) -> np.ndarray:
    arr = np.asarray(plan, dtype=np.float32)
    action_dim = int(low.size)
    if arr.shape[-1] != action_dim * int(action_block):
        raise ValueError(
            'Expected flattened action dim '
            f'{action_dim * int(action_block)}, got {arr.shape[-1]}'
        )
    arr = arr.reshape(*arr.shape[:-2], arr.shape[-2], int(action_block), action_dim)
    arr = arr.reshape(*arr.shape[:-3], arr.shape[-3] * int(action_block), action_dim)
    if 'action' in process:
        arr = process['action'].inverse_transform(arr)
    return np.clip(arr, low, high).astype(np.float32)


def save_mpc_debug_npz(path: Path, records: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        replan_index=np.asarray([r['replan_index'] for r in records], np.int32),
        cem_iteration=np.asarray([r['cem_iteration'] for r in records], np.int32),
        state=np.stack([r['state'] for r in records]).astype(np.float32),
        goal_state=np.stack([r['goal_state'] for r in records]).astype(np.float32),
        selected_action_plan=np.stack(
            [r['selected_action_plan'] for r in records]
        ).astype(np.float32),
        topk_action_plans=np.stack(
            [r['topk_action_plans'] for r in records]
        ).astype(np.float32),
        topk_costs=np.stack([r['topk_costs'] for r in records]).astype(np.float32),
        best_cost=np.asarray([r['best_cost'] for r in records], np.float32),
        mean_cost=np.asarray([r['mean_cost'] for r in records], np.float32),
    )


def mpc_debug_json(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'seed': int(records[0]['seed']),
        'num_frames': len(records),
        'records': [
            {
                'replan_index': int(r['replan_index']),
                'cem_iteration': int(r['cem_iteration']),
                'best_cost': float(r['best_cost']),
                'mean_cost': float(r['mean_cost']),
                'topk_costs': np.asarray(r['topk_costs'], dtype=float).tolist(),
                'selected_first_action': np.asarray(
                    r['selected_action_plan'][0], dtype=float
                ).tolist(),
            }
            for r in records
        ],
    }


def render_mpc_debug_frame(env: gym.Env, record: dict[str, Any]) -> np.ndarray:
    from kinder.envs.utils import render_2dstate

    state = np.asarray(record['state'], dtype=np.float32)
    kinder_env = env.unwrapped.kinder_env
    obj_state = kinder_env.observation_space.devectorize(state)
    oc_env = kinder_env._object_centric_env
    config = oc_env.config

    topk_paths = [
        simulate_robot_xy_rollout(env, state, plan)
        for plan in np.asarray(record['topk_action_plans'], dtype=np.float32)
    ]
    selected_path = simulate_robot_xy_rollout(
        env, state, np.asarray(record['selected_action_plan'], dtype=np.float32)
    )
    current_xy = robot_xy_from_vector(env, state)
    goal_xy = robot_xy_from_vector(env, record['goal_state'])

    def draw(ax) -> None:
        for idx, path in enumerate(topk_paths):
            if len(path) < 2:
                continue
            alpha = 0.18 + 0.35 * (1.0 - idx / max(1, len(topk_paths) - 1))
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=(0.05, 0.28, 0.95, alpha),
                linewidth=1.2,
                zorder=100,
            )
        if len(selected_path) >= 2:
            ax.plot(
                selected_path[:, 0],
                selected_path[:, 1],
                color=(1.0, 0.85, 0.0, 0.95),
                linewidth=5.0,
                zorder=110,
            )
            ax.plot(
                selected_path[:, 0],
                selected_path[:, 1],
                color=(0.9, 0.05, 0.02, 0.95),
                linewidth=2.0,
                zorder=111,
            )
        ax.scatter(
            [current_xy[0]],
            [current_xy[1]],
            s=42,
            c='white',
            edgecolors='black',
            linewidths=1.0,
            zorder=120,
        )
        ax.scatter(
            [goal_xy[0]],
            [goal_xy[1]],
            s=125,
            c='lime',
            edgecolors='black',
            linewidths=1.0,
            marker='*',
            zorder=121,
        )
        ax.text(
            0.02,
            0.98,
            (
                f"seed={record['seed']}  replan={record['replan_index']}  "
                f"cem={record['cem_iteration']}  best={record['best_cost']:.3f}"
            ),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            color='black',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.78,
                'pad': 3,
            },
            zorder=130,
        )

    frame = render_2dstate(
        obj_state,
        {},
        config.world_min_x,
        config.world_max_x,
        config.world_min_y,
        config.world_max_y,
        config.render_dpi,
        ax_callback=draw,
    )
    return ensure_uint8_rgb(frame)


def simulate_robot_xy_rollout(
    env: gym.Env,
    state: np.ndarray,
    action_plan: np.ndarray,
) -> np.ndarray:
    current = np.asarray(state, dtype=np.float32).copy()
    points = [robot_xy_from_vector(env, current)]
    for action in np.asarray(action_plan, dtype=np.float32):
        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        ).astype(env.action_space.dtype)
        try:
            current, _, terminated = env.unwrapped.get_transition(current, action)
        except Exception:
            break
        points.append(robot_xy_from_vector(env, current))
        if terminated:
            break
    return np.asarray(points, dtype=np.float32)


def robot_xy_from_vector(env: gym.Env, state: np.ndarray) -> np.ndarray:
    obj_state = env.unwrapped.kinder_env.observation_space.devectorize(
        np.asarray(state, dtype=np.float32)
    )
    robot = object_by_name(obj_state, 'robot')
    return np.asarray(
        [obj_state.get(robot, 'x'), obj_state.get(robot, 'y')],
        dtype=np.float32,
    )


def ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    pad_h = (-h) % 16
    pad_w = (-w) % 16
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return arr


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
    def __init__(self, *args: Any, planner_debug=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.planner_debug = planner_debug

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, 'env'), 'Environment not set for the policy'

        raw_info_dict = info_dict
        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0

        terminated = info_dict.get('terminated')
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        replan_idx = [
            i
            for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]

        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_idx]
                else:
                    sliced[k] = v

            sliced_init = (
                self._next_init[idx_tensor]
                if self._next_init is not None
                else None
            )

            outputs = self.solver(sliced, init_action=sliced_init)

            if self.planner_debug is not None:
                self.planner_debug.capture(
                    raw_info_dict=raw_info_dict,
                    replan_idx=replan_idx,
                    outputs=outputs,
                    solver=self.solver,
                    config=self.cfg,
                    action_space=self.env.single_action_space,
                )

            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(
                        n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype
                    )
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(
                len(replan_idx), self.flatten_receding_horizon, -1
            )

            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float('nan'))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape)
        action = action.float().numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

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
