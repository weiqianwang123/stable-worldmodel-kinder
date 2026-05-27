"""ClutteredStorage2D adapter for the KinDER environment family."""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np

from ._utils import load_kindergarden_class


def _load_cluttered_storage2d_env(home: str | os.PathLike | None = None):
    return load_kindergarden_class(
        'kinder.envs.kinematic2d.clutteredstorage2d',
        'ClutteredStorage2DEnv',
        env_label='KinDER ClutteredStorage2D',
        dependency_hint='kindergarden[kinematic2d]',
        home=home,
    )


class KinderClutteredStorage2D(gym.Env):
    """Thin Gymnasium adapter around KinDER's ``ClutteredStorage2DEnv``.

    The wrapper keeps KinDER's vector observation intact and adds the stable
    world model info keys used by data collection and dataset-driven eval.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        num_blocks: int = 1,
        *,
        kindergarden_home: str | os.PathLike | None = None,
        expose_goal_image: bool = True,
        **kwargs: Any,
    ) -> None:
        ClutteredStorage2DEnv = _load_cluttered_storage2d_env(
            kindergarden_home
        )
        kwargs.setdefault('allow_state_access', True)
        self._env = ClutteredStorage2DEnv(num_blocks=num_blocks, **kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = getattr(self._env, 'render_mode', None)
        self.metadata = dict(getattr(self._env, 'metadata', self.metadata))
        self.num_blocks = num_blocks
        self.expose_goal_image = expose_goal_image
        self._goal_state: np.ndarray | None = None
        self._goal_image: np.ndarray | None = None

    @property
    def unwrapped(self):
        return self

    @property
    def kinder_env(self):
        return self._env

    @property
    def np_random(self):
        return self._env.np_random

    def reset(self, *args: Any, **kwargs: Any):
        options = kwargs.get('options')
        goal_state = None
        if options is not None:
            options = dict(options)
            goal_state = options.pop('goal_state', None)
            if 'state' in options and 'init_state' not in options:
                options['init_state'] = np.asarray(
                    options.pop('state'), dtype=np.float32
                )
            kwargs['options'] = options

        obs, info = self._env.reset(*args, **kwargs)
        if goal_state is not None:
            self._set_goal_state(goal_state)
        else:
            self._refresh_goal_cache(obs)
        return obs, self._augment_info(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, self._augment_info(
            obs, info
        )

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def get_state(self):
        return self._env.get_state()

    def set_state(self, state) -> None:
        self._env.set_state(np.asarray(state, dtype=np.float32))

    def _set_state(self, state) -> None:
        """Stable-worldmodel dataset-eval hook."""
        self.set_state(state)
        self._refresh_goal_cache(self.get_state())

    def _set_goal_state(self, goal_state) -> None:
        """Set the cached storage goal from a full vectorized state."""
        goal_arr = np.asarray(goal_state, dtype=np.float32)
        if goal_arr.shape != self.observation_space.shape:
            raise ValueError(
                'KinDER ClutteredStorage2D expects a full-vector '
                f'goal_state with shape {self.observation_space.shape}, got '
                f'{goal_arr.shape}.'
            )
        self._goal_state = goal_arr.copy()
        self._goal_image = self._render_goal_image()

    def _get_goal_state(self) -> np.ndarray:
        if self._goal_state is None:
            self._refresh_goal_cache(self.get_state())
        return np.asarray(self._goal_state, dtype=np.float32)

    def get_transition(self, state, action):
        return self._env.get_transition(state, action)

    def get_next_state(self, state, action):
        return self._env.get_next_state(state, action)

    def get_reward_and_done(self, state, action):
        return self._env.get_reward_and_done(state, action)

    def get_action_from_gui_input(self, gui_input: dict[str, Any]):
        return self._env.get_action_from_gui_input(gui_input)

    def _augment_info(self, obs, info: dict) -> dict:
        info = dict(info)
        obs_arr = np.asarray(obs, dtype=np.float32)
        info.setdefault('env_name', 'KinderClutteredStorage2D')
        info['state'] = obs_arr
        info['proprio'] = self._robot_proprio_from_state(obs_arr)
        if self._goal_state is None:
            self._refresh_goal_cache(obs_arr)
        info['goal_state'] = np.asarray(self._goal_state, dtype=np.float32)
        info['goal_proprio'] = self._robot_proprio_from_state(
            self._goal_state
        )
        if self.expose_goal_image:
            if self._goal_image is None:
                self._goal_image = self._render_goal_image()
            info['goal'] = self._goal_image
        return info

    def _refresh_goal_cache(self, obs) -> None:
        self._goal_state = self._storage_goal_state_from_obs(obs)
        self._goal_image = (
            self._render_goal_image() if self.expose_goal_image else None
        )

    def _current_object_state(self):
        return self._env.observation_space.devectorize(self.get_state())

    def _set_object_state(self, obj_state) -> None:
        vec_state = self._env.observation_space.vectorize(obj_state)
        self._env.set_state(vec_state)

    @staticmethod
    def _object_by_name(obj_state, name: str):
        for obj in obj_state:
            if obj.name == name:
                return obj
        raise KeyError(
            f'KinDER ClutteredStorage2D state has no object named {name!r}'
        )

    @staticmethod
    def _block_objects(obj_state):
        return sorted(
            [obj for obj in obj_state if obj.name.startswith('block')],
            key=lambda obj: int(obj.name.removeprefix('block')),
        )

    def _robot_proprio_from_state(self, state) -> np.ndarray:
        obj_state = self._env.observation_space.devectorize(
            np.asarray(state, dtype=np.float32)
        )
        robot = self._object_by_name(obj_state, 'robot')
        keys = ('x', 'y', 'theta', 'arm_joint', 'vacuum')
        return np.asarray([obj_state.get(robot, k) for k in keys], np.float32)

    def _storage_goal_state_from_obs(self, obs) -> np.ndarray:
        obj_state = self._env.observation_space.devectorize(
            np.asarray(obs, dtype=np.float32).copy()
        )
        shelf = self._object_by_name(obj_state, 'shelf')
        robot = self._object_by_name(obj_state, 'robot')
        blocks = self._block_objects(obj_state)

        obj_state.set(robot, 'vacuum', 0.0)
        obj_state.set(robot, 'arm_joint', float(obj_state.get(robot, 'base_radius')))

        shelf_x = float(obj_state.get(shelf, 'x1'))
        shelf_y = float(obj_state.get(shelf, 'y1'))
        shelf_w = float(obj_state.get(shelf, 'width1'))
        shelf_h = float(obj_state.get(shelf, 'height1'))
        if not blocks:
            return np.asarray(
                self._env.observation_space.vectorize(obj_state),
                dtype=np.float32,
            )

        block_w = max(float(obj_state.get(block, 'width')) for block in blocks)
        block_h = max(float(obj_state.get(block, 'height')) for block in blocks)
        cols = max(1, int(np.floor(shelf_w / max(block_w, 1e-6))))
        rows = max(1, int(np.ceil(len(blocks) / cols)))
        x_pad = max(0.0, shelf_w - cols * block_w) / (cols + 1)
        y_pad = max(0.0, shelf_h - rows * block_h) / (rows + 1)

        for idx, block in enumerate(blocks):
            row, col = divmod(idx, cols)
            x = shelf_x + x_pad + col * (block_w + x_pad)
            y = shelf_y + y_pad + row * (block_h + y_pad)
            obj_state.set(block, 'x', float(x))
            obj_state.set(block, 'y', float(y))
            obj_state.set(block, 'theta', 0.0)

        return np.asarray(
            self._env.observation_space.vectorize(obj_state),
            dtype=np.float32,
        )

    def _render_goal_image(self) -> np.ndarray:
        current_vec = np.asarray(self.get_state(), dtype=np.float32)
        goal_vec = (
            self._goal_state
            if self._goal_state is not None
            else self._storage_goal_state_from_obs(current_vec)
        )
        try:
            self._env.set_state(np.asarray(goal_vec, dtype=np.float32))
            return np.asarray(self.render()).copy()
        finally:
            self._env.set_state(current_vec)


__all__ = ['KinderClutteredStorage2D']
