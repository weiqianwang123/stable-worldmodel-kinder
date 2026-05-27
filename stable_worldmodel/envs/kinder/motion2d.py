"""Motion2D adapter for the KinDER environment family.

Imports are resolved lazily when the environment is constructed. This keeps
``stable_worldmodel.envs`` importable when KinDER or its optional dependencies
are not installed.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np

from ._utils import load_kindergarden_class


def _load_motion2d_env(home: str | os.PathLike | None = None):
    return load_kindergarden_class(
        'kinder.envs.kinematic2d.motion2d',
        'Motion2DEnv',
        env_label='KinDER Motion2D',
        dependency_hint='kindergarden[kinematic2d]',
        home=home,
    )


class KinderMotion2D(gym.Env):
    """Thin Gymnasium adapter around KinDER's ``Motion2DEnv``.

    Args:
        num_passages: Number of narrow passages in the Motion2D variant.
        kindergarden_home: Optional local checkout root. Defaults to
            ``$KINDERGARDEN_HOME`` or ``/home/robin_wang/kindergarden``.
        **kwargs: Forwarded to KinDER's ``Motion2DEnv``.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        num_passages: int = 3,
        *,
        kindergarden_home: str | os.PathLike | None = None,
        expose_goal_image: bool = True,
        **kwargs: Any,
    ) -> None:
        Motion2DEnv = _load_motion2d_env(kindergarden_home)
        kwargs.setdefault('allow_state_access', True)
        self._env = Motion2DEnv(num_passages=num_passages, **kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = getattr(self._env, 'render_mode', None)
        self.metadata = dict(getattr(self._env, 'metadata', self.metadata))
        self.num_passages = num_passages
        self.expose_goal_image = expose_goal_image
        self._goal_xy: np.ndarray | None = None
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
        self._env.set_state(state)

    def _set_state(self, state) -> None:
        """Stable-worldmodel dataset-eval hook."""
        self.set_state(np.asarray(state, dtype=np.float32))
        self._refresh_goal_cache(self.get_state())

    def _set_goal_state(self, goal_state) -> None:
        """Set the Motion2D target region from a 2D or full-vector goal."""
        goal_arr = np.asarray(goal_state, dtype=np.float32)
        target_xy = self._goal_xy_from_state(goal_arr)
        obj_state = self._current_object_state()
        target = self._object_by_name(obj_state, 'target_region')
        obj_state.set(target, 'x', float(target_xy[0]))
        obj_state.set(target, 'y', float(target_xy[1]))
        self._set_object_state(obj_state)
        self._goal_xy = np.asarray(target_xy, dtype=np.float32)
        self._goal_state = self._goal_state_from_xy(
            self.get_state(), self._goal_xy, source_state=goal_arr
        )
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
        info.setdefault('env_name', 'KinderMotion2D')
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
        obs_arr = np.asarray(obs, dtype=np.float32)
        self._goal_xy = self._target_xy_from_state(obs_arr)
        self._goal_state = self._goal_state_from_xy(obs_arr, self._goal_xy)
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
        raise KeyError(f'KinDER Motion2D state has no object named {name!r}')

    def _robot_proprio_from_state(self, state) -> np.ndarray:
        obj_state = self._env.observation_space.devectorize(
            np.asarray(state, dtype=np.float32)
        )
        robot = self._object_by_name(obj_state, 'robot')
        keys = ('x', 'y', 'theta', 'arm_joint', 'vacuum')
        return np.asarray([obj_state.get(robot, k) for k in keys], np.float32)

    def _target_xy_from_state(self, state) -> np.ndarray:
        obj_state = self._env.observation_space.devectorize(
            np.asarray(state, dtype=np.float32)
        )
        target = self._object_by_name(obj_state, 'target_region')
        return np.asarray(
            [obj_state.get(target, 'x'), obj_state.get(target, 'y')],
            dtype=np.float32,
        )

    def _goal_xy_from_state(self, state) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape == (2,):
            return arr

        obj_state = self._env.observation_space.devectorize(arr)
        robot = self._object_by_name(obj_state, 'robot')
        return np.asarray(
            [obj_state.get(robot, 'x'), obj_state.get(robot, 'y')],
            dtype=np.float32,
        )

    def _goal_state_from_xy(
        self,
        base_state,
        goal_xy,
        *,
        source_state=None,
    ) -> np.ndarray:
        obj_state = self._env.observation_space.devectorize(
            np.asarray(base_state, dtype=np.float32).copy()
        )
        robot = self._object_by_name(obj_state, 'robot')

        source_arr = (
            None
            if source_state is None
            else np.asarray(source_state, dtype=np.float32).copy()
        )
        if source_arr is not None and source_arr.shape != (2,):
            source_obj_state = self._env.observation_space.devectorize(
                source_arr
            )
            source_robot = self._object_by_name(source_obj_state, 'robot')
            for key in ('x', 'y', 'theta', 'arm_joint', 'vacuum'):
                obj_state.set(
                    robot,
                    key,
                    float(source_obj_state.get(source_robot, key)),
                )
        else:
            obj_state.set(robot, 'x', float(goal_xy[0]))
            obj_state.set(robot, 'y', float(goal_xy[1]))

        target = self._object_by_name(obj_state, 'target_region')
        obj_state.set(target, 'x', float(goal_xy[0]))
        obj_state.set(target, 'y', float(goal_xy[1]))
        return np.asarray(
            self._env.observation_space.vectorize(obj_state),
            dtype=np.float32,
        )

    def _render_goal_image(self) -> np.ndarray:
        current_vec = np.asarray(self.get_state(), dtype=np.float32)
        if self._goal_state is None:
            target_xy = self._goal_xy
            if target_xy is None:
                target_xy = self._target_xy_from_state(current_vec)
            vec_state = self._goal_state_from_xy(current_vec, target_xy)
        else:
            vec_state = self._goal_state
        goal_state = self._env.observation_space.devectorize(vec_state)

        try:
            self._set_object_state(goal_state)
            return np.asarray(self.render()).copy()
        finally:
            self._env.set_state(current_vec)


__all__ = ['KinderMotion2D']
