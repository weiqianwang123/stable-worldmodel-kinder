from pathlib import Path

import numpy as np
import pytest


gym = pytest.importorskip('gymnasium')


def test_kinder_motion2d_smoke():
    from stable_worldmodel.envs.kinder._utils import (
        ensure_kindergarden_on_path,
    )

    if not Path('/home/robin_wang/kindergarden').exists():
        pytest.importorskip('kinder')

    ensure_kindergarden_on_path()
    pytest.importorskip('kinder')
    pytest.importorskip('tomsgeoms2d')
    pytest.importorskip('relational_structs')
    pytest.importorskip('prpl_utils')

    import stable_worldmodel.envs  # noqa: F401

    env = gym.make('swm/KinderMotion2D-p2-v0')
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert env.unwrapped.__class__.__name__ == 'KinderMotion2D'
        assert isinstance(info, dict)
        np.testing.assert_allclose(info['state'], obs)
        np.testing.assert_allclose(info['state'], env.unwrapped.get_state())
        assert info['goal_state'].shape == obs.shape
        assert info['proprio'].shape == info['goal_proprio'].shape

        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        frame = env.render()
        assert frame.ndim == 3
        assert frame.shape[-1] in (3, 4)

        goal_state = info['state'].copy()
        obs, info = env.reset(seed=1, options={'goal_state': goal_state})
        assert env.observation_space.contains(obs)
        np.testing.assert_allclose(info['state'], obs)
        np.testing.assert_allclose(info['state'], env.unwrapped.get_state())
        assert info['goal_state'].shape == obs.shape
        assert info['goal_proprio'].shape == info['proprio'].shape
    finally:
        env.close()


def test_kinder_cluttered_storage2d_smoke():
    from stable_worldmodel.envs.kinder._utils import (
        ensure_kindergarden_on_path,
    )

    if not Path('/home/robin_wang/kindergarden').exists():
        pytest.importorskip('kinder')

    ensure_kindergarden_on_path()
    pytest.importorskip('kinder')
    pytest.importorskip('tomsgeoms2d')
    pytest.importorskip('relational_structs')
    pytest.importorskip('prpl_utils')

    import stable_worldmodel.envs  # noqa: F401

    env = gym.make('swm/KinderClutteredStorage2D-b1-v0')
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert env.unwrapped.__class__.__name__ == (
            'KinderClutteredStorage2D'
        )
        assert isinstance(info, dict)
        np.testing.assert_allclose(info['state'], obs)
        np.testing.assert_allclose(info['state'], env.unwrapped.get_state())
        assert info['goal_state'].shape == obs.shape
        assert info['proprio'].shape == info['goal_proprio'].shape

        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        frame = env.render()
        assert frame.ndim == 3
        assert frame.shape[-1] in (3, 4)

        goal_state = info['goal_state'].copy()
        obs, info = env.reset(seed=1, options={'goal_state': goal_state})
        assert env.observation_space.contains(obs)
        np.testing.assert_allclose(info['state'], obs)
        np.testing.assert_allclose(info['state'], env.unwrapped.get_state())
        assert info['goal_state'].shape == obs.shape
        assert info['goal_proprio'].shape == info['proprio'].shape
    finally:
        env.close()
