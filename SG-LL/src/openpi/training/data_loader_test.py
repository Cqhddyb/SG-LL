import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _FakeLeRobotDataset:
    def __init__(self, items):
        self.items = items
        self.hf_dataset = items
        self.episode_data_index = {"to": [len(items)]}

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


def _robotwin_item(index: int, episode_index: int = 0):
    image = np.full((3, 4, 5), index, dtype=np.uint8)
    return {
        "episode_index": np.asarray(episode_index),
        "observation.images.cam_high": image,
        "observation.images.cam_left_wrist": image + 1,
        "observation.images.cam_right_wrist": image + 2,
    }


def test_future_teacher_lerobot_dataset_adds_robotwin_triplet():
    dataset = _data_loader.FutureTeacherLeRobotDataset(
        _FakeLeRobotDataset([_robotwin_item(0), _robotwin_item(1), _robotwin_item(2)]),
        future_obs_offset=1,
    )

    item = dataset[0]

    assert set(item["teacher_images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert set(item["future_teacher_images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    np.testing.assert_array_equal(item["teacher_images"]["cam_high"], np.full((3, 4, 5), 0, dtype=np.uint8))
    np.testing.assert_array_equal(item["future_teacher_images"]["cam_high"], np.full((3, 4, 5), 1, dtype=np.uint8))
    assert item["future_teacher_fallback"].shape == ()
    assert bool(item["future_teacher_fallback"]) is False


def test_future_teacher_lerobot_dataset_robotwin_fallback_uses_episode_final_frame():
    dataset = _data_loader.FutureTeacherLeRobotDataset(
        _FakeLeRobotDataset([_robotwin_item(0), _robotwin_item(1), _robotwin_item(2)]),
        future_obs_offset=5,
    )

    item = dataset[1]

    np.testing.assert_array_equal(item["future_teacher_images"]["cam_high"], np.full((3, 4, 5), 2, dtype=np.uint8))
    assert bool(item["future_teacher_fallback"]) is True


def _calvin_item(index: int, episode_index: int = 0):
    image = np.full((3, 4, 5), index, dtype=np.uint8)
    return {
        "episode_index": np.asarray(episode_index),
        "observation.images.top": image,
        "observation.images.wrist": image + 1,
    }


def test_future_teacher_lerobot_dataset_adds_calvin_pair() -> None:
    dataset = _data_loader.FutureTeacherLeRobotDataset(
        _FakeLeRobotDataset([_calvin_item(0), _calvin_item(1), _calvin_item(2)]),
        future_obs_offset=1,
    )

    item = dataset[0]

    assert set(item["teacher_images"]) == {"top", "wrist"}
    assert set(item["future_teacher_images"]) == {"top", "wrist"}
    np.testing.assert_array_equal(item["teacher_images"]["top"], np.full((3, 4, 5), 0, dtype=np.uint8))
    np.testing.assert_array_equal(
        item["future_teacher_images"]["wrist"], np.full((3, 4, 5), 2, dtype=np.uint8)
    )
    assert bool(item["future_teacher_fallback"]) is False


def test_future_teacher_lerobot_dataset_adds_single_camera_pair() -> None:
    dataset = _data_loader.FutureTeacherLeRobotDataset(
        _FakeLeRobotDataset(
            [
                {"episode_index": np.asarray(0), "observation.images.image_0": np.full((3, 4, 5), i, dtype=np.uint8)}
                for i in range(3)
            ]
        ),
        future_obs_offset=3,
    )

    item = dataset[1]

    np.testing.assert_array_equal(item["teacher_image"], np.full((3, 4, 5), 1, dtype=np.uint8))
    np.testing.assert_array_equal(item["future_teacher_image"], np.full((3, 4, 5), 2, dtype=np.uint8))
    assert bool(item["future_teacher_fallback"]) is True


def test_future_teacher_lerobot_dataset_accepts_fedor_primary_camera() -> None:
    dataset = _data_loader.FutureTeacherLeRobotDataset(
        _FakeLeRobotDataset(
            [
                {
                    "episode_index": np.asarray(0),
                    "observation.images.primary": np.full((3, 4, 5), i, dtype=np.uint8),
                }
                for i in range(3)
            ]
        ),
        future_obs_offset=1,
    )

    item = dataset[0]

    np.testing.assert_array_equal(item["teacher_image"], np.full((3, 4, 5), 0, dtype=np.uint8))
    np.testing.assert_array_equal(item["future_teacher_image"], np.full((3, 4, 5), 1, dtype=np.uint8))
    assert bool(item["future_teacher_fallback"]) is False
