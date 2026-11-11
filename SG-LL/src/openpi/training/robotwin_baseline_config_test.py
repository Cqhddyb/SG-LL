from openpi.models import pi0_config
from openpi.training import config


def test_robotwin_original_structure_baseline_preserves_route_a_io_contract() -> None:
    task_name = "adjust_bottle"
    baseline = config.get_config(f"pi05_robotwin_{task_name}_base")
    route_a = config.get_config(f"pi05_robotwin_{task_name}_geo")

    assert "geo" not in baseline.name
    assert isinstance(baseline.model, pi0_config.Pi0Config)
    assert baseline.model.pi05
    assert baseline.model.action_dim == route_a.model.action_dim == 32
    assert baseline.model.action_horizon == route_a.model.action_horizon == 50

    assert baseline.data.repo_id == route_a.data.repo_id == f"robotwin/{task_name}_demo_clean_50"
    assert baseline.data.future_teacher_obs_offset is None
    assert baseline.data.use_delta_joint_actions is True
    assert baseline.data.adapt_to_pi is False
    assert baseline.data.assets.asset_id == baseline.data.repo_id
    assert baseline.data.assets.assets_dir == f"./assets/pi05_robotwin_{task_name}_geo"

    assert baseline.pytorch_weight_path == route_a.pytorch_weight_path
    assert baseline.freeze_vlm_backbone is True
    assert baseline.train_action_expert is True
    assert route_a.train_action_expert is False
    assert baseline.num_train_steps == route_a.num_train_steps == 20_000
    assert baseline.lr_schedule == route_a.lr_schedule
