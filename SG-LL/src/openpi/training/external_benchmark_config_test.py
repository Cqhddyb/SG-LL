from openpi.models import pi0_config
from openpi.training import config


def _assert_matched(base_name: str, geo_name: str, *, offset: int) -> None:
    baseline = config.get_config(base_name)
    planned = config.get_config(geo_name)

    assert isinstance(baseline.model, pi0_config.Pi0Config)
    assert baseline.model == planned.model
    assert baseline.data.repo_id == planned.data.repo_id
    assert baseline.data.future_teacher_obs_offset is None
    assert planned.data.future_teacher_obs_offset == offset
    assert baseline.pytorch_weight_path == planned.pytorch_weight_path == "./checkpoints/pi05_base_full_torch"
    assert baseline.freeze_vlm_backbone is planned.freeze_vlm_backbone is True
    assert baseline.train_action_expert is planned.train_action_expert is True
    assert baseline.batch_size == planned.batch_size == 8
    assert baseline.num_train_steps == planned.num_train_steps == 10_000
    assert baseline.lr_schedule == planned.lr_schedule
    assert planned.scheme4_horizon_id == offset
    assert planned.scheme4_delta_enabled is True


def test_calvin_original_structure_and_sgvtp_configs_are_matched() -> None:
    _assert_matched("pi05_calvin_base", "pi05_calvin_geo", offset=10)


def test_calvin_vggt_path_defaults_locally_and_supports_explicit_remote_override(monkeypatch) -> None:
    monkeypatch.delenv("OPENPI_CALVIN_VGGT_WEIGHT_PATH", raising=False)
    assert config.get_config("pi05_calvin_geo").vggt_weight_path == "checkpoints/vggt"

    monkeypatch.setenv("OPENPI_CALVIN_VGGT_WEIGHT_PATH", "/remote/data/models/vggt_ckpt")
    assert config.get_config("pi05_calvin_geo").vggt_weight_path == "/remote/data/models/vggt_ckpt"


def test_simpler_bridge_original_structure_and_sgvtp_configs_are_matched() -> None:
    _assert_matched("pi05_simpler_bridge_base", "pi05_simpler_bridge_geo", offset=5)
    baseline = config.get_config("pi05_simpler_bridge_base")
    planned = config.get_config("pi05_simpler_bridge_geo")
    assert baseline.data.repo_id == planned.data.repo_id == "FedorX8/bridge_v2_lerobot"
    assert baseline.data.image_key == planned.data.image_key == "observation.images.primary"
    assert baseline.data.state_layout == planned.data.state_layout == "bridge_eef7_pad_zero"
    assert (
        baseline.data.output_action_layout
        == planned.data.output_action_layout
        == "simpler_axis_angle_binary_gripper"
    )
    assert baseline.optimizer == planned.optimizer
    assert baseline.ema_decay is planned.ema_decay is None


def test_standard_libero_action_expert_pair_is_explicit_and_budget_matched() -> None:
    baseline = config.get_config("pi05_standard_libero_action_expert_base")
    planned = config.get_config("pi05_standard_libero_action_expert_geo")

    assert isinstance(baseline.model, pi0_config.Pi0Config)
    assert baseline.model == planned.model
    assert baseline.data.repo_id == planned.data.repo_id == "physical-intelligence/libero"
    assert baseline.data.future_teacher_obs_offset is None
    assert planned.data.future_teacher_obs_offset == 10
    assert baseline.pytorch_weight_path == planned.pytorch_weight_path == "./checkpoints/pi05_libero_official_torch"
    assert baseline.freeze_vlm_backbone is planned.freeze_vlm_backbone is True
    assert baseline.train_action_expert is planned.train_action_expert is True
    assert baseline.num_train_steps == planned.num_train_steps == 10_000
    assert baseline.lr_schedule == planned.lr_schedule
    assert baseline.optimizer == planned.optimizer
    assert baseline.ema_decay is planned.ema_decay is None
    assert baseline.batch_size * baseline.gradient_accumulation_steps == 16
    assert planned.batch_size * planned.gradient_accumulation_steps == 16
    assert planned.scheme4_horizon_id == 10
    assert planned.scheme4_delta_enabled is True
