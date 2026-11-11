# Tensor Contracts

## Optional request noise (2026-11-10)

- `CONFIRMED`: `_policy_noise_seed` is a scalar 63-bit integer. A private
  device Torch generator produces float32 `[H_action,A_dim]`, passed through
  existing inference batching to `[1,H_action,A_dim]`. Dimensions come from the
  loaded model config (pilot: 10,32). Seed removed before observation transforms.
- `CONFIRMED`: Optional acknowledgment contains protocol, seed and noise SHA256;
  no-key requests retain legacy RNG/schema. Fixed replay matches 70 action
  hashes/diagnostics; no-lookahead is exactly zero across 120 logged requests.
  Evidence: `src/openpi/policies/policy_request_noise_test.py` and
  `artifacts/adaptive_lookahead/20261110_router_paired_eval/summaries/`.

## SimplerEnv Fedor Bridge-v2 Boundary

- `CONFIRMED_CODE`: The pinned Fedor source exposes
  `observation.images.primary[480,640,3]`,
  `observation.state[7] = [x,y,z,roll,pitch,yaw,gripper]`, and
  `action[7] = [dx,dy,dz,droll,dpitch,dyaw,open_gripper]`.
- `CONFIRMED_CODE`: `SimplerEnvInputs(state_layout="bridge_eef7_pad_zero")`
  maps state `[7]` to exactly
  `[x,y,z,roll,pitch,yaw,0,gripper][8]`. It rejects `[8]` in this mode and
  therefore never relies on implicit broadcasting or ambiguous shape reuse.
- `CONFIRMED_CODE`: The single current/future camera is explicitly
  `observation.images.primary`; it maps to `base_0_rgb`, while both wrist
  images are same-shape zeros with false image masks.
- `CONFIRMED_CODE`: Training actions remain Bridge Euler/open-gripper values.
  `SimplerEnvOutputs(action_layout="simpler_axis_angle_binary_gripper")`
  converts each `[H,7]` policy chunk to
  `[dxyz,rotation_vector,gripper][H,7]`; gripper is exactly `-1` or `+1` at
  threshold `open_gripper > 0.5`.
- `PROPOSED_RUNTIME_GATE`: A real simulator receipt still must prove the EEF
  state extraction and gripper-state conversion before matched training or
  evaluation is authorized. Metadata and action-source inspection alone do
  not close that input-side simulator contract.

This file records tensor contracts for current pi0.5 paths and proposed scheme3 modules. All future implementation should add shape assertions at module boundaries.

## Current pi0.5 Contracts

- `CONFIRMED`: `actions`: `[B, H_action, A_dim]`, where LIBERO-plus config uses `H_action=10`; model config action dimension is projected through 32-wide action input/output projections.
- `CONFIRMED`: `noise`: same shape and dtype-compatible with `actions`.
- `CONFIRMED`: `time`: `[B]`, broadcast explicitly as `time[:, None, None]` in training.
- `CONFIRMED`: `x_t`: `[B, H_action, A_dim]`, computed as `time * noise + (1 - time) * actions`.
- `CONFIRMED`: `u_t`: `[B, H_action, A_dim]`, computed as `noise - actions`.
- `CONFIRMED`: `prefix_embs`: `[B, N_prefix, D_model]`.
- `CONFIRMED`: `suffix_embs`: `[B, H_action, D_expert]` for pi0.5 action suffix.
- `CONFIRMED`: `suffix_out[:, -H_action:]`: `[B, H_action, D_expert]`.
- `CONFIRMED`: `v_t`: `[B, H_action, A_dim]`.
- `CONFIRMED`: `estimate_action_endpoint(noisy_actions, timestep, velocity)` requires `noisy_actions=[B,H_action,A_dim]`, `timestep=[B]`, and `velocity=[B,H_action,A_dim]`; it explicitly expands `timestep` to `[B,H_action,A_dim]` and returns `action_hat=[B,H_action,A_dim]`. Evidence: `src/openpi/models_pytorch/flow_matching_utils.py`.

## Adaptive Scheme4 Router Contracts

- `CONFIRMED_CODE`: Router inputs are spatial tokens `[B,N,D_spatial]`,
  optional token mask `[B,N]`, language summary `[B,D_language]`, state
  `[B,D_state]`, and detached action intent `[B,D_expert]`. Output probability
  is `[B]` in `[0,1]`.
- `CONFIRMED_CODE`: During router-only training, sampled-flow suffix and
  initial-noise suffix both have `[B,H_action,D_expert]`. The opt-in `t1_noise`
  mode uses a detached full forward pass at `t=1` with the same Gaussian noise
  tensor used by the training suffix; legacy default `sampled_flow` uses the
  original sampled-flow suffix.
- `CONFIRMED_CODE`: Router utility target and benefit are detached `[B]`
  tensors: `benefit = base_action_loss - full_lookahead_action_loss`, with
  sigmoid soft target using configured temperature/margin. Router-only flow
  fusion scale is detached, so policy-flow loss cannot update router weights.
- `CONFIRMED_CODE`: `none` resolves to exact zero `[B]` scale and the full
  serving branch retains spatial-prefix tokens while bypassing the intent
  suffix pass and future predictor. Full-model paired fixed-noise equivalence
  remains unverified.
- `CONFIRMED_CODE`: Adaptive serving diagnostics add optional bounded `[B]`
  values `lookahead_probability` and `lookahead_scale`; legacy payloads omit
  both. `fusion_scale` is the resulting scale and may include an independent
  global multiplier, so it need not be bounded by one.

## RoboTwin 2.0 Route A Data Contracts

- `CONFIRMED`: RoboTwin LeRobot single-task datasets are addressed as `robotwin/<task>_demo_clean_50` under `HF_LEROBOT_HOME=<WORKSPACE>/robotwin_double_align/lerobot_data`. Evidence: `src/openpi/training/config.py` and `<WORKSPACE>/robotwin_double_align/lerobot_data/robotwin/adjust_bottle_demo_clean_50/meta/info.json`.
- `CONFIRMED`: Raw RoboTwin Aloha-AgileX state has shape `[14]`; raw action has shape `[14]`; LeRobot action sequence loading with pi0.5 horizon produces raw actions `[H_action=50,14]`. Evidence: `meta/info.json` and real dataloader smoke on `pi05_robotwin_adjust_bottle_geo`.
- `CONFIRMED`: After `PadStatesAndActions(model_action_dim=32)`, RoboTwin policy training tensors are `obs.state=[B,32]` and `actions=[B,50,32]`. Evidence: real dataloader smoke output `state (1, 32)` and `actions (1, 50, 32)`.
- `CONFIRMED`: RoboTwin policy-boundary input image keys are `cam_high`, `cam_left_wrist`, and `cam_right_wrist`, each accepted as `[3,H,W]` or `[H,W,3]`.
- `CONFIRMED`: `RoboTwinInputs` maps RoboTwin camera keys to pi0.5 standard image keys:
  - `cam_high -> base_0_rgb`
  - `cam_left_wrist -> left_wrist_0_rgb`
  - `cam_right_wrist -> right_wrist_0_rgb`
- `CONFIRMED`: After model transforms, current RoboTwin images have shape `[B,3,224,224]` for keys `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`. Evidence: real dataloader smoke on `pi05_robotwin_adjust_bottle_geo`.
- `CONFIRMED`: For Scheme4/Route A training, `FutureTeacherLeRobotDataset` emits RoboTwin teacher image triplets as `teacher_images` and `future_teacher_images` before repack; `RoboTwinInputs` converts them to `teacher_image` and `future_teacher_image` dicts using the same standard pi0.5 keys.
- `CONFIRMED`: After model transforms, RoboTwin teacher and future-teacher images have shape `[B,3,224,224]` for the same three standard pi0.5 image keys. Evidence: real dataloader smoke output for `teacher` and `future_teacher`.
- `CONFIRMED`: `future_teacher_fallback` has shape `[B]` and dtype bool after batching. It records whether `idx + future_teacher_obs_offset` exceeded the episode boundary and the episode final frame had to be used. Evidence: `src/openpi/training/data_loader.py`, `src/openpi/training/data_loader_test.py`, and real dataloader smoke output `fallback (1,) torch.bool [False]`.
- `CONFIRMED`: `RoboTwinOutputs` returns only the first 14 action dimensions from model output `[H_action,32]`, so deployed RoboTwin actions have shape `[50,14]`. Evidence: `src/openpi/policies/robotwin_policy.py` and `src/openpi/policies/robotwin_policy_test.py`.

## CALVIN and SimplerEnv External-Benchmark Contracts

- `CONFIRMED`: `LeRobotCalvinDataConfig` maps CALVIN LeRobot fields
  `observation.images.top`, `observation.images.wrist`, `observation.state`,
  and `action` to the policy boundary. Raw state is `[15]`; raw actions are
  `[H_action,7]`, containing six relative Cartesian/orientation dimensions
  and one discrete gripper command. `CalvinInputs` maps top/wrist to
  `base_0_rgb`/`left_wrist_0_rgb` and creates an explicit zero,
  masked-off `right_wrist_0_rgb`. Evidence:
  `src/openpi/policies/calvin_policy.py` and
  `src/openpi/policies/calvin_policy_test.py`.
- `CONFIRMED`: The distinct post-model-transform CALVIN training boundary
  applies `PadStatesAndActions(model_action_dim=32)`: for `B=1`, it produces
  `state=[1,32]` and `actions=[1,H_action=10,32]`. Prompt transforms produce
  `tokenized_prompt=[1,200]` with `tokenized_prompt_mask=[1,200]` of dtype
  bool. Raw 15/7 widths must not be asserted at this later boundary. Evidence:
  remote receipt
  `<REMOTE_WORKSPACE>/src/openpi-SF-calvin-normalized-contract-v2-20260727/artifacts/aaai27/pi/calvin_normalized_loader_contract_v2_20260727/receipt.json`.
- `CONFIRMED`: `CalvinOutputs` slices model output `[H_action,32]` to
  `[H_action,7]` and explicitly thresholds the gripper to `{-1,+1}` without
  mutating its input. This matches the verified CALVIN runtime command
  contract. Evidence: `src/openpi/policies/calvin_policy.py`,
  `artifacts/aaai27/pi/remote_harvest/calvin_debug_runtime_v9_20260724/runtime_receipt.json`.
- `CONFIRMED`: `LeRobotSimplerEnvDataConfig` uses one selected RGB stream,
  raw state `[8]`, and raw action `[H_action,7]`. `SimplerEnvInputs` maps the
  RGB stream to `base_0_rgb` and constructs explicit same-shaped zero arrays
  with `False` masks for both absent wrist views; no image slots are created by
  implicit broadcasting. Evidence:
  `src/openpi/policies/simpler_env_policy.py` and
  `src/openpi/policies/simpler_env_policy_test.py`.
- `CONFIRMED`: `FutureTeacherLeRobotDataset` recognizes CALVIN's two image
  keys and one-camera external datasets separately. It emits current/future
  teacher fields within the same episode and explicitly records terminal-frame
  fallback as scalar bool. Evidence:
  `src/openpi/training/data_loader.py` and
  `src/openpi/training/data_loader_test.py`.
- `PROPOSED`: The CALVIN protocol fixes `H_action=10` and future offset
  `10` raw frames at 10 Hz. The candidate SimplerEnv WidowX/Bridge protocol
  fixes `H_action=10` and offset `5` raw frames at 5 Hz. These become
  confirmed only after full data and evaluator validation.

## Existing Geometry Contracts

- `CONFIRMED`: `prefix_hidden`: `[B, N_prefix, D_model]`.
- `CONFIRMED`: `vision_hidden`: `[B, N_vision, D_model]`.
- `CONFIRMED`: `vision_mask`: `[B, N_vision]`.
- `CONFIRMED`: `teacher_current_hidden`: `[B, N_teacher, D_teacher]`, configured with `D_teacher=2048`.
- `CONFIRMED`: `teacher_future_hidden`: `[B, N_teacher, D_teacher]`.
- `CONFIRMED`: existing raw `teacher_delta`: `[B, N_teacher, D_teacher]`.
- `CONFIRMED`: For the sampled `pi05_libero_plus_geo` LIBERO-plus batch, current/future VGGT teacher tensors after pooling are `[1, 768, 2048]`, and `vision_mask` is `[1, 768]` with 512 valid tokens. Evidence: `artifacts/scheme3_vggt_input_checks/vggt_b1_smoke.json`.
- `CONFIRMED`: VGGT before pooling sees three image slots and produces hidden patches `[1, 3, 1369, 2048]` after removing 5 special tokens from `[1, 3, 1374, 2048]`. Evidence: `artifacts/scheme3_vggt_input_checks/vggt_b1_smoke.json`.

## Corrected Scheme2 Unified Spatial Contracts

- `CONFIRMED`: `scheme2_unified` is implemented by `Scheme2UnifiedSpatialPI0Pytorch` and `Scheme2UnifiedSpatialBottleneckAdapter`. Evidence: `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: The adapter input contract is:
  - `vision_hidden`: `[B, N_visual, D_visual]`
  - `vision_mask`: optional `[B, N_visual]`
  - `spatial_tokens S_t`: `[B, N_visual, D_teacher]`
  - `aux_tokens`: `[B, N_aux, D_visual]` for concat mode or `[B, N_visual, D_visual]` for residual-prefix mode
  - `aux_mask`: optional `[B, N_aux]` or `[B, N_visual]`
- `CONFIRMED`: In residual-prefix mode, `aux_tokens` are added only to the first `N_visual` prefix positions after explicit shape checks in the existing Scheme2 injection path. Evidence: `Scheme2VisualConcatPI0Pytorch._inject_aux_prefix_tokens`.
- `CONFIRMED`: `S_t` is the same tensor used for current VGGT alignment and for action-prefix residual projection. Evidence: `Scheme2UnifiedSpatialPI0Pytorch.forward`.
- `CONFIRMED`: The action-prefix residual is `to_action_residual(S_t) * tanh(gate).view(1, 1, D_visual)`, where `gate` is initialized exactly to zeros. Evidence: `Scheme2UnifiedSpatialBottleneckAdapter`.
- `CONFIRMED`: With `current_geo_loss_coeff=0.0`, `scheme2_unified` skips current VGGT teacher encoding and sets `current_geo_loss` to an explicit zero tensor, making it the parameter-matched no-teacher/no-spatial-loss adapter control. Evidence: `Scheme2UnifiedSpatialPI0Pytorch._scheme2_current_geo_loss`.
- `CONFIRMED`: `scheme2_unified_aux` uses the same unified spatial bottleneck but returns the original prefix embeddings unchanged, so it is the auxiliary-only spatial loss control. Evidence: `Scheme2UnifiedSpatialAuxOnlyPI0Pytorch`.
- `CONFIRMED`: Module-boundary shape checks, zero-gate identity, gradient-flow tests, deterministic eval, and tiny overfit smoke are covered by `src/openpi/models_pytorch/scheme2_unified_spatial_test.py`.

## Direct VGGT Injection Online-Teacher Contracts

- `CONFIRMED`: `direct_vggt_injection` is implemented by `DirectVGGTInjectionPI0Pytorch` and `DirectVGGTInjectionAdapter`. Evidence: `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: This variant is an online-teacher baseline, not a teacher-free final method. At training it uses trainer-provided frozen VGGT features; at inference it lazily loads VGGT from `config.vggt_weight_path` and runs VGGT on current observation images. Evidence: `DirectVGGTInjectionPI0Pytorch.forward` and `DirectVGGTInjectionPI0Pytorch.sample_actions`.
- `CONFIRMED`: The adapter input/output contract is:
  - `teacher_tokens`: `[B, N_teacher, D_teacher]`, with current VGGT hidden dim configured as `D_teacher=2048`
  - `token_mask`: optional `[B, N_teacher]`
  - `action_residual`: `[B, N_teacher, D_visual]`
  - `aux_tokens`: `[B, N_aux, D_visual]` in concat mode or `[B, N_teacher, D_visual]` in residual-prefix mode
  - `aux_mask`: optional `[B, N_aux]` or `[B, N_teacher]`
- `CONFIRMED`: The adapter computes `LayerNorm(D_teacher) -> Linear(D_teacher, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, D_visual)`, then multiplies by `tanh(gate).view(1, 1, D_visual)`. `gate` is initialized exactly to zeros. Evidence: `DirectVGGTInjectionAdapter`.
- `CONFIRMED`: In the default residual-prefix mode, `aux_tokens` are added to existing visual prefix positions through the inherited Scheme2 prefix injection path, preserving prefix token count and the existing pi0.5 KV-cache denoising structure. Evidence: `DirectVGGTInjectionPI0Pytorch.sample_actions` and `Scheme2VisualConcatPI0Pytorch._inject_aux_prefix_tokens`.
- `CONFIRMED`: `teacher_current_hidden` is detached after VGGT encoding, so VGGT is target/teacher-only and receives no gradients. Evidence: `DirectVGGTInjectionPI0Pytorch._direct_vggt_tokens`.
- `CONFIRMED`: During inference, current VGGT tokens and the augmented prefix KV cache are computed once before the flow denoising loop; VGGT is not rerun at every denoise step. Evidence: `DirectVGGTInjectionPI0Pytorch.sample_actions`.
- `CONFIRMED`: Shape checks, zero-gate identity, gradient-flow tests, deterministic eval, and tiny overfit smoke are covered by `src/openpi/models_pytorch/direct_vggt_injection_test.py`.

## Proposed Scheme3 Delta Tokenizer Contracts

- `CONFIRMED`: `delta_tokens`: `[B, M_delta, D_delta]`; default planned config remains `M_delta=8`, `D_delta=256`, and tests use smaller dimensions for speed.
- `CONFIRMED`: `DeltaEncoder(current_tokens, future_tokens, token_mask?, horizon_ids?, camera_ids?) -> delta_tokens`.
- `CONFIRMED`: `DeltaDecoder(current_tokens, delta_tokens, token_mask?, horizon_ids?, camera_ids?) -> future_recon_tokens`.
- `CONFIRMED`: `future_recon_tokens` matches `teacher_future_hidden` shape `[B, N_teacher, D_teacher]`.
- `CONFIRMED`: Implemented module checks ranks, pair shapes, mask shapes, horizon id shape/range, camera id shape/range, teacher dim, delta dim, max token count, and device consistency. Evidence: `src/openpi/models_pytorch/delta_tokenizer.py`.
- `CONFIRMED`: The decoder default `zero_init_decoder=True` makes zero delta reconstruct current tokens exactly at initialization. Evidence: `test_zero_delta_identity_with_zero_init_decoder`.
- `CONFIRMED`: The S3-M1B real smoke trained `DeltaTokenizerConfig(teacher_dim=2048, hidden_dim=256, delta_dim=256, num_delta_tokens=8, max_tokens=768, max_horizon=64)` on concatenated tensors `current_tokens=[1, 768, 2048]`, `future_tokens=[1, 768, 2048]`, `token_mask=[1, 768]`, and `horizon_ids=[1]`; `token_mask` had 512 valid tokens. Evidence: `artifacts/delta_tokenizer_smoke/s3_m1b_real_b1_steps12/summary.json`.
- `PROPOSED`: No implicit broadcasting is allowed for `time`, `horizon`, masks, or action condition tokens in future policy-coupled modules.

## Transition-Token Semantics Audit Contracts

- `CONFIRMED`: The audit does not alter the tokenizer or policy. It freezes the
  offset-10 teacher and cached VGGT tensors, and reuses the tokenizer
  checkpoint's saved train/validation indices. Evidence:
  `scripts/aaai27/pi/audit_transition_token_semantics.py` and
  `docs/decisions/2026-07-21_transition_token_semantics_audit.md`.
- `CONFIRMED`: Frozen audit inputs are:
  - `current_tokens G_t`: `[S, N_teacher, D_teacher]`
  - `future_tokens G_t+h`: `[S, N_teacher, D_teacher]`
  - `token_mask`: `[S, N_teacher]`
  - `horizon_ids`: `[S]`
  - teacher `Z = E(G_t, G_t+h)`: `[S, M_delta, D_delta]`.
- `CONFIRMED`: `MatchedFutureDecoderProbe` takes
  `current_tokens: [B,N,D_teacher]` and `delta_tokens: [B,M,D_delta]`, then
  explicitly selects one of `current_only`, `transition_only`, or
  `current_transition` before calling a fresh `DeltaDecoder`; output is
  `[B,N,D_teacher]`. Zero tensors are materialized with `zeros_like`, not
  broadcast. Evidence: `src/openpi/models_pytorch/transition_token_semantics.py`.
- `CONFIRMED`: `MatchedDeltaTokenEncoderProbe` takes
  `current_tokens/future_tokens: [B,N,D_teacher]`, explicitly selects
  `current_only`, `future_only`, or `current_future`, and produces
  `[B,M,D_delta]` through a fresh `DeltaEncoder`. Evidence: the same module.
- `CONFIRMED`: `build_swap_indices` consumes metadata id tensors `[S]` and
  returns explicit replacement indices `[S]` and validity masks `[S]`; invalid
  rows are excluded rather than silently substituted. Evidence: the same
  module and `transition_token_semantics_test.py`.
- `CONFIRMED`: Tests cover probe shapes, invalid shapes, selected-input
  gradient flow, zero-initialized decoder identity, deterministic
  initialization, deterministic swap construction, and tiny-batch overfit.
  Evidence: `src/openpi/models_pytorch/transition_token_semantics_test.py`.

## Scheme3 VGGT Feature Cache Contracts

- `CONFIRMED`: Cache manifest schema is `scheme3_vggt_feature_cache_manifest_v1`; shard schema is `scheme3_vggt_feature_cache_shard_v1`. Evidence: `scripts/delta_vggt_cache_utils.py` and `artifacts/delta_vggt_cache/s3_m2b_smoke_b1_n2/manifest.json`.
- `CONFIRMED`: Cache shard tensor contract:
  - `current_tokens`: `[S, N_vision, D_teacher]`
  - `future_tokens`: `[S, N_vision, D_teacher]`
  - `token_mask`: `[S, N_vision]`
  - `horizon_ids`: `[S]`
  - `future_teacher_fallback`: `[S]`
  - numeric identity tensors include `dataset_index`, `raw_future_index`, `episode_index`, `frame_index`, `timestamp`, `task_index`, `source_loader_batch_idx`, `source_sample_idx`, and `valid_token_counts`
  - optional auxiliary tensors currently saved by the builder include `actions: [S, H_action, A_dim]` and `state: [S, A_dim]`
- `CONFIRMED`: D2 estimated-action augmented caches add `estimated_actions: [S, H_action, A_dim]` and `estimated_action_times: [S]` without changing the core VGGT cache tensors. Evidence: `scripts/add_estimated_actions_to_delta_cache.py` and `artifacts/delta_vggt_cache/s3_d2_estact_smoke_b1_n2/manifest.json`.
- `CONFIRMED`: The S3-M2B smoke cache wrote two shards with one sample each; every sample had `current_tokens=[1,768,2048]`, `future_tokens=[1,768,2048]`, `token_mask=[1,768]`, and `valid_token_counts=[512]`. Evidence: `artifacts/delta_vggt_cache/s3_m2b_smoke_b1_n2/manifest.json`.

## Scheme3 Auxiliary Action Delta Predictor Contracts

- `CONFIRMED`: `DeltaActionPredictor(current_tokens, actions, state?, token_mask?, horizon_ids?) -> pred_delta_tokens`. Evidence: `src/openpi/models_pytorch/delta_action_predictor.py`.
- `CONFIRMED`: D1 real-cache config uses:
  - `current_tokens`: `[B, 768, 2048]`
  - `token_mask`: `[B, 768]`
  - `actions`: `[B, 10, 32]`
  - `state`: `[B, 32]`
  - `horizon_ids`: `[B]`
  - `target_delta_tokens`: `[B, 8, 256]`
  - `pred_delta_tokens`: `[B, 8, 256]`
- `CONFIRMED`: The predictor module checks current token rank/dim, action shape, optional state shape, token mask shape/device, horizon shape/range/device, target delta shape/device, and output shape. Evidence: `src/openpi/models_pytorch/delta_action_predictor.py` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -q src/openpi/models_pytorch/delta_action_predictor_test.py`.
- `CONFIRMED`: The D1 training script freezes the DeltaTokenizer, precomputes detached `target_delta_tokens`, trains only `DeltaActionPredictor`, and writes `args.json`, `predictor_config.json`, `metrics.jsonl`, `summary.json`, and `checkpoint.pt`. Evidence: `scripts/train_delta_action_predictor_from_cache.py` and `artifacts/delta_action_predictor/s3_m3_d1_gtact_n507_steps1000_seed0/summary.json`.
- `CONFIRMED`: D1 remains auxiliary-only; `pred_delta_tokens` are decoded for diagnostics but are not fed into pi0.5 action inference or training. Evidence: `scripts/train_delta_action_predictor_from_cache.py`.
- `CONFIRMED`: D1 condition ablations keep the same tensor contracts. `--condition-mode` disables current/action/state conditions by replacing the disabled input with an explicit same-shaped zero tensor before predictor forward; the frozen decoder still receives real `current_tokens` for future reconstruction. Evidence: `scripts/train_delta_action_predictor_from_cache.py`.
- `CONFIRMED`: `scripts/train_delta_action_predictor_from_cache.py --action-source estimated` keeps the same predictor tensor contract but reads the action condition from cache metadata `estimated_actions` instead of `actions`. Evidence: `artifacts/delta_action_predictor/s3_d2_estact_smoke_n2_steps2/summary.json`.

## Scheme3 D3 Policy-Fusion Contracts

- `CONFIRMED`: D3 policy fusion is implemented by `Scheme3D3FusionPI0Pytorch`, `Scheme3D3ActionDeltaAdapter`, and `Scheme3D3DeltaSuffixFusion`. Evidence: `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: D3 keeps Scheme2's prefix path and KV-cache protocol. At inference, the model builds one augmented prefix KV cache, stores predicted current tokens in `geo_cache`, and recomputes only suffix-side D3 fusion during denoising. Evidence: `Scheme3D3FusionPI0Pytorch.sample_actions` and `denoise_step` in `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: D3 policy-fusion tensors use:
  - `suffix_out`: `[B, H_action, D_suffix]`
  - `current_tokens`: `[B, N_current, D_teacher]`
  - `estimated_actions`: `[B, H_action, A_dim]`
  - `state`: `[B, 32]`
  - `token_mask`: optional `[B, N_current]`
  - `pred_delta_tokens`: `[B, M_delta, D_delta]`, default `[B, 8, 256]`
  - `refined_suffix`: `[B, H_action, D_suffix]`
  - `v_t`: `[B, H_action, A_dim]`
- `CONFIRMED`: D3 computes the action condition as `action_hat = estimate_action_endpoint(x_t, timestep, base_v_t, detach=True)`, so the estimated-action condition is stop-gradient. Evidence: `Scheme3D3FusionPI0Pytorch._scheme3_apply_delta_fusion`.
- `CONFIRMED`: `Scheme3D3DeltaSuffixFusion.gate` is initialized exactly to zeros, and the zero-gate module output is exactly `suffix_out.float()` in tests. Evidence: `test_delta_suffix_fusion_shape_and_zero_gate_identity` and `test_action_delta_adapter_shapes_and_zero_gate_identity`.
- `CONFIRMED`: D3 module-boundary shape checks are covered by tests, including bad suffix and delta-token shapes. Evidence: `src/openpi/models_pytorch/scheme3_d3_fusion_test.py`.

## Corrected Scheme3 Delta-Only Contracts

- `CONFIRMED`: Corrected Delta-only v1 is implemented by `Scheme3CorrectedDeltaPI0Pytorch`, `Scheme3DeployableCurrentAdapter`, `Scheme3DeltaQueryPredictor`, and `Scheme3CorrectedDeltaAdapter`. Evidence: `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: Corrected Delta-only v2 action-intent is implemented by `Scheme3CorrectedDeltaActionIntentPI0Pytorch`; it reuses the v1 current adapter, frozen tokenizer, learned delta queries, and zero-gated suffix fusion. Evidence: `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: This variant is teacher-free at inference. It uses frozen VGGT and a frozen `DeltaTokenizer` only during training target construction. Evidence: `Scheme3CorrectedDeltaPI0Pytorch.forward` and `Scheme3CorrectedDeltaPI0Pytorch.sample_actions`.
- `CONFIRMED`: Training-side teacher tensors use:
  - `teacher_current_hidden G_t`: `[B, N_visual, D_teacher]`, default `[B, 768, 2048]`
  - `teacher_future_hidden G_t+h`: `[B, N_visual, D_teacher]`
  - `target_delta_tokens Z_delta*`: `[B, M_delta, D_delta]`, default `[B, 8, 256]`
  - `oracle_future_recon`: `[B, N_visual, D_teacher]`
- `CONFIRMED`: Deployable current adapter contract:
  - `vision_hidden`: `[B, N_visual, D_pi]`
  - `vision_mask`: optional `[B, N_visual]`
  - `pred_current_hidden`: `[B, N_visual, D_teacher]`
  - masked visual tokens are explicitly multiplied by `vision_mask.unsqueeze(-1)`.
- `CONFIRMED`: Delta query predictor contract:
  - `source_tokens`: `[B, N_visual, D_teacher]`
  - `token_mask`: optional `[B, N_visual]`
  - `language_features`: `[B, D_pi]`
  - `state`: `[B, A_model]`
  - optional `action_intent`: `[B, D_suffix]`; required only when the predictor is configured with `action_intent_dim`.
  - learned delta queries: `[M_delta, hidden_dim]`
  - `pred_delta_tokens Z_delta_hat`: `[B, M_delta, D_delta]`
- `CONFIRMED`: In v2 action-intent, `action_intent = mean(suffix_out, dim=1).detach()`, so its shape is `[B, D_suffix]` and delta losses do not backpropagate into the frozen action-expert hidden path. Evidence: `Scheme3CorrectedDeltaActionIntentPI0Pytorch._action_intent_features`.
- `CONFIRMED`: Delta future reconstruction path decodes `D_delta(pred_current_hidden, Z_delta_hat) -> pred_future_tokens`, with `pred_future_tokens` shaped `[B, N_visual, D_teacher]`. Gradients flow through `pred_current_hidden` and `Z_delta_hat`; tokenizer parameters are frozen. Evidence: `Scheme3CorrectedDeltaPI0Pytorch._predicted_delta_and_future`.
- `CONFIRMED`: Delta suffix injection reuses `Scheme3D3DeltaSuffixFusion`:
  - `suffix_out`: `[B, H_action, D_suffix]`
  - `pred_delta_tokens`: `[B, M_delta, D_delta]`
  - `refined_suffix`: `[B, H_action, D_suffix]`
  - `v_t`: `[B, H_action, A_dim]`
  - gate is initialized exactly zero, so the suffix path is identity before training.
- `CONFIRMED`: Training loss is `L_flow + current_geo_loss_coeff * L_current_geo + delta_loss_coeff * (scheme3_delta_token_loss_coeff * L_delta_token + scheme3_delta_future_loss_coeff * L_future)`. Evidence: `Scheme3CorrectedDeltaPI0Pytorch.forward`.
- `CONFIRMED`: Inference computes `pred_current_hidden` and fixed `pred_delta_tokens` once from prefix hidden state before the denoising loop, then builds the original prefix KV cache and reuses it for suffix denoising. Evidence: `Scheme3CorrectedDeltaPI0Pytorch.sample_actions` and `denoise_step`.
- `CONFIRMED`: In v2 action-intent inference, the model first computes the original prefix KV cache, runs one initial suffix pass at `t=1` with the sampled initial noise to obtain stop-gradient `action_intent`, predicts fixed `pred_delta_tokens`, and then reuses the same prefix KV cache for the denoising loop. Evidence: `Scheme3CorrectedDeltaActionIntentPI0Pytorch.sample_actions`.
- `CONFIRMED`: Shape checks, zero-gate identity, gradient flow, deterministic eval, tiny predictor overfit, and action-intent condition checks are covered by `src/openpi/models_pytorch/scheme3_delta_corrected_test.py`.

## Scheme4 LAC Double Align Contracts

- `CONFIRMED`: Scheme4 is implemented as proposal variant `scheme4_lac_double_align` by `Scheme4LACDoubleAlignPI0Pytorch` and `Scheme4LanguageActionDeltaAdapter`. Evidence: `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: Scheme4 preserves the corrected Scheme2 spatial path:
  - frozen pi prefix hidden `H_pi`: `[B, N_prefix, D_pi]`
  - visual hidden `H_vis`: `[B, N_visual, D_pi]`
  - unified spatial tokens `S_t`: `[B, N_visual, D_teacher]`
  - spatial prefix residual aux tokens: `[B, N_visual, D_pi]` in `residual_prefix` mode
  - spatial residual gate: `[D_pi]`, initialized exactly zero.
- `CONFIRMED`: The same `S_t` tensor is used for current VGGT alignment, delta prediction input, future reconstruction decoder input, and spatial prefix residual projection. Evidence: `Scheme4LACDoubleAlignPI0Pytorch.forward`.
- `CONFIRMED`: Scheme4 delta predictor input contract is:
  - `spatial_tokens S_t`: `[B, N_visual, D_teacher]`
  - `vision_mask`: optional `[B, N_visual]`
  - `language_features`: `[B, D_pi]`, masked mean over original pi language prefix hidden tokens
  - `state`: `[B, A_model]`
  - `action_intent`: `[B, D_suffix]`, first implementation `mean(suffix_out, dim=1).detach()`
  - learned delta queries: `[M_delta, D_hidden]`
  - predicted delta tokens `Z_delta_hat`: `[B, M_delta, D_delta]`.
- `CONFIRMED`: Default Scheme4 tokenizer contract uses the offset-10 frozen tokenizer at `artifacts/delta_tokenizer_cached/s3_m4_h10_n2048_seed31_tok_m8_steps4000_lr3e4_noid/checkpoint.pt`, with `M_delta=8` and `D_delta=256`. Evidence: `src/openpi/training/config.py`.
- `CONFIRMED`: Scheme4 suffix fusion contract is:
  - `suffix_out`: `[B, H_action, D_suffix]`
  - `pred_delta_tokens`: `[B, M_delta, D_delta]`
  - `refined_suffix`: `[B, H_action, D_suffix]`
  - `delta_residual = refined_suffix - suffix_out`: `[B, H_action, D_suffix]`
  - delta fusion gate: `[D_suffix]`, initialized exactly zero.
- `CONFIRMED`: `scheme4_delta_enabled=False` returns identity suffix features and same-shaped zero delta tokens. It also freezes the Scheme4 delta adapter parameters, making the scheme4 path a delta-disabled sanity configuration close to `scheme2_unified`. Evidence: `Scheme4LanguageActionDeltaAdapter`.
- `CONFIRMED`: Scheme4 training loss is `L_flow + current_geo_loss_coeff * L_current_geo + scheme4_delta_token_loss_coeff * L_delta_token + scheme4_future_recon_loss_coeff * L_future_recon`. The first-run defaults are `current_geo_loss_coeff=0.06`, `scheme4_delta_token_loss_coeff=0.0`, and `scheme4_future_recon_loss_coeff=0.01`. Evidence: `docs/decisions/2026-07-07_scheme4_lac_double_align.md`, `src/openpi/training/config.py`, and `scripts/launch_scheme4_lac_double_align_plus_gpu0.sh`.
- `CONFIRMED`: Scheme4 inference computes `S_t`, language summary, action intent, and predicted delta tokens from current observation/state/action-noise only; it does not use VGGT, future image, future VGGT tokens, DeltaTokenizer encoder, DeltaTokenizer decoder, or GT action at inference. Evidence: `Scheme4LACDoubleAlignPI0Pytorch.sample_actions`.
- `CONFIRMED`: Scheme4 inference condition ablations are controlled by `scheme4_inference_condition_ablation` and default to `none`. They only alter the conditions passed to the delta predictor, not the main policy language/state inputs:
  - `language_features`: `[B, D_pi]`
  - `action_intent`: `[B, D_suffix]`
  - `zero_language` returns zeroed language features and unchanged action intent
  - `zero_action` returns unchanged language features and zeroed action intent
  - `zero_language_action` returns both conditions zeroed
  - `wrong_language` uses the previous-request language feature when shape-compatible, otherwise same-shaped zeros
  - `wrong_action` uses the previous-request action-intent feature when shape-compatible, otherwise same-shaped zeros
  - `shuffled_language` rolls language features across batch when `B>1`; for online `B=1` serving it falls back to previous-request language or zeros
  - `shuffled_action` rolls action-intent features across batch when `B>1`; for online `B=1` serving it falls back to previous-request action intent or zeros.
  Evidence: `apply_scheme4_inference_condition_ablation` and `src/openpi/models_pytorch/scheme4_lac_double_align_test.py`.
- `CONFIRMED`: The matched training-time action-condition control is controlled
  by `scheme4_training_action_condition` and defaults to `normal`:
  - raw action-expert-derived condition: `[B, D_suffix]`
  - `normal` returns the detached raw condition unchanged
  - `zero_action` returns explicit same-shaped zeros in both training and
    serving before delta prediction; it does not alter the base pi0.5 action
    expert, action-flow suffix, or action output projection.
  - `zero_action` rejects an action-shuffle condition-negative objective,
    because a constant zero condition has no meaningful action identity to
    shuffle.
  Evidence: `apply_scheme4_training_action_condition` and
  `Scheme4LACDoubleAlignPI0Pytorch.{forward,sample_actions}`.
- `CONFIRMED`: Scheme4 delta-content inference controls are controlled by `scheme4_inference_delta_control` and default to `normal`. They alter only the fixed predicted delta tokens stored in the denoising `geo_cache`; they do not change prefix construction, action horizon, flow steps, or any training loss:
  - `pred_delta_tokens`: `[B, M_delta, D_delta]`
  - `normal` returns the predicted delta unchanged
  - `zero_delta` returns same-shaped zeros but still runs the trained delta fusion module
  - `mean_delta` loads `scheme4_inference_delta_mean_path` as `[M_delta, D_delta]` or `[1, M_delta, D_delta]`, then explicitly expands to `[B, M_delta, D_delta]`
  - `random_delta` samples same-shaped Gaussian tokens from an explicit `scheme4_inference_control_seed`
  - `norm_matched_random_delta` samples Gaussian tokens and rescales each token to match the corresponding predicted-token L2 norm `[B, M_delta, 1]`
  - `cross_episode_delta` uses the previous-request predicted delta when shape-compatible, otherwise same-shaped zeros
  - `batch_shuffled_delta` rolls delta tokens across batch when `B>1`; for online `B=1` serving it falls back to previous-request delta or zeros.
  Evidence: `apply_scheme4_inference_delta_control`, `Scheme4LACDoubleAlignPI0Pytorch.sample_actions`, and `src/openpi/models_pytorch/scheme4_lac_double_align_test.py`.
- `CONFIRMED`: `scheme4_delta_enabled=False` and `scheme4_inference_delta_control=zero_delta` are different controls. The former bypasses the delta branch and returns identity suffix features; the latter supplies zero delta tokens to the already-trained suffix fusion, so learned fusion biases or constant responses remain testable. Evidence: `Scheme4LanguageActionDeltaAdapter.forward` and `apply_scheme4_inference_delta_control`.
- `CONFIRMED`: Scheme4 preserves the prefix KV-cache protocol: the augmented Scheme2 prefix cache is built once before denoising, action intent is computed once at `t=1`, predicted delta tokens are fixed for the denoising loop, and only suffix hidden states are recomputed per flow step. Evidence: `Scheme4LACDoubleAlignPI0Pytorch.sample_actions` and `denoise_step`.
- `CONFIRMED`: Scheme4 logs required diagnostics: `future_recon_loss`, `delta_token_loss`, `spatial_gate_abs_mean/max`, `delta_gate_abs_mean/max`, `delta_token_norm`, `delta_residual_norm`, `language_summary_norm`, `action_intent_norm`, and `language_token_residual_norm`. Evidence: `Scheme4LACDoubleAlignPI0Pytorch.forward` and `scripts/train_align_pytorch.py`.
- `CONFIRMED`: Shape checks, zero-gate identity, gradient-flow, deterministic eval, delta-disabled identity, and tiny predictor overfit tests are covered by `src/openpi/models_pytorch/scheme4_lac_double_align_test.py`.
- `CONFIRMED`: Default-disabled Scheme4 inference diagnostics preserve the
  policy output while exposing per-request mechanism tensors. At each flow
  step the exact contracts are:
  - unfused suffix hidden `H`: `[B, H_action, D_suffix]`
  - learned suffix residual `R`: `[B, H_action, D_suffix]`
  - fused suffix hidden `H + R`: `[B, H_action, D_suffix]`
  - unfused projected velocity `V_base=action_out_proj(H)`: `[B, H_action, A_dim]`
  - fused projected velocity `V_full=action_out_proj(H+R)`: `[B, H_action, A_dim]`
  - every L2 mean/max/ratio diagnostic: detached `[B]`
  - finalized flow metric vectors: detached `[B, S_flow]`
  - predicted-latent norm, action-context norm, and fusion scale: detached `[B]`.
  Exact shape equality is asserted before subtraction; no implicit tensor
  broadcasting is used. When diagnostics are disabled, the extra base
  projection is not computed. Evidence:
  `src/openpi/models_pytorch/scheme4_inference_diagnostics.py` and
  `Scheme4LACDoubleAlignPI0Pytorch.{denoise_step,sample_actions}`.
- `CONFIRMED`: Route A condition-negative diagnostics are optional and default-disabled. Tensor contracts:
  - normal predicted delta `Z_delta_hat`: `[B, M_delta, D_delta]`
  - unconditioned teacher target `Z_delta*`: `[B, M_delta, D_delta]`
  - pooled language condition: `[B, D_pi]`
  - action-intent condition: `[B, D_suffix]`
  - rolled language/action conditions preserve the original condition shape and produce a scalar `valid` flag that is false for `B=1`
  - `delta_token_distance(Z_delta_hat, Z_delta*)` returns a scalar tensor
  - margin loss `max(0, margin + d(normal) - d(negative))` returns a scalar and is exactly zero for invalid singleton batches.
  Evidence: `src/openpi/models_pytorch/scheme4_condition_diagnostics.py`, `Scheme4LACDoubleAlignPI0Pytorch._scheme4_condition_negative_loss`, and `src/openpi/models_pytorch/scheme4_lac_double_align_test.py`.
- `CONFIRMED`: Route A loss is included in `Scheme4LACDoubleAlignPI0Pytorch.forward` only when `scheme4_condition_negative_loss_coeff != 0.0`; default `0.0` preserves existing Scheme4 behavior. Evidence: `src/openpi/training/config.py` and `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.

## Matched Future-State Token Contracts

- `CONFIRMED`: The strict frame-token baseline is implemented by
  `FutureStateTokenizer` in `src/openpi/models_pytorch/future_state_tokenizer.py`.
  It uses the same `DeltaTokenizerConfig`, `DeltaEncoder`, and `DeltaDecoder`
  capacity as the transition tokenizer.
- `CONFIRMED`: Frame-token encoder input/output is:
  - `future_tokens S_t+h`: `[B, N_teacher, D_teacher]`
  - fixed zero current slot used only for architectural parity:
    `[B, N_teacher, D_teacher]`
  - `frame_tokens Z_f`: `[B, M_delta, D_delta]`.
- `CONFIRMED`: Frame-token decoder input/output is:
  - `frame_tokens Z_f`: `[B, M_delta, D_delta]`
  - explicit `num_tokens=N_teacher`, token mask, and horizon ids
  - fixed zero current reference: `[B, N_teacher, D_teacher]`
  - `future_recon`: `[B, N_teacher, D_teacher]`.
- `CONFIRMED`: No method in `FutureStateTokenizer` accepts a sample-specific
  current representation. The zero references are constructed with explicit
  shapes and are not implicit broadcast values. Future reconstruction targets
  are detached in `future_state_tokenizer_loss`.
- `CONFIRMED`: `scheme4_frame_token_lac_double_align` reuses Route A's online
  predictor and suffix fusion, but its teacher target is
  `Z_f=E_f(G_t+h)` and its future decoder ignores the current spatial tensor.
  The frame policy is exposed through explicit proposal/serving/evaluation
  registration and a tokenizer checkpoint override. Evidence:
  `src/openpi/models_pytorch/scheme4_frame_token_lac_pytorch.py`,
  `scripts/serve_policy_proposal.py`, and
  `examples/libero/run_parallel_eval.py`.
- `PROPOSED_GPU_PENDING`: The first matched policy comparison uses Route A's
  `future_recon_loss`, unweighted `delta_token_loss` diagnostic, normal
  success, and zero/norm-matched-random/cross-episode content controls. No
  semantic conclusion is valid before the matched 4k tokenizer and 10k policy
  artifacts exist.

## Conditional DeltaTokenizer Contracts

- `CONFIRMED`: The language/action conditioned tokenizer is implemented by `ConditionalDeltaTokenizer`, `ConditionalDeltaEncoder`, and `ConditionalDeltaDecoder`. Evidence: `src/openpi/models_pytorch/conditional_delta_tokenizer.py`.
- `CONFIRMED`: Conditional tokenizer input contract is:
  - `current_tokens G_t`: `[B, N_teacher, D_teacher]`
  - `future_tokens G_t+h`: `[B, N_teacher, D_teacher]`
  - `language_features`: `[B, D_language]`
  - `actions`: `[B, H_action, A_dim]`
  - `state`: `[B, D_state]`
  - `token_mask`: optional `[B, N_teacher]`
  - `horizon_ids`: optional `[B]`
  - `camera_ids`: optional `[B, N_teacher]`
- `CONFIRMED`: Conditional tokenizer output contract is:
  - `delta_tokens Z_delta`: `[B, M_delta, D_delta]`
  - `future_recon`: `[B, N_teacher, D_teacher]`
- `CONFIRMED`: Conditioning is additive in the first implementation: language summary, flattened action chunk, robot state, and horizon embedding are projected to a shared hidden vector; the condition biases encoder delta queries, encoder memory, decoder current-token queries, and decoder delta-token memory. Evidence: `src/openpi/models_pytorch/conditional_delta_tokenizer.py`.
- `CONFIRMED`: The decoder residual projection can be zero-initialized, so zero delta reconstructs current tokens exactly at initialization even when language/action/state conditions are present. Evidence: `src/openpi/models_pytorch/conditional_delta_tokenizer_test.py`.
- `CONFIRMED`: Cache-backed conditional tokenizer training requires `actions`, `state`, and `language_features` metadata by default. `--language-source zeros` exists only as an explicit smoke-test mode. Evidence: `scripts/train_conditional_delta_tokenizer_from_cache.py`.
- `CONFIRMED`: Existing offset-10 VGGT caches can be augmented with pi0.5 language summaries using `scripts/add_language_features_to_delta_cache.py`, which writes `language_features: [S, D_pi]` while preserving the original shard order and core tensors.
- `CONFIRMED`: The same cache augmentation script can optionally write token-level language features when `--store-token-features` is set:
  - `language_token_features`: `[S, L_language, D_pi]`
  - `language_token_mask`: `[S, L_language]`
  This option is explicit and off by default, so existing pooled-language cache generation remains unchanged. Evidence: `scripts/add_language_features_to_delta_cache.py`.

## Advanced Conditional DeltaTokenizer Contracts

- `CONFIRMED`: Route B advanced tokenizer is implemented by `AdvancedConditionalDeltaTokenizer`, `AdvancedConditionalDeltaEncoder`, and `AdvancedConditionalDeltaDecoder`. Evidence: `src/openpi/models_pytorch/advanced_conditional_delta_tokenizer.py`.
- `CONFIRMED`: Advanced tokenizer input contract is:
  - `current_tokens G_t`: `[B, N_teacher, D_teacher]`
  - `future_tokens G_t+h`: `[B, N_teacher, D_teacher]`
  - `language_token_features`: `[B, L_language, D_language]`
  - `language_token_mask`: optional `[B, L_language]`
  - `actions`: `[B, H_action, A_dim]`
  - `state`: `[B, D_state]`
  - `token_mask`: optional `[B, N_teacher]`
  - `horizon_ids`: optional `[B]`
  - `camera_ids`: optional `[B, N_teacher]`
- `CONFIRMED`: Advanced tokenizer output contract is:
  - `delta_tokens Z_delta`: `[B, M_delta, D_delta]`
  - `future_recon`: `[B, N_teacher, D_teacher]`
- `CONFIRMED`: Advanced tokenizer conditioning uses token-level language cross-attention, action-step cross-attention, and a global condition produced from masked language mean, action mean, state projection, and horizon embedding. FiLM/AdaLN modulation is applied in encoder/decoder residual blocks. Evidence: `src/openpi/models_pytorch/advanced_conditional_delta_tokenizer.py`.
- `CONFIRMED`: Condition dropout is train-only and preserves tensor shapes by multiplying language token features, action tokens, and state with explicit per-sample keep masks. Evidence: `_ConditionEncoder._maybe_drop` in `src/openpi/models_pytorch/advanced_conditional_delta_tokenizer.py`.
- `CONFIRMED`: Advanced negative-condition loss rolls language token features together with `language_token_mask`, rolls action tokens separately, and records shuffled-language/action reconstruction metrics. Evidence: `advanced_conditional_delta_tokenizer_loss` in `src/openpi/models_pytorch/advanced_conditional_delta_tokenizer.py`.
- `CONFIRMED`: Cache-backed Route B training entry `scripts/train_advanced_conditional_delta_tokenizer_from_cache.py` requires `actions`, `state`, `language_token_features`, and `language_token_mask` for formal cache mode; `--language-token-source pooled|zeros` are explicit smoke/ablation modes.
- `CONFIRMED`: Advanced tokenizer tests cover shapes, bad shape rejection, zero-delta identity, gradient flow to language/action/state conditions, deterministic eval under condition dropout, negative-condition metrics, and tiny overfit smoke. Evidence: `src/openpi/models_pytorch/advanced_conditional_delta_tokenizer_test.py`.

## Scheme4 Grounded LAC Contracts

- `CONFIRMED`: The grounded variant is implemented as proposal variant `scheme4_grounded_lac_double_align` in a separate source file. Evidence: `src/openpi/models_pytorch/scheme4_grounded_lac_pytorch.py` and registry entry in `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`.
- `CONFIRMED`: `scheme4_grounded_lac_double_align` inherits the first Scheme4 spatial branch, delta predictor, suffix-side zero-gated delta fusion, and teacher-free inference path.
- `CONFIRMED`: Training-side teacher target contract is:
  - `teacher_current_hidden G_t`: `[B, N_teacher, D_teacher]`
  - `teacher_future_hidden G_t+h`: `[B, N_teacher, D_teacher]`
  - `language_features L`: `[B, D_pi]`, masked mean over original pi prefix language hidden
  - `GT actions A_gt`: `[B, H_action, A_dim]`, stop-gradient and teacher-target only
  - `state`: `[B, A_dim]`
  - `target_delta_tokens Z_delta*`: `[B, M_delta, D_delta]`
- `CONFIRMED`: Student predictor contract remains the same as first Scheme4:
  - `spatial_tokens S_t`: `[B, N_visual, D_teacher]`
  - `language_features`: `[B, D_pi]`
  - `action_intent`: `[B, D_suffix]`
  - `state`: `[B, A_dim]`
  - `pred_delta_tokens Z_delta_hat`: `[B, M_delta, D_delta]`
- `CONFIRMED`: Predicted-future reconstruction uses a train-only bridge `Scheme4GroundedDecoderActionBridge(action_intent) -> decoder_actions` with shape `[B, H_action, A_dim]`; the bridge is zero-initialized and is not used at inference. Evidence: `src/openpi/models_pytorch/scheme4_grounded_lac_pytorch.py` and `src/openpi/models_pytorch/scheme4_grounded_lac_test.py`.
- `CONFIRMED`: Grounded inference uses the inherited Scheme4 `sample_actions`, so it does not call VGGT, future observations, conditional tokenizer encoder/decoder, or GT actions. Evidence: `Scheme4GroundedLACDoubleAlignPI0Pytorch.sample_actions`.
- `CONFIRMED`: Grounded training logs `scheme4_grounded_gt_action_norm`, `scheme4_grounded_decoder_action_norm`, and `scheme4_grounded_decoder_uses_student_intent` in addition to the first Scheme4 diagnostics. Evidence: `src/openpi/models_pytorch/scheme4_grounded_lac_pytorch.py` and `scripts/train_align_pytorch.py`.

## Proposed Future Scheme3 Action Condition Contracts

- `CONFIRMED`: `action_hat = x_t - time[:, None, None] * v_hat`, then `detach()`, is implemented for D2 cache augmentation. Evidence: `src/openpi/models_pytorch/flow_matching_utils.py` and `scripts/add_estimated_actions_to_delta_cache.py`.
- `CONFIRMED`: The first deployable action-intent policy condition uses a pooled suffix hidden vector `[B, D_suffix]`, not GT action tokens. Evidence: `Scheme3CorrectedDeltaActionIntentPI0Pytorch._action_intent_features`.
- `PROPOSED`: Later encoded action condition tokens, if added, should have explicit shape `[B, N_action_cond, D_delta_pred]`.
- `PROPOSED`: GT-action conditioned predictor may exist only as a diagnostic and must not feed the policy action path.

## Proposed Future Joint Flow Contracts

- `PROPOSED_DEFERRED`: If joint action-delta suffix is added later, do not reshape delta into action dimension. Use separate input projections, segment embeddings, velocity heads, and loss normalization.

## Scheme4 Offline Mean-Delta Artifact

The offline builder captures the normal deployable student prediction before
any inference delta control:

```text
pred_delta_tokens: [B_calibration, M_delta=8, D_delta=256]
```

`DeltaTokenMeanAccumulator` detaches every batch, converts it explicitly to
CPU float64, and reduces only over the calibration batch axis:

```text
mean_delta_tokens[m, d]
    = sum_b pred_delta_tokens[b, m, d] / N_calibration

mean_delta_tokens: [M_delta=8, D_delta=256], float32 CPU artifact
```

At controlled inference, the existing mean-delta loader accepts `[8,256]` or
`[1,8,256]` and explicitly expands it to `[B,8,256]`; implicit broadcasting is
not used. Future observations are omitted from the calibration dataset and the
artifact records checkpoint, dataset, calibration indices, subset/noise seeds,
sample count, and token norm statistics.
