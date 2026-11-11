# Scheme4 LAC Double Align Execution

本文档记录 `scheme4_lac_double_align` 首轮实验如何在本机和另一台 3x40G A100 服务器上执行。所有命令默认在仓库根目录 `<SGLL_ROOT>` 下运行。

## 目标

- `CONFIRMED`: 当前正式实现的 variant 名称是 `scheme4_lac_double_align`。代码入口是 `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py`。
- `CONFIRMED`: 首轮主实验使用 `pi05_libero_plus_geo`、官方 pi0.5-libero PyTorch checkpoint、冻结 VLM 和 action expert、LIBERO-plus future offset 10、offset-10 DeltaTokenizer。
- `CONFIRMED`: 首轮 loss 系数是 `current_geo_loss_coeff=0.06`、`scheme4_future_recon_loss_coeff=0.01`、`scheme4_delta_token_loss_coeff=0.0`。
- `CONFIRMED`: 推理阶段不使用 VGGT、未来图像、未来 VGGT token、DeltaTokenizer encoder/decoder 或 GT action。

## 必要文件

本机已有关键文件：

```text
checkpoints/pi05_libero_official_torch
artifacts/delta_tokenizer_cached/s3_m4_h10_n2048_seed31_tok_m8_steps4000_lr3e4_noid/checkpoint.pt
<WORKSPACE>/hf_lerobot/Sylvest/libero_plus_lerobot
<WORKSPACE>/LIBERO-plus/libero/libero/assets
<WORKSPACE>/LIBERO-plus/libero/libero/benchmark/task_classification.json
```

如果在另一台 A100 服务器执行，需要至少同步这些路径：

```bash
rsync -avh <SGLL_ROOT>/ <A100_HOST>:<SGLL_ROOT>/
rsync -avh <WORKSPACE>/hf_lerobot/Sylvest/libero_plus_lerobot/ <A100_HOST>:<WORKSPACE>/hf_lerobot/Sylvest/libero_plus_lerobot/
rsync -avh <WORKSPACE>/LIBERO-plus/ <A100_HOST>:<WORKSPACE>/LIBERO-plus/
```

如果 A100 服务器上的绝对路径不同，需要同步后相应修改 launcher 中的：

```text
CLASSIFICATION_JSON
--libero-pythonpath
```

## 本机优先运行

本机优先跑主实验，因为这是最重要的判断实验：

```bash
tmux new -s pi_scheme4_lac_gpu0
cd <SGLL_ROOT>
GPU=0 PORT=8240 bash scripts/launch_scheme4_lac_double_align_plus_gpu0.sh
```

主要输出：

```text
logs/proposals/scheme4_lac_double_align_plus_official_frozen_b16_cg006_fut001_tok0_h10_10k.log
checkpoints/pi05_libero_plus_geo/scheme4_lac_double_align_plus_official_frozen_b16_cg006_fut001_tok0_h10_10k/10000/model.safetensors
logs/eval/scheme4_lac_double_align_plus_10k_spatial_full_pipeline.log
data/libero/results/scheme4_lac_double_align_plus_10k_spatial_full_summary.csv
```

训练中重点看这些指标：

```text
flow_loss
current_geo_loss
future_recon_loss
delta_token_loss
spatial_gate_abs_mean / spatial_gate_max_abs
delta_gate_abs_mean / delta_gate_max_abs
delta_token_norm
delta_residual_norm
language_summary_norm
action_intent_norm
language_token_residual_norm
scheme4_future_fallback_ratio
```

其中 `language_token_residual_norm` 应该保持 `0.0`，因为 scheme4 不直接改 language token。

本机 5090 上 `BATCH_SIZE=16` 已能启动，但显存非常贴边；如果后续 OOM，使用下面设置重跑以保持有效 batch 16：

```bash
BATCH_SIZE=8 GRAD_ACCUM=2 GPU=1 PORT=8242 bash scripts/launch_scheme4_lac_double_align_plus_gpu0.sh
```

## A100 服务器建议并行任务

如果 A100 服务器有 3 张 40G 卡，建议把不如主实验关键、但能节约本机时间的任务放过去。

### GPU0: Scheme4 Main

如果本机没有启动主实验，也可以在 A100 GPU0 跑同一个主实验：

```bash
tmux new -s pi_scheme4_lac_gpu0
cd <SGLL_ROOT>
GPU=0 PORT=8240 bash scripts/launch_scheme4_lac_double_align_plus_gpu0.sh
```

### GPU1: Delta-Disabled Sanity

这个实验验证 scheme4 代码路径在关闭 delta 后是否尽量接近 `scheme2_unified`。建议先跑 1-step smoke，确认能保存 checkpoint；如 smoke 通过，再跑 10k。

1-step smoke：

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python scripts/train_pi05_libero_proposals_pytorch.py \
  --proposal-variant scheme4_lac_double_align \
  pi05_libero_plus_geo \
  --exp-name scheme4_lac_double_align_delta_disabled_smoke \
  --overwrite \
  --no-wandb-enabled \
  --num-train-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 \
  --pytorch-weight-path ./checkpoints/pi05_libero_official_torch \
  --no-train-action-expert \
  --current-geo-loss-coeff 0.06 \
  --scheme2-injection-mode residual_prefix \
  --no-scheme4-delta-enabled \
  --scheme4-delta-token-loss-coeff 0.0 \
  --scheme4-future-recon-loss-coeff 0.0 \
  --scheme4-horizon-id 10 \
  --data.future-teacher-obs-offset 10 \
  --log-interval 1 \
  --save-interval 1
```

10k sanity 训练可以把上面改成：

```text
--exp-name scheme4_lac_double_align_delta_disabled_plus_official_frozen_b16_cg006_h10_10k
--num-train-steps 10000
--batch-size 16
--num-workers 4
--log-interval 100
--save-interval 1000
```

评测命令：

```bash
uv run --no-sync python examples/libero/run_parallel_eval.py \
  --run-name scheme4_lac_double_align_delta_disabled_plus_10k_spatial_full \
  --task-suite-name libero_spatial \
  --task-start 0 \
  --num-tasks 2402 \
  --num-workers 3 \
  --num-trials-per-task 1 \
  --seed 7 \
  --port 8241 \
  --gpu 1 \
  --policy-config pi05_libero_plus_geo \
  --policy-dir checkpoints/pi05_libero_plus_geo/scheme4_lac_double_align_delta_disabled_plus_official_frozen_b16_cg006_h10_10k/10000 \
  --proposal-variant scheme4_lac_double_align \
  --policy-data-future-teacher-obs-offset 10 \
  --mujoco-egl-device-id 1 \
  --server-ready-timeout-sec 900 \
  --no-save-videos

uv run --no-sync python examples/libero/summarize_jsonl_results.py \
  --results-jsonl data/libero/results/scheme4_lac_double_align_delta_disabled_plus_10k_spatial_full \
  --task-classification <WORKSPACE>/LIBERO-plus/libero/libero/benchmark/task_classification.json \
  --suite libero_spatial \
  --out-summary-csv data/libero/results/scheme4_lac_double_align_delta_disabled_plus_10k_spatial_full_summary.csv
```

### GPU2: No Future-Loss Flow-Only Delta Control

这个实验保留 delta branch 和 flow 注入，但关闭 token/future auxiliary loss，用来判断未来重建辅助是否和 flow 冲突。

```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python scripts/train_pi05_libero_proposals_pytorch.py \
  --proposal-variant scheme4_lac_double_align \
  pi05_libero_plus_geo \
  --exp-name scheme4_lac_double_align_flowonly_delta_plus_official_frozen_b16_cg006_h10_10k \
  --overwrite \
  --no-wandb-enabled \
  --num-train-steps 10000 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --pytorch-weight-path ./checkpoints/pi05_libero_official_torch \
  --no-train-action-expert \
  --current-geo-loss-coeff 0.06 \
  --scheme2-injection-mode residual_prefix \
  --scheme4-tokenizer-checkpoint artifacts/delta_tokenizer_cached/s3_m4_h10_n2048_seed31_tok_m8_steps4000_lr3e4_noid/checkpoint.pt \
  --scheme4-delta-token-loss-coeff 0.0 \
  --scheme4-future-recon-loss-coeff 0.0 \
  --scheme4-horizon-id 10 \
  --data.future-teacher-obs-offset 10 \
  --log-interval 100 \
  --save-interval 1000
```

对应评测时把 `--policy-dir` 和 `--run-name` 改成 `flowonly_delta` 即可。

## 快速检查命令

查看训练是否还在跑：

```bash
nvidia-smi
tmux ls
tail -n 80 logs/proposals/scheme4_lac_double_align_plus_official_frozen_b16_cg006_fut001_tok0_h10_10k.log
```

查看全量评测是否完成：

```bash
ls -lh data/libero/results/scheme4_lac_double_align_plus_10k_spatial_full_summary.csv
tail -n 80 logs/eval/scheme4_lac_double_align_plus_10k_spatial_full_pipeline.log
```

对比核心 baseline：

```text
official pi0.5-libero: data/libero/results/pi05_official_torch_plus_spatial_3w_summary.csv
scheme2_unified: data/libero/results/scheme2_unified_plus_10k_spatial_full_summary.csv
scheme2_unified_nogeo: data/libero/results/scheme2_unified_nogeo_plus_10k_spatial_full_summary.csv
scheme3_delta_intent_h10tok10: data/libero/results/scheme3_delta_intent_h10tok10_plus_10k_spatial_full_summary.csv
```

## 判断标准

- `CONFIRMED_TARGET`: 首轮最重要结果是 `scheme4_lac_double_align_plus_10k_spatial_full_summary.csv`。
- `PROPOSED`: 如果 scheme4 接近或超过 `scheme2_unified`，下一步优先做 normal/zero/shuffled delta 和 input ablation，证明收益来自 delta 内容而不是参数量。
- `PROPOSED`: 如果 scheme4 明显低于 `scheme2_unified`，先看 `delta_gate_abs_mean`、`delta_residual_norm`、`future_recon_loss`、`language_token_residual_norm` 和 `action_intent_norm`，再决定是降 loss、换 action intent，还是改 fusion 位置。
