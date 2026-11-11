"""Train pi0.5 LIBERO proposal variants without modifying the baseline trainer.

Usage example:

    uv run scripts/train_pi05_libero_proposals_pytorch.py \
        --proposal-variant scheme1 \
        pi05_libero_geo --exp-name scheme1_smoke ...
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from openpi.models_pytorch import pi0_align_pytorch
from openpi.models_pytorch.pi0_proposal_variants_pytorch import get_proposal_model_class


def _load_base_train_module():
    script_path = Path(__file__).resolve().parent / "train_align_pytorch.py"
    spec = importlib.util.spec_from_file_location("_openpi_train_align_pytorch", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base trainer from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base_train = _load_base_train_module()


def _pop_proposal_variant(argv: list[str]) -> str:
    variant = "scheme1"
    cleaned = [argv[0]]
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--proposal-variant":
            if idx + 1 >= len(argv):
                raise ValueError("--proposal-variant requires a value")
            variant = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--proposal-variant="):
            variant = arg.split("=", 1)[1]
            idx += 1
            continue
        cleaned.append(arg)
        idx += 1
    argv[:] = cleaned
    return variant


def main() -> None:
    variant = _pop_proposal_variant(sys.argv)
    model_cls = get_proposal_model_class(variant)
    pi0_align_pytorch.PI0Pytorch = model_cls
    _base_train.init_logging()
    config = _base_train._config.cli()  # noqa: SLF001
    _base_train.logging.info("Using proposal variant: %s (%s)", variant, model_cls.__name__)
    _base_train.train_loop(config)


if __name__ == "__main__":
    main()
