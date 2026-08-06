import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    def __init__(self, values):
        super().__init__()
        for key, value in values.items():
            self[key] = Config(value) if isinstance(value, dict) else value

    __getattr__ = dict.__getitem__


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as file:
        return Config(json.load(file))


def _prepare_vits_imports() -> None:
    vits_dir = str(ROOT / "tts_models" / "vits")
    sys.path.insert(0, vits_dir)
    for name in ("commons", "modules", "attentions", "transforms", "monotonic_align"):
        sys.modules.pop(name, None)


def build_models_vits(hparams, checkpoint_path: str):
    _prepare_vits_imports()
    from tts_models.vits.models import SynthesizerTrn
    from tts_models.vits.text.symbols import symbols

    model = SynthesizerTrn(
        len(symbols),
        hparams.data.filter_length // 2 + 1,
        hparams.train.segment_size // hparams.data.hop_length,
        n_speakers=hparams.data.n_speakers,
        **hparams.model,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return model
