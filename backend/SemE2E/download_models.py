import shutil
import os
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from result_adapter import SUPPORTED_TTS_MODELS

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TTS_HOME", str(ROOT / "checkpoints" / "tts"))


COQUI_TOS_MODELS = {
    "tts_models/multilingual/multi-dataset/xtts_v2",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
}


def download_files(repo_id, files, download_path):
    download_path.mkdir(parents=True, exist_ok=True)
    for file in files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=str(download_path),
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded {file}")
        except Exception as exc:
            print(f"Failed to download {file}: {exc}")


def download_coqui_tts_models():
    try:
        from TTS.utils.manage import ModelManager
    except Exception as exc:
        print(f"Skipping Coqui TTS models because TTS is not installed: {exc}")
        return

    manager = ModelManager()
    tos_agreed = os.getenv("COQUI_TOS_AGREED") == "1"
    for item in SUPPORTED_TTS_MODELS:
        model_name = str(item["backendValue"])
        if model_name in COQUI_TOS_MODELS and not tos_agreed:
            print(f"Skipping {model_name}; set COQUI_TOS_AGREED=1 after accepting Coqui terms to download it.")
            continue
        try:
            print(f"Downloading Coqui TTS model {model_name}...")
            manager.download_model(model_name)
            normalize_coqui_tts_cache(str(item["cacheName"]))
        except Exception as exc:
            print(f"Failed to download {model_name}: {exc}")


def normalize_coqui_tts_cache(cache_name):
    tts_home = Path(os.environ["TTS_HOME"])
    target = tts_home / cache_name
    nested = tts_home / "tts" / cache_name
    if not nested.exists() or target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(nested), str(target))
    print(f"Moved Coqui TTS cache to {target}")


def main():
    print("Downloading GPT-SoVITS SoVITS checkpoint...")
    download_files(
        "lj1995/GPT-SoVITS",
        ["gsv-v2final-pretrained/s2G2333k.pth"],
        ROOT / "checkpoints" / "GSV" / "base_models",
    )

    print("Downloading WavLM...")
    snapshot_download(
        repo_id="microsoft/wavlm-base-plus",
        local_dir=str(ROOT / "checkpoints" / "wavlm"),
        local_dir_use_symlinks=False,
    )

    print("Downloading CosyVoice encoders...")
    cosyvoice_dir = ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M"
    download_files(
        "FunAudioLLM/CosyVoice-300M",
        ["campplus.onnx", "speech_tokenizer_v1.onnx"],
        cosyvoice_dir,
    )

    tokenizer_src = cosyvoice_dir / "speech_tokenizer_v1.onnx"
    tokenizer_dst = ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"
    tokenizer_dst.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_src.exists() and not tokenizer_dst.exists():
        shutil.copy2(tokenizer_src, tokenizer_dst)

    print("Downloading StyleTTS2 encoder...")
    style_dir = ROOT / "checkpoints" / "StyleTTS2" / "base_models"
    download_files(
        "yl4579/StyleTTS2-LibriTTS",
        ["Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth"],
        style_dir,
    )

    nested_config = style_dir / "Models" / "LibriTTS" / "config.yml"
    nested_ckpt = style_dir / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
    if nested_config.exists():
        shutil.move(str(nested_config), str(style_dir / "config.yml"))
    if nested_ckpt.exists():
        shutil.move(str(nested_ckpt), str(style_dir / "epochs_2nd_00020.pth"))

    config_path = style_dir / "config.yml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        config["ASR_config"] = "tts_models/styletts2/Utils/ASR/config.yml"
        config["ASR_path"] = "tts_models/styletts2/Utils/ASR/epoch_00080.pth"
        config["F0_path"] = "tts_models/styletts2/Utils/JDC/bst.t7"
        config["PLBERT_dir"] = "tts_models/styletts2/Utils/PLBERT"

        with config_path.open("w", encoding="utf-8") as file:
            yaml.dump(config, file, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print("VITS is not hosted here. Put pretrained_ljs.pth at checkpoints/VITS/pretrained_ljs.pth.")

    print("Downloading supported Coqui TTS clone backends...")
    download_coqui_tts_models()


if __name__ == "__main__":
    main()
