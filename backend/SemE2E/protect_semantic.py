import argparse
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic E2E-VGuard: replace E2E Lasr with T-SemAttack Lsem."
    )
    parser.add_argument("--input_wav", required=True, help="Input wav to protect.")
    parser.add_argument("--output_wav", default=None, help="Output wav path.")
    parser.add_argument(
        "--timbre_mode",
        default="untargeted",
        choices=["untargeted", "targeted"],
        help="E2E timbre protection mode.",
    )
    parser.add_argument("--epsilon", type=float, default=8 / 255, help="Perturbation bound.")
    parser.add_argument("--epochs", type=int, default=500, help="Optimization steps.")
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument(
        "--tokenizer_path",
        default=str(ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"),
        help="Path to CosyVoice S3 speech_tokenizer_v1.onnx.",
    )
    parser.add_argument("--hubert_path", default="facebook/hubert-base-ls960")
    parser.add_argument(
        "--whisper_path",
        default="openai/whisper-small",
        help="HF Whisper model id, openai-whisper:<name>, or none.",
    )
    parser.add_argument("--no_vits", action="store_true", help="Disable VITS timbre encoder.")
    parser.add_argument("--no_gsv", action="store_true", help="Disable GPT-SoVITS timbre encoder.")
    parser.add_argument("--no_mfcc_timbre", action="store_true", help="Disable MFCC timbre branch.")
    parser.add_argument("--no_wavlm", action="store_true", help="Disable WavLM timbre encoder.")
    parser.add_argument("--no_cosyvoice", action="store_true", help="Disable CosyVoice CAM++ timbre encoder.")
    parser.add_argument("--no_style", action="store_true", help="Disable StyleTTS2 style encoder.")
    parser.add_argument("--weight_feature", type=float, default=500.0)
    parser.add_argument("--weight_semantic", type=float, default=100.0)
    parser.add_argument("--weight_psy", type=float, default=1.0e-5)
    parser.add_argument("--weight_l2", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_wav = Path(args.input_wav).resolve()
    output_wav = Path(args.output_wav).resolve() if args.output_wav else None

    os.chdir(ROOT)
    from semantic_vguard import SemanticE2EVGuard

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print("CUDA is unavailable; falling back to CPU.")

    guard = SemanticE2EVGuard(
        epsilon=args.epsilon,
        max_items=args.epochs,
        device=device,
        timbre_mode=args.timbre_mode,
        tokenizer_path=args.tokenizer_path,
        hubert_path=args.hubert_path,
        whisper_path=args.whisper_path,
        use_vits=not args.no_vits,
        use_gsv=not args.no_gsv,
        use_mfcc_timbre=not args.no_mfcc_timbre,
        use_wavlm=not args.no_wavlm,
        use_cosyvoice=not args.no_cosyvoice,
        use_style=not args.no_style,
        weight_feature=args.weight_feature,
        weight_semantic=args.weight_semantic,
        weight_psy=args.weight_psy,
        weight_l2=args.weight_l2,
    )
    result = guard.protect(
        input_wav=input_wav,
        output_wav=output_wav,
        verbose=args.verbose,
    )

    print("Protected audio:", result["output_wav"])
    print(f"SNR: {result['snr']:.3f}")
    print("Loss:", result["loss_items"])
    if result["target_speaker"]:
        print("Target speaker:", result["target_speaker"])


if __name__ == "__main__":
    main()
