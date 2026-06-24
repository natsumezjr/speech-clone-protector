# Semantic E2E-VGuard

This directory is a standalone refactor for the CISCN project. It keeps the E2E-VGuard timbre and psychoacoustic protection pipeline, and replaces the original ASR-targeted `Lasr` objective with the T-SemAttack semantic-tokenizer objective `Lsem`.

## Layout

The submission-facing code is the top-level Python files in this directory:

- `protect_semantic.py`: CLI entrypoint for generating protected audio.
- `semantic_vguard.py`: optimization loop and E2E timbre branches.
- `semantic_encoders.py`: semantic `Lsem` ensemble, using S3 tokenizer, HuBERT, Whisper, and MFCC.
- `evaluate_asr.py`: ASR threat-model evaluation CLI.
- `evaluate_tts.py`: TTS downstream evaluation CLI.
- `evaluate_downstream.py`: compatibility wrapper for older ASR/TTS commands.
- `asr_backends.py`, `speaker_similarity.py`, `audio_utils.py`, `text_metrics.py`, `io_utils.py`, `runtime.py`: reusable evaluation/runtime utilities.
- `fetch_libritts_subset.py`, `run_semantic_batch.py`, `generate_tts_xtts.py`: dataset, batch protection, and XTTS pipeline helpers.

Unused T-SemAttack migration artifacts such as ROSETok/DAC code are intentionally
not included. The current semantic loss uses the mature `s3tokenizer` package and
the CosyVoice tokenizer checkpoint instead of maintaining a local ROSETok copy.

## Run

```bash
python protect_semantic.py --input_wav path/to/input.wav --epochs 500 --timbre_mode untargeted
```

The default output is `*_semantic.wav` next to the input file.

## Objective

```text
L = weight_feature * Lfea + weight_semantic * Lsem + weight_psy * Lpsy + weight_l2 * L2
```

- `Lfea`: E2E-VGuard timbre feature loss from VITS, GPT-SoVITS, MFCC, WavLM, CosyVoice CAM++, and StyleTTS2.
- `Lsem`: T-SemAttack semantic representation loss from S3 Tokenizer, HuBERT-Large, Whisper-Large-v3, and MFCC.
- `Lpsy`: psychoacoustic masking loss.
- `L2`: perturbation norm.

## Checkpoints

Run:

```bash
python download_models.py
```

Then manually place VITS at:

```text
checkpoints/VITS/pretrained_ljs.pth
```

The semantic tokenizer defaults to:

```text
checkpoints/CosyVoice/speech_tokenizer_v1.onnx
```

No new model training is required. The method reuses mature pretrained encoders as differentiable surrogate models.

## Small LibriTTS Evaluation

Full LibriTTS is large. For reproducible downstream evidence, fetch a small
known-transcript subset first:

```bash
python fetch_libritts_subset.py \
  --split dev.clean \
  --max_items 5 \
  --output_dir data/libritts_devclean_small
```

Run semantic protection for the subset:

```bash
python run_semantic_batch.py \
  --manifest data/libritts_devclean_small/manifest.csv \
  --output_dir outputs/libritts_devclean_small \
  --epochs 20 \
  --device cuda \
  -- --no_vits --no_style
```

Evaluate ASR against the real LibriTTS transcript:

```bash
python evaluate_asr.py manifest \
  --manifest outputs/libritts_devclean_small/protected_e20.csv \
  --asr_models openai-whisper:tiny.en \
  --device cuda \
  --output_dir outputs/eval_devclean_small_openai_whisper_tiny
```

Generate XTTS-v2 samples from clean and protected speaker references:

```bash
python generate_tts_xtts.py \
  --references outputs/libritts_devclean_small/protected_e20.csv \
  --target_text "This is a controlled downstream text to speech evaluation." \
  --output_dir outputs/tts_xtts_devclean_small \
  --limit 2 \
  --device cuda
```

Evaluate post-TTS ASR and speaker similarity:

```bash
python evaluate_tts.py \
  --manifest outputs/tts_xtts_devclean_small/tts_manifest.csv \
  --similarity_reference_manifest outputs/libritts_devclean_small/protected_e20.csv \
  --similarity_reference_mode original_clean \
  --asr_models openai-whisper:tiny.en \
  --speaker_metric ecapa \
  --speaker_model speechbrain/spkrec-ecapa-voxceleb \
  --device cuda \
  --output_dir outputs/eval_tts_xtts_devclean_small
```

## Mirrors

Prefer domestic package mirrors and the Hugging Face mirror:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv pip install --python .\.venv\Scripts\python.exe `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
  -r requirements.txt
```

XTTS-v2 is optional. In the current Windows environment it was installed without
letting Coqui downgrade the existing numeric stack:

```powershell
uv pip install --python .\.venv\Scripts\python.exe `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
  TTS==0.22.0 --no-deps

uv pip install --python .\.venv\Scripts\python.exe `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
  coqpit trainer encodec anyascii num2words pysbd gruut gruut-ipa `
  gruut-lang-en matplotlib bangla bnnumerizer bnunicodenormalizer `
  hangul-romanize g2pkk dateparser tzlocal jsonlines python-crfsuite `
  spacy==3.7.5
```

On Linux/pro, use the same packages with shell syntax:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export UV_HTTP_TIMEOUT=120
uv pip install --python /home/ljh/ciscn/.venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  TTS==0.22.0 --no-deps

uv pip install --python /home/ljh/ciscn/.venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  coqpit trainer encodec anyascii num2words pysbd gruut gruut-ipa \
  gruut-lang-en matplotlib bangla bnnumerizer bnunicodenormalizer \
  hangul-romanize g2pkk dateparser tzlocal jsonlines python-crfsuite \
  spacy==3.7.5
```
