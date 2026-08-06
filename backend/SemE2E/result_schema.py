from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


CST = timezone(timedelta(hours=8))

MetricValue = int | float | None


def utc_now_iso() -> str:
    now = datetime.now(CST)
    return f"{now.year}.{now.month}.{now.day} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"


def metric_source(value: MetricValue, source: str, status: str | None = None) -> dict[str, Any]:
    resolved_status = status or ("computed" if value is not None else "unavailable")
    return {"value": value, "source": source, "status": resolved_status}


def empty_primary_metrics() -> dict[str, MetricValue]:
    return {
        "wer": None,
        "cer": None,
        "tokenChangeRate": None,
        "tokenErrorRate": None,
        "semanticDrift": None,
        "speakerSimilarity": None,
        "snr": None,
        "pesq": None,
    }


def empty_details() -> dict[str, Any]:
    return {
        "generation": {
            "mode": None,
            "epsilon": None,
            "steps": None,
            "maxSteps": None,
            "selectedStep": None,
            "snrDb": None,
            "presetName": None,
            "sampleRate": None,
            "durationSec": None,
            "lossFinal": {
                "Lid": None,
                "Lfeat": None,
                "Lsem": None,
                "Lpsy": None,
                "L2": None,
                "total": None,
            },
            "lossWeights": {
                "weight_identity": None,
                "weight_feature": None,
                "weight_semantic": None,
                "weight_psy": None,
                "weight_l2": None,
                "lambdaStft": None,
                "lambdaSnr": None,
                "targetSnrDb": None,
                "selectionSnrDb": None,
            },
            "optimizationTrace": [],
            "internalOptimizationTrace": [],
            "averageStepSec": None,
            "effectiveConfig": None,
            "lossItems": None,
            "models": None,
            "checkpoints": None,
            "source": "VoiceSheild.protect",
            "status": "unavailable",
        },
        "semantic": {
            "tokenChangeRate": None,
            "tokenErrorRate": None,
            "tokenChangeCount": None,
            "tokenTotal": None,
            "semanticDrift": None,
            "encoderDistances": [
                {
                    "encoder": "S3 Tokenizer Encoder",
                    "cosineBeforeAfter": None,
                    "distance": None,
                    "status": "unavailable",
                    "source": "semantic_encoders",
                },
                {
                    "encoder": "HuBERT",
                    "cosineBeforeAfter": None,
                    "distance": None,
                    "status": "unavailable",
                    "source": "semantic_encoders",
                },
                {
                    "encoder": "Whisper",
                    "cosineBeforeAfter": None,
                    "distance": None,
                    "status": "unavailable",
                    "source": "semantic_encoders",
                },
                {
                    "encoder": "MFCC",
                    "cosineBeforeAfter": None,
                    "distance": None,
                    "status": "unavailable",
                    "source": "librosa.mfcc",
                },
            ],
            "status": "unavailable",
        },
        "asr": {
            "model": None,
            "language": None,
            "referenceText": None,
            "cleanTranscription": None,
            "protectedTranscription": None,
            "wer": None,
            "cer": None,
            "breakdown": {
                "insertRate": None,
                "deleteRate": None,
                "substituteRate": None,
            },
            "status": "unavailable",
            "source": "evaluate_asr.py",
        },
        "speaker": {
            "metric": None,
            "simBefore": None,
            "simAfter": None,
            "simDropRate": None,
            "embeddingDistanceBefore": None,
            "embeddingDistanceAfter": None,
            "simOriginalProtected": None,
            "embeddingDistance": None,
            "status": "unavailable",
            "source": "speaker_similarity.py",
        },
        "perception": {
            "snr": None,
            "pesq": None,
            "mosLqo": None,
            "l2Norm": None,
            "psychoacousticViolationRate": None,
            "maskingCurve": [],
            "status": "unavailable",
            "source": "audio_utils.py",
        },
        "downstreamTts": {
            "enabled": False,
            "ttsModel": None,
            "simCleanClone": None,
            "simProtectedClone": None,
            "simDropRate": None,
            "ttsWer": None,
            "status": "unavailable",
            "source": "evaluate_tts.py",
        },
        "cloneEval": {
            "cloneModel": None,
            "speakerEvalModel": None,
            "targetText": None,
            "originalCloneAudio": None,
            "protectedCloneAudio": None,
            "directSimilarity": None,
            "originalSimilarity": None,
            "protectedSimilarity": None,
            "similarityDropRate": None,
            "embeddingDistanceBefore": None,
            "embeddingDistanceAfter": None,
            "embeddingDistanceIncreaseRate": None,
            "cloneConfidenceBefore": None,
            "cloneConfidenceAfter": None,
            "cloneConfidenceDropRate": None,
            "cloneRadar": None,
            "cloneTrend": None,
            "cloneDefenseScore": None,
            "status": "unavailable",
        },
        "robustness": {},
    }


def empty_charts() -> dict[str, list[Any]]:
    return {
        "optimizationTrend": [],
        "psychoacoustic": [],
        "asrComparison": [],
        "ttsComparison": [],
        "chainRadar": [],
    }


def default_chains() -> list[dict[str, Any]]:
    return [
        {
            "chainId": "protect_generation",
            "chainName": "保护音频生成",
            "type": "generation",
            "status": "unavailable",
            "metrics": {},
        },
        {
            "chainId": "semantic_tokenizer_eval",
            "chainName": "语义与 Tokenizer 评估",
            "type": "semantic",
            "status": "unavailable",
            "metrics": {},
        },
        {
            "chainId": "asr_eval",
            "chainName": "ASR 转写评估",
            "type": "asr",
            "status": "unavailable",
            "metrics": {},
        },
        {
            "chainId": "speaker_eval",
            "chainName": "音色相似度评估",
            "type": "speaker",
            "status": "unavailable",
            "metrics": {},
        },
        {
            "chainId": "downstream_tts_eval",
            "chainName": "下游语音克隆评估",
            "type": "tts",
            "status": "skipped",
            "metrics": {},
        },
        {
            "chainId": "perception_eval",
            "chainName": "感知质量与心理声学评估",
            "type": "perception",
            "status": "unavailable",
            "metrics": {},
        },
    ]
