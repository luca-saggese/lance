"""
Lance OpenAI-Compatible API Server
====================================
FastAPI server che espone un'interfaccia OpenAI Chat Completions per il modello Lance.

Task supportati (selezionati tramite il campo 'model' nella richiesta):
  lance-t2i   : Text-to-Image
  lance-t2v   : Text-to-Video
  lance-i2i   : Image Editing  (instruction + immagine di input → immagine)
  lance-v2v   : Video Editing  (instruction + video di input → video)
  lance-i2t   : Image Understanding (immagine + domanda → testo)
  lance-v2t   : Video Understanding (video + domanda → testo)

Formato input (OpenAI chat completions):
  {
    "model": "lance-t2i",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "A beautiful landscape"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}},
    ]}],
    // parametri opzionali
    "seed": 42,
    "num_frames": 50,
    "video_height": 480,
    "video_width": 848,
    "resolution": "video_480p",
    "num_timesteps": 30,
    "timestep_shift": 3.5,
    "cfg_scale": 4.0,
    "use_kvcache": true
  }

  oppure con il formato 'input' (OpenAI Responses API):
  {
    "model": "lance-t2i",
    "input": [{"content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "..."}},
    ]}],
    ...
  }

Formato output (OpenAI-compatibile):
  {
    "id": "gen-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "lance-t2i",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,             // task di understanding: stringa
        "images": [                  // task di generazione immagini
          {"imageUrl": {"url": "data:image/png;base64,..."}}
        ],
        "videos": [                  // task di generazione video
          {"videoUrl": {"url": "data:video/mp4;base64,..."}}
        ]
      },
      "finish_reason": "stop"
    }]
  }

Avvio (i modelli vengono scaricati automaticamente se non presenti):
  python lance_openai_server.py

  Opzioni principali:
    --port 8000                   Porta (default 8000)
    --gpu-image 0                 GPU per image pipeline
    --gpu-video 0                 GPU per video pipeline
    --preload                     Carica i modelli subito invece che al primo request
    --disable-image-pipeline      Disabilita i task immagine (t2i/i2i/i2t)
    --disable-video-pipeline      Disabilita i task video (t2v/v2v/v2t)
    --no-download                 Non scaricare automaticamente i modelli
    --model-path-image PATH       Override path modello immagine
    --model-path-video PATH       Override path modello video
    --downloads-dir PATH          Directory dove salvare i pesi (default: downloads/)
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
import mimetypes
import os
import random
import shutil
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# ── Lance internals ───────────────────────────────────────────────────────────
from safetensors.torch import load_file
from transformers import set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from common.utils.logging import get_logger
from common.utils.misc import AutoEncoderParams, tuple_mul
from config.config_factory import DataArguments, InferenceArguments, ModelArguments
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom import ValidationDataset
from inference_lance import (
    PROMPT_JSON_FILENAME,
    TASK_IMAGE_EDIT,
    TASK_T2I,
    TASK_T2V,
    TASK_VIDEO_EDIT,
    TASK_X2T_IMAGE,
    TASK_X2T_VIDEO,
    apply_inference_defaults,
    clean_memory,
    init_from_model_path_if_needed,
    save_prompt_results,
    validate_on_fixed_batch,
)
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
SERVER_TMP_ROOT = REPO_ROOT / "tmps" / "openai_server"
TMP_INPUT_DIR = SERVER_TMP_ROOT / "inputs"
RESULTS_ROOT = SERVER_TMP_ROOT / "results"

# HuggingFace repo da cui scaricare tutti i pesi
HF_REPO_ID = "bytedance-research/Lance"
# Percorsi di default dei modelli (relativi a REPO_ROOT)
DEFAULT_DOWNLOADS_DIR = REPO_ROOT / "downloads"
DEFAULT_MODEL_PATH_IMAGE = DEFAULT_DOWNLOADS_DIR / "Lance_3B"
DEFAULT_MODEL_PATH_VIDEO = DEFAULT_DOWNLOADS_DIR / "Lance_3B_Video"

DEFAULT_VIT_TYPE = "qwen_2_5_vl_original"
DEFAULT_TIMESTEPS = 30
DEFAULT_TIMESTEP_SHIFT = 3.5
DEFAULT_CFG_TEXT_SCALE = 4.0
USE_KVCACHE = True
TEXT_TEMPLATE = True

IMAGE_TASKS = {TASK_T2I, TASK_IMAGE_EDIT, TASK_X2T_IMAGE}
VIDEO_TASKS = {TASK_T2V, TASK_VIDEO_EDIT, TASK_X2T_VIDEO}

# Default resolution per task
TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    TASK_T2I: {
        "resolution": "image_768res",
        "video_height": 768,
        "video_width": 768,
        "num_frames": 1,
    },
    TASK_T2V: {
        "resolution": "video_480p",
        "video_height": 480,
        "video_width": 848,
        "num_frames": 50,
    },
    TASK_IMAGE_EDIT: {
        "resolution": "image_768res",
        "video_height": 768,
        "video_width": 768,
        "num_frames": 1,
    },
    TASK_VIDEO_EDIT: {
        "resolution": "video_480p",
        "video_height": 480,
        "video_width": 848,
        "num_frames": 50,
    },
    TASK_X2T_IMAGE: {
        "resolution": "image_768res",
        "video_height": 768,
        "video_width": 768,
        "num_frames": 1,
    },
    TASK_X2T_VIDEO: {
        "resolution": "video_480p",
        "video_height": 480,
        "video_width": 848,
        "num_frames": 50,
    },
}

I2T_SYSTEM_PROMPT = "Look at the image carefully and answer the question."
V2T_SYSTEM_PROMPT = "Watch the video carefully and answer the question."

# ─────────────────────────────────────────────────────────────────────────────
# Download automatico dei modelli
# ─────────────────────────────────────────────────────────────────────────────


def _is_lance_model_ready(model_path: Path) -> bool:
    """Restituisce True se i file essenziali del modello Lance sono presenti."""
    return (model_path / "llm_config.json").exists() and (
        (model_path / "ema.safetensors").exists()
        or (model_path / "model.safetensors").exists()
    )


def _is_vit_ready(downloads_dir: Path) -> bool:
    """Restituisce True se i pesi del ViT sono presenti."""
    vit_dir = downloads_dir / "Qwen2.5-VL-ViT"
    return (vit_dir / "vit.safetensors").exists()


def ensure_models_downloaded(
    image_path: Path,
    video_path: Path,
    downloads_dir: Path,
    need_image: bool = True,
    need_video: bool = True,
) -> None:
    """
    Verifica che i pesi necessari siano presenti; se mancano, scarica
    l'intero repo ``bytedance-research/Lance`` da HuggingFace Hub.

    Il download è incrementale (resume_download=True) e viene saltato
    completamente se tutti i file richiesti esistono già.
    """
    missing: list[str] = []
    if need_image and not _is_lance_model_ready(image_path):
        missing.append(f"image model ({image_path})")
    if need_video and not _is_lance_model_ready(video_path):
        missing.append(f"video model ({video_path})")
    if not _is_vit_ready(downloads_dir):
        missing.append(f"ViT ({downloads_dir / 'Qwen2.5-VL-ViT'})")

    if not missing:
        print("[lance_server] Tutti i modelli sono già presenti. Nessun download necessario.")
        return

    print("[lance_server] File mancanti:")
    for m in missing:
        print(f"  - {m}")
    print(
        f"[lance_server] Avvio download da HuggingFace: {HF_REPO_ID}\n"
        f"[lance_server] Destinazione: {downloads_dir}\n"
        "[lance_server] (può richiedere diversi minuti a seconda della connessione)"
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub non è installato. "
            "Esegui: pip install huggingface_hub"
        ) from exc

    cache_dir = downloads_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_REPO_ID,
        cache_dir=str(cache_dir),
        local_dir=str(downloads_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.py",
            "*.md",
            "*.txt",
            "*.pth",
        ],
    )
    print("[lance_server] Download completato.")

    # Verifica post-download
    still_missing: list[str] = []
    if need_image and not _is_lance_model_ready(image_path):
        still_missing.append(str(image_path))
    if need_video and not _is_lance_model_ready(video_path):
        still_missing.append(str(video_path))
    if still_missing:
        raise RuntimeError(
            "Download completato ma i seguenti path non contengono i file attesi:\n"
            + "\n".join(f"  - {p}" for p in still_missing)
            + "\nVerifica la struttura della repo HuggingFace."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models per la request/response
# ─────────────────────────────────────────────────────────────────────────────


class ImageUrl(BaseModel):
    url: str


class VideoUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None
    video_url: Optional[VideoUrl] = None


class Message(BaseModel):
    role: str = "user"
    content: Union[str, List[ContentPart]] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "lance"
    # Supporta sia "messages" (OpenAI standard) sia "input" (Responses API)
    messages: Optional[List[Message]] = None
    input: Optional[List[Message]] = None

    # Parametri di generazione opzionali
    seed: Optional[int] = None
    num_frames: Optional[int] = None
    video_height: Optional[int] = None
    video_width: Optional[int] = None
    resolution: Optional[str] = None
    num_timesteps: Optional[int] = None
    timestep_shift: Optional[float] = None
    cfg_scale: Optional[float] = None
    use_kvcache: Optional[bool] = None

    # Campi OpenAI standard ignorati ma accettati per compatibilità
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def require_messages_or_input(self) -> "ChatCompletionRequest":
        if self.messages is None and self.input is None:
            raise ValueError("Almeno uno tra 'messages' e 'input' è obbligatorio.")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Utilitiy: gestione media (base64 / URL / percorso)
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_dirs() -> None:
    TMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def _download_url(url: str, dest: Path) -> None:
    """Scarica un URL (anche di grandi dimensioni) in dest."""
    import urllib.request

    with urllib.request.urlopen(url) as resp:  # noqa: S310
        with dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)


def resolve_media(url_or_b64: str, media_type: str, save_dir: Path) -> Path:
    """
    Risolve un URL o una stringa base64 in un file locale.

    Args:
        url_or_b64: URL http/https, percorso locale, oppure data URI base64.
        media_type: "image" o "video" (usato per determinare l'estensione).
        save_dir:   directory temporanea dove salvare il file.

    Returns:
        Path del file locale.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Base64 data URI ────────────────────────────────────────────────────
    if url_or_b64.startswith("data:"):
        header, b64data = url_or_b64.split(",", 1)
        mime = header.split(";")[0].split(":")[1]  # e.g. image/jpeg
        ext = mimetypes.guess_extension(mime) or (".jpg" if media_type == "image" else ".mp4")
        # guess_extension può restituire '.jpe' → normalizza
        if ext in (".jpe", ".jpeg"):
            ext = ".jpg"
        dest = save_dir / f"input_{uuid.uuid4().hex}{ext}"
        dest.write_bytes(base64.b64decode(b64data))
        return dest

    # ── HTTP/HTTPS URL ────────────────────────────────────────────────────
    if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
        ext = Path(url_or_b64.split("?")[0]).suffix or (".jpg" if media_type == "image" else ".mp4")
        dest = save_dir / f"input_{uuid.uuid4().hex}{ext}"
        _download_url(url_or_b64, dest)
        return dest

    # ── Percorso locale ────────────────────────────────────────────────────
    local = Path(url_or_b64)
    if local.exists():
        return local
    raise ValueError(f"Impossibile risolvere il media: {url_or_b64[:80]!r}")


def encode_file_as_data_url(path: Path) -> str:
    """Legge un file e lo restituisce come data URI base64."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# Task detection
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_TO_TASK: Dict[str, str] = {
    "lance-t2i": TASK_T2I,
    "lance-t2v": TASK_T2V,
    "lance-i2i": TASK_IMAGE_EDIT,
    "lance-image-edit": TASK_IMAGE_EDIT,
    "lance-v2v": TASK_VIDEO_EDIT,
    "lance-video-edit": TASK_VIDEO_EDIT,
    "lance-i2t": TASK_X2T_IMAGE,
    "lance-x2t-image": TASK_X2T_IMAGE,
    "lance-v2t": TASK_X2T_VIDEO,
    "lance-x2t-video": TASK_X2T_VIDEO,
}


def detect_task(
    model_name: str,
    has_image: bool,
    has_video: bool,
) -> str:
    """
    Determina il task da eseguire in base al nome del modello e ai media in input.
    Fallback automatico quando il modello è generico ("lance", "lance-3b", ecc.).
    """
    model_lower = model_name.strip().lower()

    if model_lower in _MODEL_TO_TASK:
        return _MODEL_TO_TASK[model_lower]

    # Auto-detection dal contenuto
    if has_video:
        return TASK_X2T_VIDEO  # default quando c'è solo video senza tag esplicito
    if has_image:
        return TASK_X2T_IMAGE  # default quando c'è solo immagine
    # Solo testo: text-to-image di default
    return TASK_T2I


# ─────────────────────────────────────────────────────────────────────────────
# Costruzione del JSON di prompt per il dataset
# ─────────────────────────────────────────────────────────────────────────────


def build_prompt_file(
    task: str,
    prompt: str,
    media_path: Optional[Path],
    question: str,
    save_dir: Path,
) -> Path:
    """
    Crea il file JSON di input compatibile con ValidationDataset.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = save_dir / "prompt_input.json"

    if task == TASK_T2I:
        payload = {"000000.png": prompt}

    elif task == TASK_T2V:
        payload = {"000000.mp4": prompt}

    elif task == TASK_IMAGE_EDIT:
        if media_path is None:
            raise ValueError("image_edit richiede un'immagine in input.")
        img_str = str(media_path)
        payload = {
            "000000": {
                "interleave_array": [prompt, img_str, img_str],
                "element_dtype_array": ["text", "image", "image"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }

    elif task == TASK_VIDEO_EDIT:
        if media_path is None:
            raise ValueError("video_edit richiede un video in input.")
        vid_str = str(media_path)
        payload = {
            "000000": {
                "interleave_array": [prompt, vid_str, vid_str],
                "element_dtype_array": ["text", "video", "video"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }

    elif task == TASK_X2T_IMAGE:
        if media_path is None:
            raise ValueError("x2t_image richiede un'immagine in input.")
        q = question or prompt or "Describe the image."
        payload = {
            "000000": {
                "interleave_array": [str(media_path), [I2T_SYSTEM_PROMPT, q, ""]],
                "element_dtype_array": ["image", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }

    elif task == TASK_X2T_VIDEO:
        if media_path is None:
            raise ValueError("x2t_video richiede un video in input.")
        q = question or prompt or "Describe the video."
        payload = {
            "000000": {
                "interleave_array": [str(media_path), [V2T_SYSTEM_PROMPT, q, ""]],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }

    else:
        raise ValueError(f"Task non supportato: {task}")

    prompt_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompt_file


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class LancePipeline:
    """
    Pipeline riusabile che carica il modello Lance una volta sola e
    gestisce l'inferenza per richieste successive.

    Adattato da LanceT2VV2TPipeline in lance_gradio_t2v_v2t.py per
    supportare tutti i 6 task (t2i, t2v, i2i, v2v, i2t, v2t).
    """

    def __init__(self, model_path: str, device_id: int, default_task: str = TASK_T2V) -> None:
        self._init_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.initialized = False
        self.model_path = model_path
        self.device = device_id
        self.default_task = default_task
        self.logger = get_logger(f"lance_server_gpu{device_id}")

        self.model: Optional[Lance] = None
        self.vae_model: Optional[WanVideoVAE] = None
        self.vae_config: Optional[AutoEncoderParams] = None
        self.tokenizer: Optional[Qwen2Tokenizer] = None
        self.new_token_ids: Optional[dict] = None
        self.image_token_id: Optional[int] = None
        self.base_model_args: Optional[ModelArguments] = None
        self.base_inference_args: Optional[InferenceArguments] = None

    # ── Initialization ────────────────────────────────────────────────────

    def _build_base_model_args(self) -> ModelArguments:
        return ModelArguments(
            model_path=self.model_path,
            vit_type=DEFAULT_VIT_TYPE,
            llm_qk_norm=True,
            llm_qk_norm_und=True,
            llm_qk_norm_gen=True,
            tie_word_embeddings=False,
            max_num_frames=121,
            max_latent_size=64,
            latent_patch_size=[1, 1, 1],
        )

    def _build_base_inference_args(self) -> InferenceArguments:
        td = TASK_DEFAULTS[self.default_task]
        return InferenceArguments(
            validation_num_timesteps=DEFAULT_TIMESTEPS,
            validation_timestep_shift=DEFAULT_TIMESTEP_SHIFT,
            copy_init_moe=True,
            visual_und=True,
            visual_gen=True,
            vae_model_type="wan",
            apply_qwen_2_5_vl_pos_emb=True,
            apply_chat_template=False,
            cfg_type=0,
            validation_data_seed=42,
            video_height=td["video_height"],
            video_width=td["video_width"],
            num_frames=td["num_frames"],
            task=self.default_task,
            save_path_gen=str(RESULTS_ROOT),
            resolution=td["resolution"],
            text_template=TEXT_TEMPLATE,
            use_KVcache=USE_KVCACHE,
        )

    def initialize(self) -> None:
        with self._init_lock:
            if self.initialized:
                return

            _ensure_dirs()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA non disponibile. Lance richiede una GPU.")
            if self.device >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {self.device} non disponibile. Rilevate {torch.cuda.device_count()} GPU."
                )
            torch.cuda.set_device(self.device)

            model_args = self._build_base_model_args()
            data_args = DataArguments()
            inference_args = self._build_base_inference_args()
            apply_inference_defaults(model_args, data_args, inference_args)
            inference_args.validation_noise_seed = inference_args.validation_data_seed

            self.base_model_args = model_args
            self.base_inference_args = inference_args

            set_seed(inference_args.global_seed)

            t0 = time.perf_counter()
            print(f"[lance_server][gpu:{self.device}] Carico LLM config: {model_args.model_path}/llm_config.json")
            llm_config: Qwen2Config = Qwen2Config.from_json_file(
                str(Path(model_args.model_path) / "llm_config.json")
            )

            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.qk_norm_und = model_args.llm_qk_norm_und
            llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = inference_args.freeze_und
            llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

            print(f"[lance_server][gpu:{self.device}] Init LLM weights")
            language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)

            vit_model = None
            vit_config = None
            if inference_args.visual_und:
                if model_args.vit_type not in ("qwen2_5_vl", "qwen_2_5_vl_original"):
                    raise ValueError(f"vit_type non supportato: {model_args.vit_type}")
                print(f"[lance_server][gpu:{self.device}] Carico VIT da {model_args.vit_path}")
                vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
                vit_weights = load_file(str(Path(model_args.vit_path) / "vit.safetensors"))
                vit_model.load_state_dict(vit_weights, strict=True)
                clean_memory(vit_weights)

            vae_model = None
            vae_config = None
            if inference_args.visual_gen:
                print(f"[lance_server][gpu:{self.device}] Init VAE")
                vae_model = WanVideoVAE()
                vae_config = deepcopy(vae_model.vae_config)

            config = LanceConfig(
                visual_gen=inference_args.visual_gen,
                visual_und=inference_args.visual_und,
                llm_config=llm_config,
                vit_config=vit_config if inference_args.visual_und else None,
                vae_config=vae_config if inference_args.visual_gen else None,
                latent_patch_size=model_args.latent_patch_size,
                max_num_frames=model_args.max_num_frames,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=inference_args.timestep_shift,
            )
            model: Lance = Lance(
                language_model=language_model,
                vit_model=vit_model if inference_args.visual_und else None,
                vit_type=model_args.vit_type,
                config=config,
                training_args=inference_args,
            )
            print(f"[lance_server][gpu:{self.device}] Sposto Lance su GPU {self.device}")
            model = model.to(self.device)

            print(f"[lance_server][gpu:{self.device}] Carico tokenizer")
            tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

            if inference_args.copy_init_moe:
                language_model.init_moe()

            init_from_model_path_if_needed(model, model_args)

            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)
                model.language_model.config.vocab_size = len(tokenizer)

            if model_args.vit_type.lower() == "qwen2_5_vl":
                from common.model.hacks import hack_qwen2_5_vl_config

                language_model = hack_qwen2_5_vl_config(language_model)

            image_token_id = language_model.config.video_token_id
            new_token_ids.update({"image_token_id": image_token_id})
            model.update_tokenizer(tokenizer=tokenizer)

            if model_args.tie_word_embeddings:
                model.language_model.untie_lm_head()
                model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
                model_args.tie_word_embeddings = False
                llm_config.tie_word_embeddings = False
            else:
                assert (
                    model.language_model.get_input_embeddings().weight.data.data_ptr()
                    != model.language_model.get_output_embeddings().weight.data.data_ptr()
                ), "tie_word_embeddings conflict"

            model = model.to(device=self.device, dtype=torch.bfloat16)
            model.eval()
            if vae_model is not None and hasattr(vae_model, "eval"):
                vae_model.eval()

            self.model = model
            self.vae_model = vae_model
            self.vae_config = vae_config
            self.tokenizer = tokenizer
            self.new_token_ids = new_token_ids
            self.image_token_id = image_token_id

            elapsed = time.perf_counter() - t0
            print(
                f"[lance_server][gpu:{self.device}] Modello pronto in {elapsed:.1f}s",
                flush=True,
            )
            self.initialized = True

    # ── Batch builder ─────────────────────────────────────────────────────

    def _build_request_batch(
        self,
        prompt_file: Path,
        model_args: ModelArguments,
        data_args: DataArguments,
        inference_args: InferenceArguments,
    ):
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.vae_config is not None

        dataset_config = DataConfig.from_yaml(str(prompt_file))
        if inference_args.visual_und:
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if inference_args.visual_gen:
            vae_downsample = tuple_mul(
                tuple(model_args.latent_patch_size),
                (
                    self.vae_config.downsample_temporal,
                    self.vae_config.downsample_spatial,
                    self.vae_config.downsample_spatial,
                ),
            )
            dataset_config.latent_patch_size = model_args.latent_patch_size
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.max_num_frames = model_args.max_num_frames

        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        dataset_config.num_frames = inference_args.num_frames
        dataset_config.H = inference_args.video_height
        dataset_config.W = inference_args.video_width
        dataset_config.task = inference_args.task
        dataset_config.resolution = inference_args.resolution
        dataset_config.text_template = inference_args.text_template

        val_dataset = ValidationDataset(
            jsonl_path=str(prompt_file),
            tokenizer=self.tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=inference_args,
            new_token_ids=self.new_token_ids,
            dataset_config=dataset_config,
            local_rank=0,
            world_size=1,
        )
        return simple_custom_collate([val_dataset[0]])

    # ── Main generate method ─────────────────────────────────────────────

    def generate(
        self,
        task: str,
        prompt: str,
        media_path: Optional[Path],
        question: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
        use_kvcache: bool,
    ) -> Dict[str, Any]:
        """
        Esegue l'inferenza e restituisce un dizionario con:
          - "text": stringa (per task di understanding)
          - "images": lista di data URI base64 (per task immagine)
          - "videos": lista di data URI base64 (per task video)
        """
        self.initialize()

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.image_token_id is not None
        assert self.base_model_args is not None
        assert self.base_inference_args is not None

        with self._generate_lock:
            torch.cuda.set_device(self.device)

            # Crea directory di output temporanea
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = RESULTS_ROOT / f"{task}_{ts}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Crea input dir per questo request
            req_input_dir = TMP_INPUT_DIR / ts
            req_input_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Costruisci il file di prompt
                prompt_file = build_prompt_file(
                    task=task,
                    prompt=prompt,
                    media_path=media_path,
                    question=question,
                    save_dir=req_input_dir,
                )

                # Costruisci model/data/inference args per questa richiesta
                request_model_args = deepcopy(self.base_model_args)
                request_model_args.cfg_text_scale = cfg_text_scale

                request_data_args = DataArguments()
                request_data_args.val_dataset_config_file = str(prompt_file)

                td = TASK_DEFAULTS[task]
                request_inference_args = deepcopy(self.base_inference_args)
                request_inference_args.validation_num_timesteps = validation_num_timesteps
                request_inference_args.validation_timestep_shift = validation_timestep_shift
                request_inference_args.validation_data_seed = seed
                request_inference_args.validation_noise_seed = seed
                request_inference_args.video_height = height or td["video_height"]
                request_inference_args.video_width = width or td["video_width"]
                request_inference_args.num_frames = num_frames or td["num_frames"]
                request_inference_args.resolution = resolution or td["resolution"]
                request_inference_args.save_path_gen = str(save_dir)
                request_inference_args.task = task
                request_inference_args.text_template = TEXT_TEMPLATE
                request_inference_args.use_KVcache = use_kvcache
                request_inference_args.prompt_data_dict = {}

                print(
                    f"[lance_server] Avvio inferenza | task={task} | gpu={self.device} | "
                    f"seed={seed} | {height}x{width} | frames={num_frames} | resolution={resolution}",
                    flush=True,
                )
                t_start = time.perf_counter()

                val_data_cpu = self._build_request_batch(
                    prompt_file=prompt_file,
                    model_args=request_model_args,
                    data_args=request_data_args,
                    inference_args=request_inference_args,
                )
                validate_on_fixed_batch(
                    fsdp_model=self.model,
                    vae_model=self.vae_model,
                    tokenizer=self.tokenizer,
                    val_data_cpu=val_data_cpu,
                    training_args=request_inference_args,
                    model_args=request_model_args,
                    inference_args=request_inference_args,
                    new_token_ids=self.new_token_ids,
                    image_token_id=self.image_token_id,
                    device=self.device,
                    save_source_video=False,
                    save_path_gen=str(save_dir),
                    save_path_gt="",
                )
                save_prompt_results(
                    request_inference_args.prompt_data_dict, str(save_dir), self.logger
                )
                clean_memory()

                elapsed = time.perf_counter() - t_start
                print(f"[lance_server] Inferenza completata in {elapsed:.2f}s", flush=True)

                # Leggi output
                result: Dict[str, Any] = {}

                if task in {TASK_X2T_IMAGE, TASK_X2T_VIDEO}:
                    # Testo
                    pj = save_dir / PROMPT_JSON_FILENAME
                    text_out = ""
                    if pj.exists():
                        data = json.loads(pj.read_text(encoding="utf-8"))
                        if data:
                            text_out = next(iter(data.values()), "")
                    result["text"] = text_out

                elif task in {TASK_T2I, TASK_IMAGE_EDIT}:
                    # Immagini
                    images = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
                    result["images"] = [encode_file_as_data_url(img) for img in images]

                elif task in {TASK_T2V, TASK_VIDEO_EDIT}:
                    # Video
                    videos = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                    result["videos"] = [encode_file_as_data_url(vid) for vid in videos]

                return result

            finally:
                # Pulizia file temporanei
                shutil.rmtree(req_input_dir, ignore_errors=True)
                # Nota: save_dir NON viene rimossa perché potrebbe essere utile per debug.
                # Aggiungere shutil.rmtree(save_dir) per pulizia automatica.


# ─────────────────────────────────────────────────────────────────────────────
# Applicazione FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lance OpenAI-Compatible API",
    description="API REST compatibile OpenAI per il modello Lance (t2i, t2v, i2i, v2v, i2t, v2t)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Istanze globali delle pipeline (inizializzate al primo request o all'avvio)
_image_pipeline: Optional[LancePipeline] = None
_video_pipeline: Optional[LancePipeline] = None


def get_pipeline_for_task(task: str) -> LancePipeline:
    """Restituisce la pipeline appropriata per il task."""
    global _image_pipeline, _video_pipeline

    if task in IMAGE_TASKS:
        if _image_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Nessuna image pipeline disponibile per il task '{task}'. "
                    "Avvia il server con --model-path-image."
                ),
            )
        return _image_pipeline

    if task in VIDEO_TASKS:
        if _video_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Nessuna video pipeline disponibile per il task '{task}'. "
                    "Avvia il server con --model-path-video."
                ),
            )
        return _video_pipeline

    raise HTTPException(status_code=400, detail=f"Task sconosciuto: {task}")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "image_pipeline": _image_pipeline.initialized if _image_pipeline else False,
        "video_pipeline": _video_pipeline.initialized if _video_pipeline else False,
    }


@app.get("/v1/models")
async def list_models():
    """Elenca i modelli disponibili (OpenAI-compatibile)."""
    available = []
    if _image_pipeline is not None:
        for name in ["lance-t2i", "lance-i2i", "lance-i2t"]:
            available.append({"id": name, "object": "model", "owned_by": "bytedance"})
    if _video_pipeline is not None:
        for name in ["lance-t2v", "lance-v2v", "lance-v2t"]:
            available.append({"id": name, "object": "model", "owned_by": "bytedance"})
    return {"object": "list", "data": available}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Endpoint principale compatibile OpenAI Chat Completions.
    Supporta t2i, t2v, i2i (image_edit), v2v (video_edit), i2t, v2t.
    """
    # ── Normalizza messaggi ────────────────────────────────────────────────
    messages = req.messages or req.input or []

    # Estrai tutte le parti di contenuto dall'ultimo messaggio utente
    content_parts: List[ContentPart] = []
    for msg in reversed(messages):
        if msg.role in ("user", "human") or len(messages) == 1:
            if isinstance(msg.content, str):
                content_parts = [ContentPart(type="text", text=msg.content)]
            else:
                content_parts = msg.content or []
            break

    # ── Estrai testo, immagini, video ──────────────────────────────────────
    texts: List[str] = []
    image_urls: List[str] = []
    video_urls: List[str] = []

    for part in content_parts:
        if part.type == "text" and part.text:
            texts.append(part.text)
        elif part.type == "image_url" and part.image_url:
            image_urls.append(part.image_url.url)
        elif part.type == "video_url" and part.video_url:
            video_urls.append(part.video_url.url)

    prompt = " ".join(texts).strip()

    # ── Rileva il task ─────────────────────────────────────────────────────
    task = detect_task(
        model_name=req.model,
        has_image=bool(image_urls),
        has_video=bool(video_urls),
    )

    # ── Salva media in file temporanei ─────────────────────────────────────
    req_id = uuid.uuid4().hex
    tmp_media_dir = TMP_INPUT_DIR / f"media_{req_id}"
    tmp_media_dir.mkdir(parents=True, exist_ok=True)
    media_path: Optional[Path] = None

    try:
        if task in {TASK_IMAGE_EDIT, TASK_X2T_IMAGE} and image_urls:
            media_path = resolve_media(image_urls[0], "image", tmp_media_dir)
        elif task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO} and video_urls:
            media_path = resolve_media(video_urls[0], "video", tmp_media_dir)
    except Exception as exc:
        shutil.rmtree(tmp_media_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Errore nella risoluzione del media: {exc}") from exc

    # ── Parametri di generazione ───────────────────────────────────────────
    td = TASK_DEFAULTS[task]
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    num_frames = req.num_frames if req.num_frames is not None else td["num_frames"]
    height = req.video_height if req.video_height is not None else td["video_height"]
    width = req.video_width if req.video_width is not None else td["video_width"]
    resolution = req.resolution if req.resolution is not None else td["resolution"]
    num_timesteps = req.num_timesteps if req.num_timesteps is not None else DEFAULT_TIMESTEPS
    timestep_shift = req.timestep_shift if req.timestep_shift is not None else DEFAULT_TIMESTEP_SHIFT
    cfg_scale = req.cfg_scale if req.cfg_scale is not None else DEFAULT_CFG_TEXT_SCALE
    use_kvcache = req.use_kvcache if req.use_kvcache is not None else USE_KVCACHE

    # Per i task di understanding, "question" = tutto il testo
    # Per i task di generazione/editing, "prompt" = tutto il testo
    question = prompt  # per x2t_image / x2t_video

    # ── Seleziona pipeline ─────────────────────────────────────────────────
    pipeline = get_pipeline_for_task(task)

    # ── Inferenza (in thread separato per non bloccare l'event loop) ───────
    import asyncio

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.generate(
                task=task,
                prompt=prompt,
                media_path=media_path,
                question=question,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                resolution=resolution,
                validation_num_timesteps=num_timesteps,
                validation_timestep_shift=timestep_shift,
                cfg_text_scale=cfg_scale,
                use_kvcache=use_kvcache,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore durante l'inferenza: {exc}") from exc
    finally:
        shutil.rmtree(tmp_media_dir, ignore_errors=True)

    # ── Costruisci risposta OpenAI-compatibile ─────────────────────────────
    message_content: Dict[str, Any] = {"role": "assistant"}

    if "text" in result:
        message_content["content"] = result["text"]
        message_content["images"] = None
        message_content["videos"] = None
    elif "images" in result:
        message_content["content"] = None
        message_content["images"] = [
            {"imageUrl": {"url": url}} for url in result.get("images", [])
        ]
        message_content["videos"] = None
    elif "videos" in result:
        message_content["content"] = None
        message_content["images"] = None
        message_content["videos"] = [
            {"videoUrl": {"url": url}} for url in result.get("videos", [])
        ]
    else:
        message_content["content"] = None
        message_content["images"] = None
        message_content["videos"] = None

    return {
        "id": f"gen-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message_content,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance OpenAI-compatible API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Path modelli (opzionali: di default usa downloads/ con auto-download) ──
    parser.add_argument(
        "--model-path-image",
        type=str,
        default=str(DEFAULT_MODEL_PATH_IMAGE),
        help="Percorso al checkpoint Lance per task immagine (t2i, i2i, i2t). "
             "Se assente viene scaricato automaticamente da HuggingFace.",
    )
    parser.add_argument(
        "--model-path-video",
        type=str,
        default=str(DEFAULT_MODEL_PATH_VIDEO),
        help="Percorso al checkpoint Lance per task video (t2v, v2v, v2t). "
             "Se assente viene scaricato automaticamente da HuggingFace.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=str,
        default=str(DEFAULT_DOWNLOADS_DIR),
        help="Directory dove vengono salvati i pesi scaricati da HuggingFace.",
    )
    # ── GPU ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--gpu-image",
        type=int,
        default=0,
        help="ID GPU per la pipeline immagine.",
    )
    parser.add_argument(
        "--gpu-video",
        type=int,
        default=0,
        help="ID GPU per la pipeline video.",
    )
    # ── Server ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Porta del server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host del server.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Carica i modelli subito all'avvio invece che al primo request.",
    )
    # ── Controllo pipeline ────────────────────────────────────────────────────
    parser.add_argument(
        "--disable-image-pipeline",
        action="store_true",
        help="Non caricare la pipeline immagine (t2i, i2i, i2t).",
    )
    parser.add_argument(
        "--disable-video-pipeline",
        action="store_true",
        help="Non caricare la pipeline video (t2v, v2v, v2t).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Non scaricare automaticamente i modelli mancanti da HuggingFace.",
    )
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = _parse_args()

    global _image_pipeline, _video_pipeline

    image_path = Path(args.model_path_image)
    video_path = Path(args.model_path_video)
    downloads_dir = Path(args.downloads_dir)
    need_image = not args.disable_image_pipeline
    need_video = not args.disable_video_pipeline

    # ── Download automatico se i modelli non sono presenti ─────────────────
    if not args.no_download:
        try:
            ensure_models_downloaded(
                image_path=image_path,
                video_path=video_path,
                downloads_dir=downloads_dir,
                need_image=need_image,
                need_video=need_video,
            )
        except Exception as exc:
            print(f"[lance_server] ERRORE durante il download: {exc}")
            return
    else:
        print("[lance_server] --no-download attivo: download automatico disabilitato.")

    # ── Crea le pipeline ───────────────────────────────────────────────────
    if need_image:
        if _is_lance_model_ready(image_path):
            _image_pipeline = LancePipeline(
                model_path=str(image_path),
                device_id=args.gpu_image,
                default_task=TASK_T2I,
            )
            print(f"[lance_server] Image pipeline: {image_path} @ GPU {args.gpu_image}")
        else:
            print(
                f"[lance_server] AVVISO: i file del modello immagine non sono stati trovati "
                f"in '{image_path}'. Pipeline immagine non disponibile."
            )

    if need_video:
        if _is_lance_model_ready(video_path):
            _video_pipeline = LancePipeline(
                model_path=str(video_path),
                device_id=args.gpu_video,
                default_task=TASK_T2V,
            )
            print(f"[lance_server] Video pipeline: {video_path} @ GPU {args.gpu_video}")
        else:
            print(
                f"[lance_server] AVVISO: i file del modello video non sono stati trovati "
                f"in '{video_path}'. Pipeline video non disponibile."
            )

    if _image_pipeline is None and _video_pipeline is None:
        print(
            "[lance_server] ERRORE: Nessun modello disponibile. "
            "Rimuovi --no-download oppure verifica i path."
        )
        return

    # ── Pre-caricamento opzionale ──────────────────────────────────────────
    if args.preload:
        if _image_pipeline is not None:
            print("[lance_server] Pre-carico image pipeline...")
            _image_pipeline.initialize()
        if _video_pipeline is not None:
            print("[lance_server] Pre-carico video pipeline...")
            _video_pipeline.initialize()

    print(f"\n[lance_server] Server in ascolto su http://{args.host}:{args.port}")
    print("[lance_server] Endpoint: POST /v1/chat/completions")
    print("[lance_server] Docs:     http://localhost:{}/docs\n".format(args.port))

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        # Aumenta il limite per le richieste con base64 di immagini/video grandi
        h11_max_incomplete_event_size=256 * 1024 * 1024,  # 256 MB
    )


if __name__ == "__main__":
    main()
