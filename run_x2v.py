#!/usr/bin/env python3
"""
run_x2v.py
----------
Invia una richiesta x2v (Any → Video) al server Lance OpenAI-compatibile.

Uso:
    python run_x2v.py -p "prompt" --image img1.png [--image img2.png ...] \
                      [--video ref.mp4] --duration 3 [opzioni]

Esempi:
    python run_x2v.py -p "La ragazza indossa il cappello" \
        --image test_input/girl.png --image test_input/hat.jpeg \
        --duration 3 --host 192.168.1.20 --port 8088

    python run_x2v.py -p "Drive on a snowy road" \
        --video config/examples/video_edit_examples/edit_source_car.mp4 \
        --duration 4 --timesteps 30
"""

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Dipendenza mancante: installa 'requests'  →  pip install requests")

# fps usato dal server quando salva l'MP4 (decode_video_tensor → imageio fps=12)
SERVER_OUTPUT_FPS = 12
MIN_FRAMES = 11   # minimo fisico del WAN VAE (T_lat > 1 richiede T ≥ 11)
MAX_FRAMES = 121  # massimo supportato dal modello


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_to_data_uri(path: Path, mime: str) -> str:
    raw = path.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _mime_for_image(path: Path) -> str:
    ext = path.suffix.lower()
    return {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/png")


def _duration_to_frames(duration: float) -> int:
    """Converte secondi in num_frames (arrotondato, clamped in [MIN, MAX])."""
    frames = round(duration * SERVER_OUTPUT_FPS)
    if frames < MIN_FRAMES:
        print(
            f"ATTENZIONE: {duration}s → {frames} frame < minimo {MIN_FRAMES}. "
            f"Forzato a {MIN_FRAMES}.",
            file=sys.stderr,
        )
        frames = MIN_FRAMES
    if frames > MAX_FRAMES:
        print(
            f"ATTENZIONE: {duration}s → {frames} frame > massimo {MAX_FRAMES}. "
            f"Clampato a {MAX_FRAMES}.",
            file=sys.stderr,
        )
        frames = MAX_FRAMES
    return frames


def _build_content(prompt: str, images: list[Path], video: Path | None) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        mime = _mime_for_image(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": _file_to_data_uri(img, mime)},
        })
    if video is not None:
        content.append({
            "type": "video_url",
            "video_url": {"url": _file_to_data_uri(video, "video/mp4")},
        })
    return content


def _save_video(data_uri: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, b64 = data_uri.split(",", 1)
    raw = base64.b64decode(b64)
    out_path = out_dir / f"{stem}.mp4"
    out_path.write_bytes(raw)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un video con Lance x2v (Any → Video)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-p", "--prompt", required=True,
                        help="Prompt testuale")
    parser.add_argument("--image", dest="images", metavar="PATH",
                        action="append", default=[],
                        help="Immagine di riferimento (ripetibile)")
    parser.add_argument("--video", metavar="PATH",
                        help="Video di riferimento (opzionale)")
    parser.add_argument("--duration", type=float, required=True,
                        help=f"Durata output in secondi (fps server = {SERVER_OUTPUT_FPS})")
    parser.add_argument("--model", metavar="MODEL", default=None,
                        help=(
                            "Modello da usare (default: auto). "
                            "Senza --video usa 'lance-ti2v' (modalità idip: preserva identità dall'immagine). "
                            "Con --video usa 'lance-x2v' (modalità edit: editing video con riferimento immagine)."
                        ))
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host del server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8088,
                        help="Porta del server (default: 8088)")
    parser.add_argument("--timesteps", type=int, default=30,
                        help="Passi di denoising (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed per la generazione (default: casuale)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout richiesta HTTP in secondi (default: 600)")
    parser.add_argument("--cfg-vit-scale", type=float, default=None,
                        help=(
                            "Scala CFG per i token VIT (immagine). "
                            "Default: 2.0 per lance-ti2v (amplifica preservazione identità), "
                            "1.0 per gli altri modelli. "
                            "Valori > 1 aumentano l'influenza dell'immagine di riferimento."
                        ))
    parser.add_argument("--output-dir", default="test_outputs",
                        help="Cartella dove salvare il video (default: test_outputs)")
    parser.add_argument("--output-name", default=None,
                        help="Nome file output senza estensione (default: x2v_<timestamp>)")
    args = parser.parse_args()

    # ── Validazione input ──────────────────────────────────────────────────
    image_paths: list[Path] = []
    for p in args.images:
        path = Path(p)
        if not path.exists():
            sys.exit(f"Immagine non trovata: {path}")
        image_paths.append(path)

    video_path: Path | None = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            sys.exit(f"Video non trovato: {video_path}")

    if not image_paths and video_path is None:
        sys.exit("Specifica almeno un --image o un --video come riferimento.")

    num_frames = _duration_to_frames(args.duration)

    # ── Selezione automatica del modello ──────────────────────────────────
    # - Solo immagini (no video) → lance-ti2v  (task=tiv2v_idip, sample_task='idip')
    #   Il modello usa l'immagine come riferimento IDENTITÀ (ref_image modality).
    # - Con video (± immagini)  → lance-x2v   (task=x2v, sample_task='edit')
    #   Il modello usa il video come sorgente da editare (ref_source modality).
    if args.model:
        model_name = args.model
    elif video_path is None:
        model_name = "lance-ti2v"
    else:
        model_name = "lance-x2v"

    # ── Costruzione payload ────────────────────────────────────────────────
    payload: dict = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": _build_content(args.prompt, image_paths, video_path),
        }],
        "num_timesteps": args.timesteps,
        "num_frames": num_frames,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.cfg_vit_scale is not None:
        payload["cfg_vit_scale"] = args.cfg_vit_scale

    base_url = f"http://{args.host}:{args.port}"
    stem = args.output_name or f"x2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Riepilogo ──────────────────────────────────────────────────────────
    print(f"Server  : {base_url}")
    print(f"Modello : {model_name}  ({'idip – preserva identità immagine' if model_name == 'lance-ti2v' else 'edit – editing video con riferimento'})")
    print(f"Prompt  : {args.prompt!r}")
    print(f"Immagini: {[str(p) for p in image_paths] or '(nessuna)'}")
    print(f"Video   : {str(video_path) if video_path else '(nessuno)'}")
    print(f"Frames  : {num_frames}  ({args.duration}s × {SERVER_OUTPUT_FPS}fps)")
    print(f"Steps   : {args.timesteps}  |  seed: {args.seed if args.seed is not None else 'casuale'}")
    if args.cfg_vit_scale is not None:
        print(f"VIT CFG : {args.cfg_vit_scale}")
    elif model_name == "lance-ti2v":
        print(f"VIT CFG : 2.0 (default ti2v, preservazione identità)")
    print()

    # ── Richiesta ──────────────────────────────────────────────────────────
    try:
        print("Invio richiesta…", end=" ", flush=True)
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=args.timeout,
        )
    except requests.exceptions.Timeout:
        sys.exit(f"\nTimeout dopo {args.timeout}s. Aumenta --timeout.")
    except requests.exceptions.ConnectionError as e:
        sys.exit(f"\nImpossibile connettersi a {base_url}: {e}")

    # ── Risposta ───────────────────────────────────────────────────────────
    if resp.status_code != 200:
        sys.exit(f"\nHTTP {resp.status_code}: {resp.text[:500]}")

    try:
        data = resp.json()
    except Exception:
        sys.exit("\nRisposta non è JSON valido.")

    debug_info = data.get("_debug")
    if debug_info:
        print(f"\n[SERVER DEBUG] {json.dumps(debug_info, ensure_ascii=False)}")

    choices = data.get("choices", [])
    if not choices:
        sys.exit(f"\nNessun elemento in 'choices'. Risposta: {json.dumps(data)[:300]}")

    videos = choices[0].get("message", {}).get("videos", [])
    if not videos:
        sys.exit(f"\nNessun video nella risposta. Risposta: {json.dumps(data)[:300]}")

    out_dir = Path(args.output_dir)
    saved: list[Path] = []
    for i, item in enumerate(videos):
        url = item.get("videoUrl", {}).get("url", "")
        if not url.startswith("data:"):
            sys.exit(f"\nURL video non è un data URI valido: {url[:80]}")
        file_stem = stem if len(videos) == 1 else f"{stem}_{i}"
        saved.append(_save_video(url, out_dir, file_stem))

    print("OK")
    for p in saved:
        print(f"Salvato : {p.resolve()}")


if __name__ == "__main__":
    main()
