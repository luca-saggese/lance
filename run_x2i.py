#!/usr/bin/env python3
"""
run_x2i.py
----------
Invia una richiesta x2i / i2i (Any → Image) al server Lance OpenAI-compatibile.

Uso:
    python run_x2i.py -p "prompt" --image img1.png [--image img2.png ...] \
                      [--video ref.mp4] [opzioni]

Esempi:
    # Image editing semplice (lance-i2i)
    python run_x2i.py -p "Turn the sky into a dramatic stormy sky" \
        --image test_input/girl.png

    # Compositional image generation (lance-x2i) con più input
    python run_x2i.py -p "La ragazza indossa il cappello" \
        --image test_input/girl.png --image test_input/hat.jpeg

    # Any → Image con riferimento video (lance-x2i)
    python run_x2i.py -p "Extract a stylized frame from this scene" \
        --video config/examples/video_edit_examples/edit_source_woman.mp4 \
        --timesteps 30
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_to_data_uri(path: Path, mime: str) -> str:
    raw = path.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _mime_for_image(path: Path) -> str:
    ext = path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/png")


def _ext_for_mime(mime: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime, ".png")


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


def _save_image(data_uri: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    header, b64 = data_uri.split(",", 1)
    # Estrai il mime type dall'header (es. "data:image/png;base64")
    mime = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
    ext = _ext_for_mime(mime)
    raw = base64.b64decode(b64)
    out_path = out_dir / f"{stem}{ext}"
    out_path.write_bytes(raw)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un'immagine con Lance x2i / i2i (Any → Image)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-p", "--prompt", required=True,
                        help="Prompt testuale che descrive la modifica o la generazione")
    parser.add_argument("--image", dest="images", metavar="PATH",
                        action="append", default=[],
                        help="Immagine di input (ripetibile)")
    parser.add_argument("--video", metavar="PATH",
                        help="Video di riferimento (opzionale)")
    parser.add_argument("--model", metavar="MODEL", default=None,
                        help=(
                            "Modello da usare (default: auto). "
                            "Con una sola --image usa 'lance-i2i' (editing diretto). "
                            "Con più immagini o --video usa 'lance-x2i' (any→image)."
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
    parser.add_argument("--cfg-vit-scale", "--cfg_vit_scale", type=float, default=None,
                        help=(
                            "Scala CFG per i token VIT (immagine). "
                            "Valori > 1 aumentano l'influenza dell'immagine di riferimento."
                        ))
    parser.add_argument("-W", "--width", type=int, default=None,
                        help="Larghezza immagine output in pixel (default: 768)")
    parser.add_argument("-H", "--height", type=int, default=None,
                        help="Altezza immagine output in pixel (default: 768)")
    parser.add_argument("--mask", metavar="PATH", default=None,
                        help=(
                            "Maschera di inpainting (immagine in scala di grigi). "
                            "Bianco (255) = zona da modificare con il prompt. "
                            "Nero (0) = zona da preservare intatta."
                        ))
    parser.add_argument("--output-dir", default="test_outputs",
                        help="Cartella dove salvare l'immagine (default: test_outputs)")
    parser.add_argument("--output-name", default=None,
                        help="Nome file output senza estensione (default: x2i_<timestamp>)")
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
        sys.exit("Specifica almeno un --image o un --video come input.")

    # Maschera di inpainting (opzionale)
    mask_path: Path | None = None
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            sys.exit(f"Maschera non trovata: {mask_path}")

    # ── Selezione automatica del modello ──────────────────────────────────
    # - Una sola immagine (no video) → lance-i2i  (editing diretto)
    # - Più immagini o con video    → lance-x2i  (any→image compositional)
    if args.model:
        model_name = args.model
    elif len(image_paths) == 1 and video_path is None:
        model_name = "lance-i2i"
    else:
        model_name = "lance-x2i"

    # ── Costruzione payload ────────────────────────────────────────────────
    payload: dict = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": _build_content(args.prompt, image_paths, video_path),
        }],
        "num_timesteps": args.timesteps,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.cfg_vit_scale is not None:
        payload["cfg_vit_scale"] = args.cfg_vit_scale
    if args.width is not None:
        payload["video_width"] = args.width
    if args.height is not None:
        payload["video_height"] = args.height
    if mask_path is not None:
        payload["mask_url"] = _file_to_data_uri(mask_path, _mime_for_image(mask_path))

    base_url = f"http://{args.host}:{args.port}"
    stem = args.output_name or f"x2i_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Riepilogo ──────────────────────────────────────────────────────────
    print(f"Server  : {base_url}")
    print(f"Modello : {model_name}  ({'editing diretto' if model_name == 'lance-i2i' else 'any→image compositional'})")
    print(f"Prompt  : {args.prompt!r}")
    print(f"Immagini: {[str(p) for p in image_paths] or '(nessuna)'}")
    print(f"Video   : {str(video_path) if video_path else '(nessuno)'}")
    print(f"Steps   : {args.timesteps}  |  seed: {args.seed if args.seed is not None else 'casuale'}")
    if args.cfg_vit_scale is not None:
        print(f"VIT CFG : {args.cfg_vit_scale}")
    if args.width is not None or args.height is not None:
        print(f"Size    : {args.width or 'default'}x{args.height or 'default'}")
    if mask_path is not None:
        print(f"Maschera: {mask_path.resolve()}")
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

    images = choices[0].get("message", {}).get("images", [])
    if not images:
        sys.exit(f"\nNessuna immagine nella risposta. Risposta: {json.dumps(data)[:300]}")

    out_dir = Path(args.output_dir)
    saved: list[Path] = []
    for i, item in enumerate(images):
        url = item.get("imageUrl", {}).get("url", "")
        if not url.startswith("data:"):
            sys.exit(f"\nURL immagine non è un data URI valido: {url[:80]}")
        file_stem = stem if len(images) == 1 else f"{stem}_{i}"
        saved.append(_save_image(url, out_dir, file_stem))

    print("OK")
    for p in saved:
        print(f"Salvato : {p.resolve()}")


if __name__ == "__main__":
    main()
