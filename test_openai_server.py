#!/usr/bin/env python3
"""
test_openai_server.py
---------------------
Testa tutti gli endpoint del server Lance OpenAI-compatibile.

Uso:
    python test_openai_server.py <port>

Esempio:
    python test_openai_server.py 8000
"""

import argparse
import base64
import json
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Dipendenza mancante: installa 'requests'  →  pip install requests")

OUTPUT_DIR = Path("test_outputs")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⚬\033[0m"


def _result(ok: bool, label: str, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    msg = f"{icon}  {label}"
    if detail:
        msg += f"\n     {detail}"
    print(msg)


def _png_1x1(r: int = 100, g: int = 149, b: int = 237) -> bytes:
    """Genera un PNG 1×1 RGB in puro Python (nessuna dipendenza esterna)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_row = b"\x00" + bytes([r, g, b])
    idat = chunk(b"IDAT", zlib.compress(raw_row))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _png_b64() -> str:
    return "data:image/png;base64," + base64.b64encode(_png_1x1()).decode()


def _make_mp4_b64() -> str | None:
    """
    Crea un breve video MP4 (3 frame, 1×1 px) tramite ffmpeg.
    Restituisce il data URI base64, o None se ffmpeg non è disponibile.
    """
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "color=c=blue:size=64x64:rate=1:duration=3",
                "-pix_fmt", "yuv420p",
                "-t", "3",
                tmp_path,
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        raw = Path(tmp_path).read_bytes()
        return "data:video/mp4;base64," + base64.b64encode(raw).decode()
    except Exception:
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _post(base_url: str, payload: dict, timeout: int) -> requests.Response:
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )


def _save_data_uri(data_uri: str, label: str, ext: str) -> Path:
    """Decodifica un data URI base64 e lo salva in OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, b64 = data_uri.split(",", 1)
    raw = base64.b64decode(b64)
    out_path = OUTPUT_DIR / f"{label}{ext}"
    out_path.write_bytes(raw)
    return out_path


def _check_generation_response(
    resp: requests.Response, field: str, label: str
) -> tuple[bool, str]:
    """Valida la struttura di una response di generazione (image/video) e salva i file."""
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except Exception:
        return False, "Risposta non è JSON valido"
    choices = data.get("choices", [])
    if not choices:
        return False, "Nessun elemento in 'choices'"
    msg = choices[0].get("message", {})
    items = msg.get(field, [])
    if not items:
        return False, f"Campo '{field}' assente o vuoto nel messaggio"
    url_field = "imageUrl" if field == "images" else "videoUrl"
    ext = ".png" if field == "images" else ".mp4"
    saved: list[Path] = []
    for i, item in enumerate(items):
        url = item.get(url_field, {}).get("url", "")
        if not url.startswith("data:"):
            return False, f"URL nel campo '{url_field}' non è un data URI valido"
        suffix = f"_{i}" if len(items) > 1 else ""
        saved.append(_save_data_uri(url, f"{label}{suffix}", ext))
    paths = ", ".join(str(p) for p in saved)
    return True, f"Salvat{'a' if len(saved) == 1 else 'e'}: {paths}"


def _check_text_response(resp: requests.Response) -> tuple[bool, str]:
    """Valida la struttura di una response di understanding (text)."""
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except Exception:
        return False, "Risposta non è JSON valido"
    choices = data.get("choices", [])
    if not choices:
        return False, "Nessun elemento in 'choices'"
    content = choices[0].get("message", {}).get("content", None)
    if not content:
        return False, "Campo 'content' assente o vuoto"
    return True, f"Risposta: {str(content)[:120]}"


# ─────────────────────────────────────────────────────────────────────────────
# Singoli test
# ─────────────────────────────────────────────────────────────────────────────

def test_health(base_url: str, timeout: int) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        _result(ok, "GET /health", json.dumps(resp.json()) if ok else resp.text[:200])
        return ok
    except Exception as exc:
        _result(False, "GET /health", str(exc))
        return False


def test_list_models(base_url: str, timeout: int) -> list[str]:
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if resp.status_code != 200:
            _result(False, "GET /v1/models", f"HTTP {resp.status_code}")
            return []
        ids = [m["id"] for m in resp.json().get("data", [])]
        _result(True, "GET /v1/models", "Modelli: " + ", ".join(ids) if ids else "(nessun modello caricato)")
        return ids
    except Exception as exc:
        _result(False, "GET /v1/models", str(exc))
        return []


def test_t2i(base_url: str, timeout: int, seed: int) -> bool:
    payload = {
        "model": "lance-t2i",
        "messages": [{"role": "user", "content": "A serene mountain lake at sunset"}],
        "seed": seed,
        "num_timesteps": 5,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "images", "t2i")
        _result(ok, "POST /v1/chat/completions  [lance-t2i  – Text→Image]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [lance-t2i  – Text→Image]", str(exc))
        return False


def test_t2v(base_url: str, timeout: int, seed: int) -> bool:
    payload = {
        "model": "lance-t2v",
        "messages": [{"role": "user", "content": "A bird flying over the ocean"}],
        "seed": seed,
        "num_timesteps": 5,
        "num_frames": 9,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "t2v")
        _result(ok, "POST /v1/chat/completions  [lance-t2v  – Text→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [lance-t2v  – Text→Video]", str(exc))
        return False


def test_i2i(base_url: str, timeout: int, seed: int) -> bool:
    payload = {
        "model": "lance-i2i",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Make the sky pink"},
                    {"type": "image_url", "image_url": {"url": _png_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "images", "i2i")
        _result(ok, "POST /v1/chat/completions  [lance-i2i  – Image→Image]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [lance-i2i  – Image→Image]", str(exc))
        return False


def test_i2t(base_url: str, timeout: int, seed: int) -> bool:
    payload = {
        "model": "lance-i2t",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color dominates this image?"},
                    {"type": "image_url", "image_url": {"url": _png_b64()}},
                ],
            }
        ],
        "seed": seed,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_text_response(resp)
        _result(ok, "POST /v1/chat/completions  [lance-i2t  – Image→Text]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [lance-i2t  – Image→Text]", str(exc))
        return False


def test_v2v(base_url: str, timeout: int, seed: int, video_b64: str) -> bool:
    payload = {
        "model": "lance-v2v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Add a golden glow effect"},
                    {"type": "video_url", "video_url": {"url": video_b64}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
        "num_frames": 9,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "v2v")
        _result(ok, "POST /v1/chat/completions  [lance-v2v  – Video→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [lance-v2v  – Video→Video]", str(exc))
        return False


def test_v2t(base_url: str, timeout: int, seed: int, video_b64: str) -> bool:
    payload = {
        "model": "lance-v2t",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what happens in this video"},
                    {"type": "video_url", "video_url": {"url": video_b64}},
                ],
            }
        ],
        "seed": seed,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_text_response(resp)
        _result(ok, "POST /v1/chat/completions  [lance-v2t  – Video→Text]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [lance-v2t  – Video→Text]", str(exc))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test del server Lance OpenAI-compatibile")
    parser.add_argument("port", type=int, help="Porta su cui gira il server (es. 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host del server (default: 127.0.0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione (default: 42)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in secondi per ogni richiesta (default: 300)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"\nServer: {base_url}  |  seed={args.seed}  |  timeout={args.timeout}s")
    print(f"Output:  {OUTPUT_DIR.resolve()}\n")
    print("=" * 65)

    results: dict[str, bool | None] = {}

    # ── 1. Health ─────────────────────────────────────────────────────────
    results["health"] = test_health(base_url, timeout=10)

    # ── 2. Models list ────────────────────────────────────────────────────
    available_models = test_list_models(base_url, timeout=10)
    results["list_models"] = True  # se arriva fin qui senza eccezione è ok

    print()

    # ── 3. Text-to-Image ──────────────────────────────────────────────────
    if "lance-t2i" in available_models:
        results["t2i"] = test_t2i(base_url, args.timeout, args.seed)
    else:
        print(f"{SKIP}  POST /v1/chat/completions  [lance-t2i  – Text→Image]  (pipeline non caricata)")
        results["t2i"] = None

    # ── 4. Image→Image ────────────────────────────────────────────────────
    if "lance-i2i" in available_models:
        results["i2i"] = test_i2i(base_url, args.timeout, args.seed)
    else:
        print(f"{SKIP}  POST /v1/chat/completions  [lance-i2i  – Image→Image]  (pipeline non caricata)")
        results["i2i"] = None

    # ── 5. Image→Text ─────────────────────────────────────────────────────
    if "lance-i2t" in available_models:
        results["i2t"] = test_i2t(base_url, args.timeout, args.seed)
    else:
        print(f"{SKIP}  POST /v1/chat/completions  [lance-i2t  – Image→Text]  (pipeline non caricata)")
        results["i2t"] = None

    # ── Prepara video per i test video ────────────────────────────────────
    needs_video = any(m in available_models for m in ("lance-t2v", "lance-v2v", "lance-v2t"))
    video_b64: str | None = None
    if needs_video:
        print("\nCreazione video di test con ffmpeg...", end=" ", flush=True)
        video_b64 = _make_mp4_b64()
        print("ok" if video_b64 else "ffmpeg non trovato, test video saltati")

    print()

    # ── 6. Text-to-Video ──────────────────────────────────────────────────
    if "lance-t2v" in available_models:
        results["t2v"] = test_t2v(base_url, args.timeout, args.seed)
    else:
        print(f"{SKIP}  POST /v1/chat/completions  [lance-t2v  – Text→Video]  (pipeline non caricata)")
        results["t2v"] = None

    # ── 7. Video→Video ────────────────────────────────────────────────────
    if "lance-v2v" in available_models:
        if video_b64:
            results["v2v"] = test_v2v(base_url, args.timeout, args.seed, video_b64)
        else:
            print(f"{SKIP}  POST /v1/chat/completions  [lance-v2v  – Video→Video]  (ffmpeg non disponibile)")
            results["v2v"] = None
    else:
        print(f"{SKIP}  POST /v1/chat/completions  [lance-v2v  – Video→Video]  (pipeline non caricata)")
        results["v2v"] = None

    # ── 8. Video→Text ─────────────────────────────────────────────────────
    if "lance-v2t" in available_models:
        if video_b64:
            results["v2t"] = test_v2t(base_url, args.timeout, args.seed, video_b64)
        else:
            print(f"{SKIP}  POST /v1/chat/completions  [lance-v2t  – Video→Text]  (ffmpeg non disponibile)")
            results["v2t"] = None
    else:
        print(f"{SKIP}  POST /v1/chat/completions  [lance-v2t  – Video→Text]  (pipeline non caricata)")
        results["v2t"] = None

    # ── Riepilogo ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    print(f"Riepilogo: {passed} passati  |  {failed} falliti  |  {skipped} saltati\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
