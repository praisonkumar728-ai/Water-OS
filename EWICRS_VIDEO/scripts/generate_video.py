#!/usr/bin/env python3
"""EWICRS film: one-command pipeline.
    python3 scripts/generate_video.py [--skip-narration] [--rerender] [--jobs 2]
Stages: narration (Kokoro TTS) -> score + sfx -> mix -> render 8 scenes -> concat -> mux audio -> verify.
Prerequisites: ffmpeg/ffprobe, numpy, scipy, opencv-python, pillow, soundfile, kokoro-onnx and the Kokoro model files
(assets/audio/kokoro-v1.0.onnx, voices-v1.0.bin; see README.md)."""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC = ROOT / "assets/generated_scenes"
OUT = ROOT / "output/EWICRS_CINEMATIC_FILM.mp4"
NAMES = ["scene_01_problem", "scene_02_architecture", "scene_03_digital_twin", "scene_04_model_revision",
         "scene_05_pollution", "scene_06_climate", "scene_07_hardware", "scene_08_conclusion"]


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames:format=duration",
                        "-of", "json", str(p)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-narration", action="store_true")
    ap.add_argument("--rerender", action="store_true", help="delete existing scene MP4s first")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.rerender:
        for n in NAMES:
            (SC / f"{n}.mp4").unlink(missing_ok=True)
    if not a.skip_narration:
        run([sys.executable, ROOT / "scripts/make_narration.py"])
    run([sys.executable, ROOT / "scripts/make_music.py"])
    run([sys.executable, ROOT / "scripts/mix_audio.py"])
    run(["bash", ROOT / "scripts/render_scenes.sh", a.jobs])
    missing = [n for n in NAMES if not (SC / f"{n}.mp4").exists()]
    if missing:
        sys.exit(f"missing scene files: {missing}")
    lst = ROOT / "work/concat.txt"
    lst.write_text("".join(f"file '{(SC / n).with_suffix('.mp4')}'\n" for n in NAMES))
    silent = ROOT / "work/film_silent.mp4"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])
    tmp = OUT.with_name(OUT.stem + ".tmp.mp4")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", silent, "-i", ROOT / "assets/audio/final_mix.wav",
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", tmp])
    tmp.replace(OUT)
    info = probe(OUT)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    d = float(info["format"]["duration"])
    print(f"\nFINAL: {OUT}\n  {v['width']}x{v['height']} @ {v['r_frame_rate']}  {v['codec_name']}  duration {d:.2f}s ({int(d // 60)}:{d % 60:05.2f})")
    assert v["width"] >= 1920 and v["height"] >= 1080 and d <= 290.0, "specification check failed"
    print("  spec check passed (>=1920x1080, <=4:50)")


if __name__ == "__main__":
    main()
