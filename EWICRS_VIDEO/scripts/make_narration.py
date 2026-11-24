"""Synthesize the narration offline (Kokoro ONNX), fit every line to its cue and
write assets/audio/narration_full.wav plus narration_report.json and narration.txt.

Usage: python scripts/make_narration.py [--voice bm_george]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from narration import SCENES, DUR, OFFSET, TOTAL, NARRATION, READABLE

ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="bm_george")
args = ap.parse_args()

from kokoro_onnx import Kokoro
K = Kokoro(str(ROOT / "assets/audio/kokoro-v1.0.onnx"), str(ROOT / "assets/audio/voices-v1.0.bin"))
LANG = "en-gb" if args.voice.startswith("b") else "en-us"
SR = 24000
OUT = ROOT / "assets/audio/narration"
OUT.mkdir(parents=True, exist_ok=True)


def synth(text, speed):
    wav, sr = K.create(text, voice=args.voice, speed=speed, lang=LANG)
    assert sr == SR
    # trim leading / trailing silence so the cue is the first audible syllable
    a = np.abs(wav)
    idx = np.where(a > 0.012)[0]
    if len(idx) == 0:
        return wav
    s = max(0, idx[0] - int(0.03 * SR))
    e = min(len(wav), idx[-1] + int(0.12 * SR))
    return wav[s:e]


by_scene = {}
for sc, cue, text in NARRATION:
    by_scene.setdefault(sc, []).append((cue, text))

track = np.zeros(int(TOTAL * SR) + SR, dtype=np.float32)
report = []
for sc, lines in by_scene.items():
    lines.sort()
    for i, (cue, text) in enumerate(lines):
        nxt = lines[i + 1][0] if i + 1 < len(lines) else DUR[sc] - 0.35
        avail = nxt - cue - 0.20
        speed = 1.0
        wav = synth(text, speed)
        d = len(wav) / SR
        tries = 0
        while d > avail and speed < 1.12 and tries < 4:
            speed = min(1.12, speed * (d / avail) * 1.01)
            wav = synth(text, speed)
            d = len(wav) / SR
            tries += 1
        fit = d <= avail + 0.05
        t0 = OFFSET[sc] + cue
        a0 = int(t0 * SR)
        track[a0:a0 + len(wav)] += wav.astype(np.float32)
        sf.write(OUT / f"{sc}_{i:02d}.wav", wav, SR)
        report.append(dict(scene=sc, cue=cue, start=round(t0, 2), end=round(t0 + d, 2), dur=round(d, 2),
                           avail=round(avail, 2), speed=round(speed, 3), fits=bool(fit), text=text))
        print(f"{sc} {cue:5.1f}s  dur {d:5.2f}/{avail:5.2f}  speed {speed:.2f}  {'OK ' if fit else 'LONG'}  {text[:60]}", flush=True)

peak = np.max(np.abs(track))
track = track / peak * 0.89
sf.write(ROOT / "assets/audio/narration_full.wav", track, SR)
(ROOT / "assets/audio/narration_report.json").write_text(json.dumps(report, indent=1))

# human-readable script with global timecodes
lines_out = ["EWICRS - CINEMATIC FILM: NARRATION SCRIPT", "(timecodes are global; cue = first spoken word)", ""]
for sc, title, dur in SCENES:
    lines_out.append(f"=== {sc.upper()}  {title}   [{int(OFFSET[sc]//60)}:{OFFSET[sc]%60:04.1f} - {int((OFFSET[sc]+dur)//60)}:{(OFFSET[sc]+dur)%60:04.1f}]")
    for r in report:
        if r["scene"] == sc:
            txt = r["text"]
            for k, v in READABLE.items():
                txt = txt.replace(k, v)
            g = r["start"]
            lines_out.append(f"  [{int(g//60)}:{g%60:05.2f}]  {txt}")
    lines_out.append("")
(HERE / "narration.txt").write_text("\n".join(lines_out))
print("narration done; lines that do not fit:", [r["text"][:40] for r in report if not r["fits"]])
