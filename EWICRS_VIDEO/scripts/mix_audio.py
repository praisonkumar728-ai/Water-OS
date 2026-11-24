"""Mix narration + score + sfx -> assets/audio/final_mix.wav (48 kHz stereo). The score is ducked under the narration."""
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt
from pathlib import Path
from math import gcd

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "assets/audio"
SR = 48000
TOTAL = 289.9
N = int(TOTAL * SR)


def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    out = np.zeros((N, 2), np.float32)
    n = min(N, len(x))
    out[:n] = x[:n]
    return out


def db(x):
    return 10 ** (x / 20)


nar, mus, sfx = load(A / "narration_full.wav"), load(A / "music.wav"), load(A / "sfx.wav")
# narration: gentle level normalisation to about -3 dBFS peak
nar *= 0.7 / (np.abs(nar).max() + 1e-9)
# ducking envelope from the narration (fast attack, slow release)
env = np.abs(nar).max(1)
win = int(0.05 * SR)
env = np.convolve(env, np.ones(win) / win, mode="same")
act = (env > 0.02).astype(np.float32)
sos = butter(1, 1.6, "low", fs=SR, output="sos")
act = sosfilt(sos, act)
duck = 1.0 - 0.55 * np.clip(act * 1.3, 0, 1)
mix = nar * 1.0 + mus * db(-15) * duck[:, None] + sfx * db(-14) * (1.0 - 0.3 * act)[:, None]
# soft limiter and peak normalisation to -1 dBFS
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
mix *= db(-1.0) / (np.abs(mix).max() + 1e-9)
# 1 s fade-in / 1.5 s fade-out
fi, fo = int(1.0 * SR), int(1.5 * SR)
mix[:fi] *= np.linspace(0, 1, fi)[:, None]
mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
sf.write(A / "final_mix.wav", mix, SR, subtype="PCM_16")
print("final_mix.wav written:", len(mix) / SR, "s; narration coverage %.0f%%" % (100 * act.mean()))
