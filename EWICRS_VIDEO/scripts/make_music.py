"""Synthesises the original score (ambient pads, sub drone, sparse bell arpeggios, tension pulse) and the sound effects,
timed to the film's cues. Output: assets/audio/music.wav, assets/audio/sfx.wav (48 kHz stereo float32).
No samples or third-party music are used; everything is generated here with NumPy/SciPy."""
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
TOTAL = 289.9
OFF = {"s1": 0, "s2": 45, "s3": 75, "s4": 115, "s5": 155, "s6": 195, "s7": 235, "s8": 275}
N = int(TOTAL * SR) + SR
rng = np.random.default_rng(11)
T = np.arange(N) / SR


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def env_seg(a, b, fi=2.0, fo=2.0):
    e = np.clip((T - a) / fi, 0, 1) * np.clip((b - T) / fo, 0, 1)
    return e * e * (3 - 2 * e)


def reverb(x, secs=3.0, wet=0.35):
    n = int(secs * SR)
    ir = rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / (SR * secs / 4.5))[:, None]
    ir = lp(ir.T, 5000).T
    out = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    out /= np.abs(out).max() + 1e-9
    return x * (1 - wet) + out * wet * np.abs(x).max()


# ---------------------------------------------------------------- chord plan (A minor world, slow harmonic rhythm)
CHORDS = [  # midi notes, bass
    ([57, 60, 64, 67], 33), ([53, 57, 60, 64], 29), ([48, 55, 60, 64], 36), ([55, 59, 62, 66], 31),
]
SEG = 18.0


def pad():
    out = np.zeros((N, 2), np.float32)
    nseg = int(np.ceil(TOTAL / SEG)) + 1
    for s in range(nseg):
        notes, bass = CHORDS[s % 4]
        a, b = s * SEG - 3.0, (s + 1) * SEG + 3.0
        i0, i1 = max(0, int(a * SR)), min(N, int(b * SR))
        if i1 <= i0:
            continue
        t = T[i0:i1]
        e = np.clip((t - a) / 4.0, 0, 1) * np.clip((b - t) / 4.0, 0, 1)
        e = (e * e * (3 - 2 * e)).astype(np.float32)
        for k, m in enumerate(notes):
            for det, pan in ((-0.12, 0.0), (0.0, 0.5), (0.12, 1.0)):
                f = hz(m) * 2 ** (det / 12)
                ph = rng.uniform(0, 6.28)
                lfo = 0.5 + 0.5 * np.sin(2 * np.pi * (0.05 + 0.013 * k) * t + ph)
                w = np.sin(2 * np.pi * f * t + ph) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * lfo + 0.15 * np.sin(2 * np.pi * 3 * f * t)
                w = (w * e * 0.045).astype(np.float32)
                out[i0:i1, 0] += w * (1 - pan)
                out[i0:i1, 1] += w * pan
        sub = (np.sin(2 * np.pi * hz(bass) * t) * e * 0.12).astype(np.float32)
        out[i0:i1, 0] += sub
        out[i0:i1, 1] += sub
    return out


def bells(starts_ends_density):
    out = np.zeros((N, 2), np.float32)
    scale = [69, 72, 74, 76, 79, 81, 84]
    for a, b, dens, vol in starts_ends_density:
        t = a
        while t < b:
            m = scale[rng.integers(0, len(scale))] + (12 if rng.random() < 0.15 else 0)
            i0 = int(t * SR)
            L = int(3.2 * SR)
            tt = np.arange(L) / SR
            f = hz(m)
            tone = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2.01 * f * tt) * np.exp(-tt * 3) + 0.2 * np.sin(2 * np.pi * 3.9 * f * tt) * np.exp(-tt * 6)) * np.exp(-tt * 1.6)
            tone *= vol * min(1.0, (b - t) / 2.0)
            pan = rng.uniform(0.25, 0.75)
            j = min(L, N - i0)
            out[i0:i0 + j, 0] += (tone[:j] * (1 - pan)).astype(np.float32)
            out[i0:i0 + j, 1] += (tone[:j] * pan).astype(np.float32)
            t += rng.exponential(1.0 / dens) + 0.4
    return out


def pulse(a, b, bpm=60, vol=0.07, f=55.0):
    """low heartbeat-like pulse for the tension scenes"""
    out = np.zeros(N, np.float32)
    step = 60.0 / bpm
    t = a
    while t < b:
        i0 = int(t * SR)
        L = int(0.5 * SR)
        tt = np.arange(L) / SR
        k = np.sin(2 * np.pi * (f + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 8)
        j = min(L, N - i0)
        out[i0:i0 + j] += (k[:j] * vol * min(1, (b - t) / 2)).astype(np.float32)
        t += step
    return out


def build_music():
    m = pad()
    # scene-wise bell arpeggios (sparser in tension scenes, fuller toward the end)
    m += bells([(OFF["s1"] + 8, 45, 0.35, 0.020), (OFF["s2"], 75, 0.6, 0.025), (OFF["s3"], 115, 0.5, 0.022),
                (OFF["s4"], 155, 0.25, 0.018), (OFF["s5"], 195, 0.3, 0.018), (OFF["s6"], 235, 0.5, 0.022),
                (OFF["s7"], 275, 0.55, 0.024), (275, 289, 0.8, 0.030)])
    p = pulse(OFF["s4"], OFF["s5"], 54, 0.08) + pulse(OFF["s5"] + 3, OFF["s5"] + 33, 50, 0.07) + pulse(OFF["s6"] + 20, OFF["s6"] + 26, 56, 0.06)
    m[:, 0] += p
    m[:, 1] += p
    # opening swell + final swell
    sw = (np.clip(T / 18.0, 0, 1) ** 2 * (T < 37.2)) * 0.5
    air = lp(rng.standard_normal(N), 2200)
    air = hp(air, 300) * 0.02
    m[:, 0] += (air * sw).astype(np.float32)
    m[:, 1] += (np.roll(air, 977) * sw).astype(np.float32)
    m = reverb(m, 3.2, 0.32)
    # master macro-dynamics: quiet start, build to title, supportive mid, lift at the end
    macro = np.interp(T, [0, 3, 30, 37.2, 38.0, 70, 160, 230, 275, 286, TOTAL],
                      [0.0, 0.55, 0.75, 0.95, 0.65, 0.70, 0.65, 0.72, 0.95, 0.9, 0.0])
    m *= macro[:, None].astype(np.float32)
    m /= np.abs(m).max() + 1e-9
    return m * 0.9


# ---------------------------------------------------------------- SFX
def noise_whoosh(center, dur=1.6, vol=0.5, up=True):
    out = np.zeros((N, 2), np.float32)
    i0 = max(0, int((center - dur * 0.6) * SR))
    L = int(dur * SR)
    tt = np.arange(L) / SR
    x = rng.standard_normal((L, 2))
    fc = (300 * (1 + 7 * (tt / dur) ** 1.7)) if up else (2400 * (1 - 0.85 * (tt / dur)))
    y = np.zeros_like(x)
    for c in range(2):
        for k in range(0, L, 2048):
            seg = x[k:k + 2048, c]
            f = float(np.clip(fc[k], 120, 9000))
            y[k:k + 2048, c] = sosfilt(butter(2, [f * 0.6, f * 1.6], "band", fs=SR, output="sos"), seg)
    y *= (np.sin(np.pi * tt / dur) ** 2)[:, None] * vol
    j = min(L, N - i0)
    out[i0:i0 + j] = y[:j]
    return out


def boom(at, vol=0.9):
    out = np.zeros((N, 2), np.float32)
    i0 = int(at * SR)
    L = int(3.5 * SR)
    tt = np.arange(L) / SR
    b = np.sin(2 * np.pi * (38 + 90 * np.exp(-tt * 9)) * tt) * np.exp(-tt * 1.3)
    sh = np.zeros(L)
    for f in (880, 1320, 1760, 2640):
        sh += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 1.8) * 0.12
    y = (b + sh) * vol
    j = min(L, N - i0)
    out[i0:i0 + j, 0] = y[:j]
    out[i0:i0 + j, 1] = y[:j]
    return out


def ping(at, f=1760.0, vol=0.35, dur=1.0):
    out = np.zeros((N, 2), np.float32)
    i0 = int(at * SR)
    L = int(dur * SR)
    tt = np.arange(L) / SR
    y = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt * 7) * vol
    j = min(L, N - i0)
    out[i0:i0 + j, 0] = y[:j]
    out[i0:i0 + j, 1] = y[:j]
    return out


def tick(at, vol=0.18):
    return ping(at, 3200.0, vol, 0.12)


def thud(at, vol=0.7):
    return boom(at, vol * 0.5) * 0.7


def build_sfx():
    s = np.zeros((N, 2), np.float32)
    # transitions
    for k in ("s2", "s3", "s4", "s5", "s6", "s7", "s8"):
        s += noise_whoosh(OFF[k] + 0.2, 1.5, 0.30, up=True)
    # scene 1: title reveal riser + impact, pollution event ping, rain-start tick
    s += noise_whoosh(37.2 + 1.0, 4.4, 0.45, up=True)
    s += boom(37.4, 0.8)
    s += ping(OFF["s1"] + 22.2, 660, 0.3)
    # scene 2: layer activations
    for st in (3.6, 10.2, 15.0, 20.0):
        s += ping(OFF["s2"] + st, 880 + 110 * (st > 12), 0.28)
    s += ping(OFF["s2"] + 25.4, 523, 0.3)
    # scene 3: tank/dashboard focus pings (panel cue times)
    for st in (28.8, 30.4, 32.0, 33.6, 35.8, 38.0):
        s += tick(OFF["s3"] + st)
    # scene 4: leak onset, residual highlights, hypothesis cards
    s += thud(OFF["s4"] + 7.0, 0.6)
    for st in (16.3, 18.5, 20.6):
        s += ping(OFF["s4"] + st, 1320, 0.3)
    for i in range(4):
        s += tick(OFF["s4"] + 23.2 + 1.4 * i)
    for i in range(6):
        s += tick(OFF["s4"] + 34.0 + 0.8 * i, 0.12)
    # scene 5: pollutant entry, valve close
    s += ping(OFF["s5"] + 5.0, 440, 0.4)
    for i in range(5):
        s += tick(OFF["s5"] + 13.2 + 1.6 * i)
    s += thud(OFF["s5"] + 32.4, 0.9)
    # scene 6: regime changes
    for st in (0.6, 6.2, 11.2, 16.2, 21.2):
        s += ping(OFF["s6"] + st, 990, 0.22)
    for st in (27.4, 29.0, 30.6):
        s += tick(OFF["s6"] + st)
    # scene 7: sensors appearing, pipeline, status
    for i in range(4):
        s += ping(OFF["s7"] + 8.0 + 2.2 * i, 1480 + 120 * i, 0.22)
    for i in range(7):
        s += tick(OFF["s7"] + 19.6 + 1.05 * i, 0.12)
    s += boom(OFF["s7"] + 34.4, 0.45)
    # scene 8: final lines
    for i in range(3):
        s += ping(OFF["s8"] + 11.6 + 0.8 * i, 880 * (1 + 0.25 * i), 0.3)
    s += boom(OFF["s8"] + 12.2, 0.55)
    s = reverb(s, 1.8, 0.18)
    s /= np.abs(s).max() + 1e-9
    return s * 0.8


if __name__ == "__main__":
    out = ROOT / "assets/audio"
    m = build_music()
    sf.write(out / "music.wav", m[: int(TOTAL * SR)], SR, subtype="FLOAT")
    s = build_sfx()
    sf.write(out / "sfx.wav", s[: int(TOTAL * SR)], SR, subtype="FLOAT")
    print("music + sfx written", m.shape, s.shape)
