"""SCENE 6 - Experiment 3: climate resilience (40 s).
Five climate regimes shown on one simulated school roof-and-tank system, then the three strategies' decision logic
and the evaluation metrics. NO Experiment 3 results were supplied: no outcome curves, rankings or percentages are shown.
The tank level here is an ILLUSTRATIVE water-balance animation of the regime, not a strategy result."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *
from engine3d import *

DUR = 40.0
ENV = Env(horizon=(26, 52, 92), zenith=(3, 8, 22), floor=(6, 11, 22), grid=(30, 130, 180), grid_size=4.0, fog=0.010,
          pool=(14, 62, 96), pool_r=38.0, pool_c=(0, 0))
# regime: name, start, end, rain 0-1, demand 0-1, level at end
REG = [("NORMAL RAINFALL", 0.6, 6.0, 0.35, 0.45, 0.62),
       ("PROLONGED DROUGHT", 6.0, 11.0, 0.0, 0.45, 0.22),
       ("EXTREME RAINFALL", 11.0, 16.0, 1.0, 0.45, 0.94),
       ("INCREASED DEMAND", 16.0, 21.0, 0.35, 0.95, 0.46),
       ("DROUGHT + INCREASED DEMAND", 21.0, 26.0, 0.0, 0.95, 0.08)]
LV = [(0.0, 0.58), (6.0, 0.62), (11.0, 0.22), (16.0, 0.94), (21.0, 0.46), (26.0, 0.08), (40.0, 0.30)]
CAM_KEYS = [(0, -10, 24, -56, 0, 3.5, 0), (13, -6, 21, -50, 2, 3.5, 0), (26, 0, 20, -48, 3, 4.0, 0), (40, 4, 22, -52, 4, 4.0, 0)]
TANK = (14.0, 0.0, 0.0)
TH, TR = 8.0, 3.4


def regime(t):
    for i, r in enumerate(REG):
        if r[1] <= t < r[2]:
            return i
    return -1


def smoothed(t, idx):
    """regime parameter with soft transitions"""
    ks = []
    for r in REG:
        ks += [(r[1] + 0.8, r[idx]), (r[2] - 0.2, r[idx])]
    return float(keys(t, [(0, REG[0][idx])] + ks + [(40, REG[-1][idx])])[0]) if not hasattr(keys(t, [(0, 0), (1, 1)]), "__len__") else float(keys(t, [(0, REG[0][idx])] + ks + [(40, REG[-1][idx])])[0])


def level(t):
    v = keys(t, [(a, b) for a, b in LV])
    return float(v[0] if hasattr(v, "__len__") else v)


def cam_at(t):
    k = keys(t, CAM_KEYS, ease=smooth)
    return Cam(k[0:3], k[3:6], fov=50)


def rain_cfg(t):
    i = regime(t)
    if i < 0:
        return None
    r = REG[i][3]
    if r <= 0.01:
        return None
    a = smooth(seg(t, REG[i][1], REG[i][1] + 0.8)) * (1 - smooth(seg(t, REG[i][2] - 0.6, REG[i][2])))
    return dict(center=(-4, 0, -2), n=int(250 + 900 * r), spread=44, top=30, speed=26 + 18 * r, wind=(1.5, 0.3), alpha=0.55 * a * (0.5 + 0.5 * r))


def build(w):
    t = w.t
    i = regime(t)
    L = level(t)
    rain = REG[i][3] if i >= 0 else 0.0
    dem = REG[i][4] if i >= 0 else 0.5
    drought = i in (1, 4)
    # school building with sloped roof catchment
    w.box((-20, 2.2, 0), (16, 4.4, 10), (46, 62, 90), edge=(56, 120, 160))
    w.quad([(-28, 4.4, -5.6), (-12, 4.4, -5.6), (-12, 6.2, 0), (-28, 6.2, 0)], (60, 78, 108), N=(0, 0.7, -0.7))
    w.quad([(-28, 6.2, 0), (-12, 6.2, 0), (-12, 4.4, 5.6), (-28, 4.4, 5.6)], (60, 78, 108), N=(0, 0.7, 0.7))
    for k in range(5):
        w.quad([(-27 + k * 3.2, 1.3, 5.02), (-25 + k * 3.2, 1.3, 5.02), (-25 + k * 3.2, 2.9, 5.02), (-27 + k * 3.2, 2.9, 5.02)],
               (150, 205, 225), N=(0, 0, 1), emissive=True, bias=-0.4)
    # gutter + pipe into tank
    p = [(-12.0, 4.4, 5.6), (-9.0, 1.0, 5.6), (TANK[0], 1.0, 5.6), (TANK[0], 1.0, 0.0)]
    w.tube(p, 0.26, (70, 112, 142))
    if rain > 0.02:
        n = int(6 + 22 * rain)
        u = (np.arange(n) / n + t * 0.5) % 1.0
        path = Path(p)
        w.dots(path.at(u), 0.2, (120, 225, 255), bias=-0.3)
    # tank + water
    w.cyl(TANK, TR, TH, (60, 110, 160), alpha=0.35, edge=(90, 210, 245), cap=(110, 200, 240), bottom=False)
    wh = max(0.2, TH * L)
    lowc = (255, 120, 70) if L < 0.2 else (24, 120, 180)
    w.cyl((TANK[0], 0.02, TANK[2]), TR - 0.25, wh, lowc, alpha=0.88, edge=lowc, cap=(110, 200, 240), bottom=False)
    # demand line: tap pipes, flow proportional to demand
    q = [(TANK[0] + TR, 1.6, 0), (30, 1.6, 0), (30, 1.6, -9)]
    w.tube(q, 0.26, (70, 112, 142))
    n = int(4 + 16 * dem)
    u = (np.arange(n) / n + t * 0.35 * (0.5 + dem)) % 1.0
    w.dots(Path(q).at(u), 0.2, AMBER if dem > 0.7 else (120, 225, 255), bias=-0.3)
    for k in range(3):
        w.box((30 + k * 4.5 - 1.5 * (k > 0) * 0, 1.7, -11), (3.2, 3.4, 3.2), (44, 60, 88), edge=(56, 120, 160))
    # sun glows in drought
    if drought:
        a = 1.0
        w.sphere((-6.0, 30.0, 40.0), 3.4, (255, 190, 90), glow=True)
    ea = lambda a, b: smooth(seg(t, a, a + .6)) * (1 - smooth(seg(t, b, b + .6)))
    w.label((-20, 6.6, 0), "Rooftop catchment", ("rainwater harvesting",), CYAN, -40, -120, ea(0.8, 5.6))
    w.label((TANK[0], TH + 0.4, 0), "Storage tank", ("illustrative level",), CYAN, 30, -120, ea(1.4, 5.8))


def timeline(f, t):
    a = smooth(seg(t, 0.5, 1.3))
    x0, x1, y0, y1 = 96, 1824, 880, 1030
    glass(f, x0, y0, x1, y1, a=0.66 * a, border=CYAN, ba=0.35 * a, r=16)
    text(f, "CLIMATE REGIMES", x0 + 26, y0 + 28, 17, CYAN, a, "lm", "semibold", spacing=4)
    text(f, "schematic · not experimental results", x1 - 26, y0 + 28, 16, GREY, a, "rm", "regular")
    n = len(REG)
    bw = (x1 - x0 - 52) / n
    for k, r in enumerate(REG):
        bx0 = x0 + 26 + k * bw
        on = smooth(seg(t, r[1], r[1] + 0.5)) * (1 - 0.55 * smooth(seg(t, r[2], r[2] + 0.5)))
        cur = r[1] <= t < r[2]
        col = AMBER if k in (1, 4) else (CYAN if k != 3 else ICE)
        cv2.rectangle(f, (int(bx0 + 4), y0 + 50), (int(bx0 + bw - 4), y0 + 134), tuple(int(c * (0.18 + 0.5 * on) * a) for c in (col)), -1)
        if cur:
            cv2.rectangle(f, (int(bx0 + 4), y0 + 50), (int(bx0 + bw - 4), y0 + 134), tuple(int(c * a) for c in col), 2, cv2.LINE_AA)
        text(f, r[0].title() if k != 4 else "Drought + Demand", bx0 + bw / 2, y0 + 78, 20, WHITE, a * (0.5 + 0.5 * on), "cm", "semibold")
        rain = ["moderate", "none", "extreme", "moderate", "none"][k]
        dm = ["baseline", "baseline", "baseline", "high", "high"][k]
        text(f, f"rain {rain} · demand {dm}", bx0 + bw / 2, y0 + 110, 16, GREY, a * (0.5 + 0.5 * on), "cm", "regular")
    # progress
    px = x0 + 26 + (x1 - x0 - 52) * clamp((t - 0.6) / (26.0 - 0.6))
    cv2.line(f, (int(px), y0 + 44), (int(px), y0 + 140), (255, 255, 255), 2, cv2.LINE_AA)


def strategies(f, t):
    a = smooth(seg(t, 26.4, 27.2)) * (1 - smooth(seg(t, 39.2, 39.9)))
    if a < 0.01:
        return
    cards = [("A", "FIXED-RULE", "Reacts to thresholds", ("if level < limit: restrict use", "no model of why the system changed"), ICE, 27.4),
             ("B", "FORECAST-BASED", "Acts on predicted inflow and demand", ("plans from a forecast", "assumes the forecast model is right"), CYAN, 29.0),
             ("C", "EPISTEMIC + SCIENTIFIC ADAPTIVE", "Checks its own understanding first", ("tests the model against physical law", "revises when evidence demands, then acts"), AMBER, 30.6)]
    for k, (ltr, name, sub, lines, col, st) in enumerate(cards):
        ca = smooth(seg(t, st, st + 0.7)) * a
        x0 = 96 + k * 580
        glass(f, x0, 190, x0 + 550, 560, a=0.7 * ca, border=col, ba=0.55 * ca, r=18)
        text(f, f"STRATEGY {ltr}", x0 + 30, 232, 18, col, ca, "lm", "semibold", spacing=4)
        text(f, name, x0 + 30, 284, 27 if k < 2 else 23, WHITE, ca, "lm", "semibold")
        text(f, sub, x0 + 30, 336, 22, ICE, ca, "lm", "regular")
        for j, ln in enumerate(lines):
            text(f, ln, x0 + 30, 400 + 44 * j, 21, GREY, ca, "lm", "regular")
        cv2.line(f, (x0 + 30, 506), (x0 + 520, 506), tuple(int(c * 0.5 * ca) for c in col), 1, cv2.LINE_AA)
        text(f, "same scenarios · same conditions", x0 + 30, 534, 17, GREY, ca, "lm", "regular")
    ia = smooth(seg(t, 32.0, 32.8)) * a
    text(f, "Identical environment for every strategy", W / 2, 150, 24, CYAN, ia, "cm", "semibold", spacing=3)


def metrics(f, t):
    a = smooth(seg(t, 34.0, 34.8)) * (1 - smooth(seg(t, 39.2, 39.9)))
    if a < 0.01:
        return
    glass(f, 96, 600, 1824, 1010, a=0.72 * a, border=CYAN, ba=0.5 * a, r=18)
    text(f, "EVALUATION METRICS", 130, 640, 19, CYAN, a, "lm", "semibold", spacing=4)
    names = [("Reliability", "water available when needed"), ("Unmet demand", "demand that could not be served"),
             ("Shortage duration", "time spent in shortage"), ("Recovery time", "time to return to normal")]
    for k, (n, d) in enumerate(names):
        ca = smooth(seg(t, 34.4 + 0.5 * k, 35.0 + 0.5 * k)) * a
        x = 130 + k * 430
        cv2.rectangle(f, (x, 690), (x + 380, 694), tuple(int(c * ca) for c in CYAN), -1)
        text(f, n, x, 740, 30, WHITE, ca, "lm", "semibold")
        text(f, d, x, 780, 19, GREY, ca, "lm", "regular")
    ra = smooth(seg(t, 36.8, 37.6)) * a
    glass(f, 130, 840, 1790, 980, a=0.55 * ra, border=AMBER, ba=0.5 * ra, r=14, bw=1)
    text(f, "Outcome values are not shown in this film.", 160, 886, 26, AMBER, ra, "lm", "semibold")
    text(f, "Experiment 3 results were not supplied for this video; no rankings or improvement figures are claimed.", 160, 934, 21, ICE, ra, "lm", "regular")


def frame(t):
    cam = cam_at(t)
    f = render_world(cam, ENV, build, t, refl=0.25, bloom=0.85, thr=135, rain=rain_cfg(t))
    if t > 26.0:
        # dim 3D for the strategy/metric cards
        d = 0.5 * smooth(seg(t, 26.0, 27.0))
        f = (f.astype(np.float32) * (1 - d)).astype(np.uint8)
    ha = smooth(seg(t, 0.3, 1.0))
    text(f, "SCENE 06  ·  EXPERIMENT 3", 96, 62, 20, CYAN, ha, "lm", "semibold", spacing=5)
    text(f, "Climate Resilience", 96, 112, 48, WHITE, ha, "lm", "light")
    i = regime(t)
    if i >= 0:
        ra = smooth(seg(t, REG[i][1], REG[i][1] + 0.6)) * (1 - smooth(seg(t, REG[i][2] - 0.5, REG[i][2])))
        text(f, REG[i][0], 96, 170, 26, AMBER if i in (1, 4) else ICE, ra, "lm", "semibold", spacing=4)
    tag(f, "ILLUSTRATIVE SIMULATION · NOT EXPERIMENTAL RESULTS", 1824, 62, AMBER, 18, smooth(seg(t, 0.6, 1.4)) * 0.95, "r")
    if t < 26.5:
        timeline(f, t)
    strategies(f, t)
    metrics(f, t)
    return finish(f, t, DUR, fin=0.7, fout=0.6)


if __name__ == "__main__":
    scene_main("scene_06_climate", DUR, frame)
