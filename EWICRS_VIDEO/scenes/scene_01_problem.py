"""SCENE 1 - The water intelligence problem (45 s).
Reservoir basin + urban network flyover, cited statistic, the intelligence gap, title reveal."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *
from engine3d import *

DUR = 45.0
ENV = Env(horizon=(26, 52, 92), zenith=(3, 8, 22), floor_on=False, fog=0.008)

CAM_KEYS = [  # t, pos(3), target(3)
    (0.0, -62, 30, -12, -8, 2, 34),
    (8.0, -50, 22, 0, -14, 2, 33),
    (13.0, -40, 38, -2, -24, 1.0, 34),
    (21.0, -36, 34, 2, -24, 1.0, 34),
    (24.5, -2, 14, -4, 24, 5, 30),
    (30.0, 8, 17, 4, 34, 5, 32),
    (37.0, 6, 24, -30, 28, 5, 30),
    (45.0, 12, 34, -64, 26, 5, 30),
]
XG = np.linspace(-70, 70, 121)
ZG = np.linspace(-20, 80, 87)
PX, PZ = np.meshgrid(XG, ZG)
R2 = ((PX + 24) / 30.0) ** 2 + ((PZ - 32) / 22.0) ** 2
YG = 14.0 * (1 - np.exp(-R2 * 0.9)) + 1.1 * np.sin(PX * 0.21) * np.cos(PZ * 0.17) * (1 - np.exp(-R2 * 0.9))
PL = np.clip((PX - 4) / 14.0, 0, 1)
PL = PL * PL * (3 - 2 * PL)
YG = (YG * (1 - PL) + 6.0 * PL).astype(np.float32)
YC = (YG[:-1, :-1] + YG[1:, 1:]) / 2
rng = np.random.default_rng(4)
CITY = []
for ix in range(8):
    for iz in range(5):
        x, z = 22 + ix * 6.2, 9 + iz * 9.4
        if abs(z - 30) < 3.0 or any(abs(x - cx) < 2.6 for cx in (24, 36, 48)):
            continue
        CITY.append((x, z, 2.5 + rng.random() * 9.0, 3.0 + rng.random() * 2.0))
MAIN = Path([(-8, 2.4, 30), (4, 5.0, 30), (14, 6.4, 30), (62, 6.4, 30)])
BRANCH = [Path([(x, 6.4, 30), (x, 6.4, 8)]) for x in (24, 36, 48)] + [Path([(x, 6.4, 30), (x, 6.4, 52)]) for x in (24, 36, 48)]
X_POLL = 27.0


def level(t):
    return float(keys(t, [(0, 1.8), (11, 3.7), (20.5, 0.9), (23.5, 2.4), (28, 1.6), (45, 1.8)])[0])


def cam_at(t):
    k = keys(t, CAM_KEYS, ease=smooth)
    return Cam(k[0:3], k[3:6], fov=54)


def build(w):
    t = w.t
    L = level(t)
    base = np.array([26, 46, 72], np.float32)
    dry = np.array([128, 98, 66], np.float32)
    col = np.broadcast_to(base, YC.shape + (3,)).copy()
    maxl = 3.9
    drymask = ((YC < maxl) & (YC > L - 0.05) & (np.abs(YC) < 20)).astype(np.float32)[..., None]
    stripe = (0.86 + 0.14 * np.sin(YC * 2.6))[..., None]
    col = col * (1 - drymask * 0.85) + dry * stripe * drymask * 0.85
    pm = (((XG[:-1] + XG[1:]) / 2) > 8)[None, :, None]
    col = np.where(pm, col * 0.55 + np.array([20, 34, 54], np.float32) * 0.45, col)  # plateau
    w.terrain(XG, ZG, YG, col, water=L, water_col=(14, 80, 122), t=t)
    # city
    for x, z, h, s in CITY:
        w.box((x, 6.0 + h / 2, z), (s, h, s), (44, 60, 88), edge=(56, 120, 160))
        lit = int(t * 0.4 + x * 3 + z) % 3 != 0
        if lit:
            w.quad([(x - s * .3, 6 + h * .35, z + s / 2 + .02), (x + s * .3, 6 + h * .35, z + s / 2 + .02),
                    (x + s * .3, 6 + h * .6, z + s / 2 + .02), (x - s * .3, 6 + h * .6, z + s / 2 + .02)], (150, 205, 225), N=(0, 0, 1), emissive=True, bias=-0.4)
    # reservoir intake tower, tanks on the plateau
    w.cyl((-8, 0, 30), 1.5, 5.0, (60, 90, 120), edge=(90, 210, 245))
    for z in (14, 46):
        w.cyl((16, 6.0, z), 2.2, 4.4, (80, 150, 200), alpha=0.9, edge=(90, 210, 245), cap=(110, 200, 240))
    # pipelines
    for p in [MAIN] + BRANCH:
        w.tube(p.p, 0.34, (70, 112, 142))
    u = (np.arange(34) / 34 + t * 3.2 / MAIN.L) % 1.0
    w.dots(MAIN.at(u), 0.26, (120, 225, 255), bias=-0.3)
    for i, b in enumerate(BRANCH):
        u = (np.arange(10) / 10 + t * 3.0 / b.L + i * 0.17) % 1.0
        w.dots(b.at(u), 0.22, (120, 225, 255), bias=-0.3)
    # pollutant plume travelling downstream along the main line (after t = 21.5)
    if t > 21.5:
        head = clamp((t - 21.5) * 5.0 / MAIN.L * 1.0) * MAIN.L
        s0 = MAIN.L * 0.0
        xs0 = MAIN.c[2] + (X_POLL - MAIN.p[2][0])
        n = 26
        d = xs0 + np.linspace(0, max(head - xs0 + MAIN.c[2] * 0, 0.01), n) if head > xs0 else None
        if d is not None:
            pts = MAIN.at(np.clip(d / MAIN.L, 0, 1))
            w.dots(pts, 0.34, (255, 140, 60), bias=-0.5)
    w.sphere((X_POLL, 7.4, 30), 0.5, AMBER, glow=True) if t > 21 else None
    ea = lambda a, b: smooth(seg(t, a, a + .5)) * (1 - smooth(seg(t, b, b + .6)))
    w.label((-8, 4.0, 30), "Reservoir intake", ("storage rises and falls with rain",), CYAN, -20, -170, ea(16.0, 21.0))
    w.label((X_POLL, 7.4, 30), "Pollutant entry", ("synthetic event, unseen by the network",), AMBER, 40, -150, ea(22.4, 28.0))
    w.label((36, 12, 30), "Urban distribution network", ("pipes, tanks, taps",), ICE, 30, -170, ea(9.0, 14.5))


def rain_cfg(t):
    a = smooth(seg(t, 0.2, 1.5)) * (1 - smooth(seg(t, 11.0, 13.0)))
    b = smooth(seg(t, 20.5, 21.5)) * (1 - smooth(seg(t, 24.5, 26.0)))
    al = max(a, b) * 0.6
    if al < 0.02:
        return None
    return dict(center=(-10, 0, 30), n=700, spread=60, top=44, speed=36, wind=(2.5, 0.5), alpha=al)


def gap_graphic(f, t):
    a0, a1 = 28.3, 36.6
    a = smooth(seg(t, a0, a0 + 0.8)) * (1 - smooth(seg(t, a1, a1 + 0.9)))
    if a < 0.01:
        return
    x0, y0, x1, y1 = 1130, 230, 1840, 650
    glass(f, x0, y0, x1, y1, a=0.68 * a, border=CYAN, ba=0.4 * a, r=18)
    text(f, "THE INTELLIGENCE GAP", x0 + 34, y0 + 40, 20, CYAN, a, "lm", "semibold", spacing=5)
    text(f, "illustrative schematic", x1 - 30, y0 + 40, 17, GREY, a * 0.9, "rm", "regular")
    gx0, gx1, gy0, gy1 = x0 + 50, x1 - 50, y0 + 90, y1 - 70
    n = 120
    u = np.linspace(0, 1, n)
    real = 0.5 + 0.20 * np.sin(6.0 * u) + 0.08 * np.sin(14 * u + 1.0) + 0.22 * np.clip(u - 0.5, 0, 1)
    drift = np.clip((u - 0.42) / 0.58, 0, 1)
    model = real - 0.30 * drift ** 1.4 + 0.00
    P = lambda arr: np.stack([gx0 + u * (gx1 - gx0), gy1 - arr * (gy1 - gy0)], 1)
    pr, pm = P(real), P(model)
    prog = smooth(seg(t, a0 + 0.8, a0 + 4.8))
    k = max(2, int(n * prog))
    fx = FX()
    if k > 2:
        lo = int(n * 0.42)
        if k > lo:
            poly = np.concatenate([pr[lo:k], pm[lo:k][::-1]]).astype(np.int32)
            ov = f.copy()
            cv2.fillPoly(ov, [poly], (190, 52, 60))
            f[:] = cv2.addWeighted(ov, 0.38 * a, f, 1 - 0.38 * a, 0)
        fx.poly(pr[:k], CYAN, 4, a)
        fx.poly(pm[:k], AMBER, 4, a)
        fx.circle(pr[k - 1], 7, CYAN, -1, a)
        fx.circle(pm[k - 1], 7, AMBER, -1, a)
    f[:] = fx.apply(f, 0.8)
    text(f, "PHYSICAL REALITY", gx0, y1 - 34, 20, CYAN, a, "lm", "semibold", spacing=3)
    text(f, "MODEL UNDERSTANDING", gx0 + 330, y1 - 34, 20, AMBER, a, "lm", "semibold", spacing=3)
    la = smooth(seg(t, a0 + 4.2, a0 + 5.2)) * a
    if la > 0.01:
        i = int(n * 0.86)
        mx, my = (pr[i, 0] + pm[i, 0]) / 2, (pr[i, 1] + pm[i, 1]) / 2
        text(f, "Who notices?", gx1 - 10, gy0 + 4, 30, WHITE, la, "rm", "semibold", glow=4)
        cv2.line(f, (int(mx), int(my - 6)), (int(gx1 - 120), int(gy0 + 22)), (255, 255, 255), 1, cv2.LINE_AA)


def title_reveal(f, t):
    a0 = 37.2
    m = smooth(seg(t, a0 - 0.4, a0 + 1.2))
    if m <= 0.001:
        return f
    f = (f.astype(np.float32) * (1 - 0.48 * m)).astype(np.uint8)
    p = smooth(seg(t, a0, a0 + 2.6))
    s = "EWICRS"
    size = 170
    sp = 30
    wd = text_width(s, size, "light", spacing=sp)
    x0 = W / 2 - wd / 2
    # layer: text with a left-to-right wipe + light bar
    layer = np.zeros_like(f)
    text(layer, s, x0, 470, size, WHITE, 1.0, "lm", "light", spacing=sp, glow=14)
    edge = x0 - 40 + p * (wd + 120)
    mask = np.zeros(f.shape[:2], np.float32)
    mask[:, :max(0, int(edge))] = 1
    mask = cv2.GaussianBlur(mask, (0, 0), 28)[..., None]
    f = np.clip(f.astype(np.float32) + layer.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    if 0 < p < 1:
        fx = FX()
        fx.line((edge, 380), (edge, 560), ICE, 3, 0.9)
        f = fx.apply(f, 1.1)
    sa = smooth(seg(t, a0 + 2.2, a0 + 3.4))
    text(f, "EPISTEMIC WATER INTELLIGENCE & CLIMATE RESILIENCE SYSTEM", W / 2, 590, 30, ICE, sa, "cm", "regular", spacing=7, sym=False)
    ln = smooth(seg(t, a0 + 2.4, a0 + 4.0))
    cv2.line(f, (int(W / 2 - 360 * ln), 626), (int(W / 2 + 360 * ln), 626), tuple(int(c * 0.9) for c in CYAN), 2, cv2.LINE_AA)
    return f


def frame(t):
    cam = cam_at(t)
    f = render_world(cam, ENV, build, t, refl=0.0, bloom=0.8, thr=140, rain=rain_cfg(t))
    # opening question
    qa = lambda a, b: smooth(seg(t, a, a + 0.9)) * (1 - smooth(seg(t, b, b + 0.8)))
    if t < 7.2:
        dim = 0.55 * smooth(seg(t, 0.3, 1.4)) * (1 - smooth(seg(t, 5.8, 7.0)))
        f = (f.astype(np.float32) * (1 - dim)).astype(np.uint8)
        lines = [("WHAT IF THE SYSTEM ITSELF", 1.0), ("CANNOT RECOGNIZE WHEN ITS", 1.8), ("UNDERSTANDING OF REALITY IS WRONG?", 2.7)]
        for i, (s, a0) in enumerate(lines):
            a = qa(a0, 5.6)
            col = AMBER if i == 2 else WHITE
            text(f, s, W / 2, 430 + i * 82, 58, col, a, "cm", "light", spacing=4, glow=6 if i == 2 else 0)
    # statistic card
    ca = smooth(seg(t, 7.0, 7.9)) * (1 - smooth(seg(t, 14.0, 14.9)))
    if ca > 0.01:
        glass(f, 96, 700, 880, 980, a=0.66 * ca, border=CYAN, ba=0.4 * ca, r=18)
        k = smooth(seg(t, 7.0, 8.9))
        val = 2.1 * k
        text(f, f"{val:.1f} billion", 130, 800, 104, WHITE, ca, "lm", "light", glow=5)
        text(f, "people still lack safely managed drinking water", 130, 874, 29, ICE, ca, "lm", "regular")
        text(f, "about 1 in 4 people worldwide", 130, 914, 25, GREY, ca, "lm", "regular")
        text(f, "Source: WHO / UNICEF JMP, 2025 report (data 2000–2024)", 130, 954, 19, GREY, 0.9 * ca, "lm", "regular")
    # reservoir level gauge (illustrative)
    ga = smooth(seg(t, 12.0, 13.0)) * (1 - smooth(seg(t, 26.0, 27.0)))
    if ga > 0.01:
        gx, gy0, gy1 = 130, 600, 960
        glass(f, 96, 560, 420, 1000, a=0.55 * ga, border=CYAN, ba=0.35 * ga, r=16)
        text(f, "RESERVOIR LEVEL", 124, 600, 19, CYAN, ga, "lm", "semibold", spacing=4)
        text(f, "illustrative", 124, 628, 17, GREY, ga, "lm", "regular")
        cv2.rectangle(f, (gx, 660), (gx + 70, 960), (80, 110, 140), 2, cv2.LINE_AA)
        fill = clamp((level(t) - 0.5) / 3.6)
        hh = int(300 * fill)
        col = RED if level(t) < 1.3 else CYAN
        cv2.rectangle(f, (gx + 4, 958 - hh), (gx + 66, 958), tuple(int(c * 0.75 * ga) for c in col), -1)
        lbl = "extreme rainfall" if 20.8 < t < 24.8 else ("drought" if 14.5 < t < 20.8 else ("rainfall" if t < 14.5 else ""))
        if lbl:
            text(f, lbl.upper(), 230, 810, 24, AMBER if lbl == "drought" else ICE, ga, "lm", "semibold", spacing=3)
    gap_graphic(f, t)
    f = title_reveal(f, t)
    ta = smooth(seg(t, 8.0, 8.8)) * (1 - smooth(seg(t, 36.0, 37.0)))
    tag(f, "SYNTHETIC 3D VISUALIZATION", 1824, 84, AMBER, 19, ta * 0.9, "r")
    return finish(f, t, DUR, fin=1.2, fout=1.0)


if __name__ == "__main__":
    scene_main("scene_01_problem", DUR, frame)
