"""SCENE 3 - Physical digital twin (40 s): 3D school water system + the real dashboard screenshot."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *
from engine3d import *

DUR = 40.0
ENV = Env(horizon=(22, 48, 88), zenith=(3, 8, 22), floor=(6, 11, 22), grid=(30, 130, 180), grid_size=4.0, fog=0.011,
          pool=(14, 62, 96), pool_r=40.0, pool_c=(4, 2))

CAM_KEYS = [  # t, pos(3), target(3)
    (0.0, -36, 14, 40, 3, 3, 0),
    (8.0, -14, 10, 44, 6, 3, 2),
    (15.0, 8, 12, 48, 6, 2.5, 3),
    (16.5, 20, 8, 31, 8, 1.5, 8),
    (22.0, -2, 8, 30, 3, 1.5, 8),
    (26.5, -14, 9, 38, 4, 2.5, 2),
]

P_I = Path([(52, 0.7, 2), (24.3, 0.7, 2)])                                        # inflow from source
P_R = Path([(14, 6.5, 6.6), (14, 0.65, 6.6), (16, 0.65, 6.6), (16, 0.9, 3.4), (18.2, 0.9, 3.4)])  # roof -> tank
P_G = Path([(-14, 6.6, 6.7), (14, 6.6, 6.7)])                                      # gutter
P_D = Path([(18.2, 0.9, 0.8), (16.6, 0.9, 0.8), (16.6, 0.65, 8.4), (-13, 0.65, 8.4)])  # distribution
X_LEAK, X_TAPS, X_POLL, X_UP, X_DN, X_VALVE = -8.0, (-5.0, -11.0), 12.0, 14.6, 5.0, -1.0

BD = Backdrop(seed=11)
DASH = Shot("dashboard.png", crop=(1, 3, 707, 471))
TABLE = Shot("table.png")
PANELS = {
    "storage": (0, 0, 357, 159), "rain": (357, 0, 705, 159), "temp": (0, 159, 357, 313),
    "quality": (357, 159, 705, 313), "avp": (0, 313, 357, 468), "risk": (357, 313, 705, 468),
}
TOUR = [  # key, start, title, line1, line2
    ("storage", 28.3, "Tank Water Storage", "falls from ≈1000 to 0 by t ≈ 43", "rises to ≈410 near t ≈ 79"),
    ("rain", 30.2, "Rainfall", "≈20–60, then near 0 for t ≈ 30–60", "≈80–120 for t ≈ 60–80"),
    ("temp", 32.1, "Temperature", "≈28–33, then ≈38–43 for t ≈ 30–60", "≈24–30 after t ≈ 60"),
    ("quality", 34.0, "Water Quality", "holds at 100", "dips to ≈67 for t ≈ 65–76"),
    ("avp", 35.8, "Actual vs Predicted Storage", "curves coincide until storage reaches 0", "leakage event marked at t = 50"),
    ("risk", 38.0, "Risk Score", "0 → 20–60 from t ≈ 33", "80 for t ≈ 65–76, then 40"),
]
DASH_RECT = (70, 112, 1450, 1016)
OV = (353, 234, 1.93)


def cam_at(t):
    k = keys(t, CAM_KEYS, ease=smooth)
    return Cam(k[0:3], k[3:6], fov=52)


def flow(w, path, n, speed, color, r=0.20, t0=0.0, bias=-0.3):
    u = (np.arange(n) / n + (w.t + t0) * speed / path.L) % 1.0
    w.dots(path.at(u), r, color, bias=bias)


def build(w):
    t = w.t
    # ground pads
    w.box((0, 0.08, 3), (34, 0.16, 20), (18, 24, 36), bias=60)
    for z in (2, -5):
        w.cyl((21, 0, z), 4.4, 0.18, (20, 28, 44), bias=60)
    # school block
    w.box((0, 3.0, 0), (26, 6, 11), (54, 68, 94))
    w.box((0, 6.25, 0), (28.4, 0.5, 13.2), (36, 46, 70), edge=(60, 120, 160))
    for i in range(9):
        x = -11.2 + 2.8 * i
        lit = ((i * 7 + int(t * 0.5)) % 5) != 0
        w.quad([(x - 0.9, 1.8, 5.52), (x + 0.9, 1.8, 5.52), (x + 0.9, 4.4, 5.52), (x - 0.9, 4.4, 5.52)],
               (150, 205, 225) if lit else (24, 44, 66), N=(0, 0, 1), emissive=lit, bias=-0.4)
    w.quad([(-1.1, 0, 5.52), (1.1, 0, 5.52), (1.1, 2.6, 5.52), (-1.1, 2.6, 5.52)], (16, 24, 36), N=(0, 0, 1), bias=-0.5)
    # rooftop catchment overlay
    w.quad([(-14, 6.52, -6.6), (14, 6.52, -6.6), (14, 6.52, 6.6), (-14, 6.52, 6.6)], (30, 130, 185), N=(0, 1, 0),
           alpha=0.26, edge=(90, 210, 245), ew=2, bias=-0.6)
    # gutter and downpipe
    w.tube([(-14, 6.55, 6.7), (14, 6.55, 6.7)], 0.2, (95, 125, 150))
    # tanks (glass shell + water + level)
    lvl = 0.62 + 0.18 * math.sin(t * 0.35)
    for z, f in ((2, 1.0), (-5, 0.8)):
        lv = clamp(lvl * f)
        w.cyl((21, 0.18, z), 2.85, 6.6 * lv, (22, 120, 190), alpha=0.92, cap=(70, 190, 235), bias=0.5, bottom=False)
        w.cyl((21, 0.18, z), 3.0, 6.9, (110, 180, 220), alpha=0.20, edge=(70, 190, 235), cap=(120, 200, 235), bias=-0.4, bottom=False)
    # pipes
    for path, col, r in ((P_I, (60, 100, 130), 0.30), (P_R, (70, 110, 140), 0.26), (P_D, (70, 110, 140), 0.30)):
        w.tube(path.p, r, col)
    w.tube([(14, 6.55, 6.7), (14, 6.55, 6.6)], 0.2, (95, 125, 150))
    # source
    w.box((55, 1.4, 2), (6, 2.8, 5), (40, 58, 84))
    w.cyl((55, 2.8, 2), 2.2, 0.3, (60, 130, 180), edge=(90, 210, 245))
    # flowing water
    flow(w, P_I, 16, 3.5, (120, 225, 255))
    flow(w, P_R, 10, 3.0, (150, 220, 255))
    flow(w, P_G, 8, 3.2, (150, 220, 255), r=0.16)
    flow(w, P_D, 26, 3.6, (120, 225, 255))
    # pollutant entry + contaminated stream downstream of entry
    pe = np.array([X_POLL, 0.65, 8.4], np.float32)
    ph = (t * 1.6 + np.arange(5) / 5.0) % 1.0
    w.dots(np.stack([np.full(5, X_POLL), 5.0 - ph * 4.3, np.full(5, 8.4)], 1), 0.16, (255, 150, 70), bias=-0.4)
    w.box((X_POLL, 2.6, 8.4), (0.5, 4.0, 0.5), (90, 60, 40))
    pp = Path([(X_POLL, 0.65, 8.4), (-13, 0.65, 8.4)])
    flow(w, pp, 9, 3.2, (255, 130, 60), r=0.19, t0=0.4)
    # leakage
    xl = X_LEAK
    lk = (t * 1.9 + np.arange(6) / 6.0) % 1.0
    w.dots(np.stack([np.full(6, xl + 0.1), 0.55 - lk * 0.5, 8.4 + lk * 1.8], 1), 0.13, (255, 90, 80), bias=-0.4)
    w.quad([(xl - 0.9, 0.2, 9.4), (xl + 0.9, 0.2, 9.4), (xl + 1.2, 0.2, 10.5), (xl - 1.2, 0.2, 10.5)], (150, 40, 50), N=(0, 1, 0),
           alpha=0.7, edge=(255, 90, 80), bias=40)
    # taps (demand)
    for xt in X_TAPS:
        w.tube([(xt, 0.65, 8.4), (xt, 2.0, 8.4), (xt - 0.9, 2.0, 8.4)], 0.17, (100, 130, 155))
        dd = (t * 1.5 + np.arange(4) / 4.0 + xt) % 1.0
        w.dots(np.stack([np.full(4, xt - 0.9), 1.9 - dd * 1.7, np.full(4, 8.4)], 1), 0.1, (150, 225, 255), bias=-0.4)
    # sensors, valve, treatment
    for xs, c in ((X_UP, CYAN), (X_DN, CYAN)):
        w.tube([(xs, 0.65, 8.4), (xs, 1.8, 8.4)], 0.07, (120, 160, 190))
        w.sphere((xs, 2.1, 8.4), 0.42, c, glow=True)
    w.sphere((21, 7.55, 2), 0.42, CYAN, glow=True)
    w.box((X_VALVE, 1.0, 8.4), (1.5, 1.6, 1.5), (110, 80, 30), edge=(255, 190, 70))
    w.cyl((X_VALVE, 0.2, 11.2), 1.3, 2.8, (60, 90, 120), edge=(90, 210, 245))
    w.tube([(X_VALVE, 0.65, 8.4), (X_VALVE, 0.65, 11.0)], 0.16, (90, 120, 150))
    # labels
    ew = lambda a, b: smooth(seg(t, a, a + 0.5)) * (1 - smooth(seg(t, b, b + 0.7)))
    w.label((38, 1.2, 2), "I(t)  Inflow", ("from supply source",), CYAN, 40, -120, ew(10.4, 15.5))
    w.label((0, 6.7, 6.7), "R(t)  Rainwater harvest", ("roof catchment → gutter → tank",), ICE, -30, -150, ew(11.6, 15.5))
    w.label((X_TAPS[1], 2.0, 8.4), "D(t)  Demand", ("taps and outlets",), WHITE, -40, -150, ew(12.8, 15.5))
    w.label((X_LEAK, 0.4, 8.6), "L(t)  Leakage", ("loss from the pipeline",), RED, 10, 130, ew(13.7, 15.5))
    w.label((X_POLL, 4.0, 8.4), "Pollutant entry point", ("synthetic contamination event",), AMBER, 40, -190, ew(15.8, 22.2))
    w.label((X_UP, 2.1, 8.4), "Upstream sensor", ("water-quality monitoring",), CYAN, 40, -60, ew(16.4, 22.2))
    w.label((X_DN, 2.1, 8.4), "Downstream sensor", ("water-quality monitoring",), CYAN, -20, -170, ew(17.2, 22.2))
    w.label((X_VALVE, 1.9, 8.4), "Isolation & treatment", ("intervention points",), AMBER, -20, 150, ew(18.4, 22.2))


def world_frame(t):
    cam = cam_at(t)
    rain = dict(center=(5, 0, 5), n=620, spread=55, top=38, speed=34, wind=(2.0, 0.5), alpha=0.55)
    img = render_world(cam, ENV, build, t, refl=0.28, bloom=0.85, thr=135, rain=rain)
    f = img
    # --- overlays
    a = smooth(seg(t, 0.6, 1.4)) * (1 - smooth(seg(t, 5.4, 6.2)))
    if a > 0:
        text(f, "SCENE 03", 96, 92, 20, CYAN, a, "lm", "semibold", spacing=5)
        text(f, "Physical Digital Twin", 96, 140, 56, WHITE, a, "lm", "light")
        text(f, "A time-stepped simulation of a school water system", 96, 190, 28, GREY, a, "lm", "regular")
    ea = smooth(seg(t, 8.6, 9.4)) * (1 - smooth(seg(t, 21.6, 22.4)))
    if ea > 0.01:
        glass(f, 360, 930, 1560, 1040, a=0.66 * ea, border=CYAN, ba=0.35 * ea, r=16)
        wins = {"I": (10.6, 11.9), "R": (11.9, 13.0), "D": (13.0, 13.9), "L": (13.9, 15.2)}

        def hl(k):
            a0, a1 = wins[k]
            return 0.45 + 0.55 * (smooth(seg(t, a0, a0 + .25)) * (1 - smooth(seg(t, a1, a1 + .3))))
        parts = [("S(t+1)", WHITE, 1), (" = S(t) + ", GREY, 1), ("I(t)", CYAN, hl("I")), (" + ", GREY, 1), ("R(t)", ICE, hl("R")),
                 (" − ", GREY, 1), ("D(t)", WHITE, hl("D")), (" − ", GREY, 1), ("L(t)", RED, hl("L"))]
        terms(f, parts, 960, 978, 44, ea, "c", sym=True, w="medium")
        text(f, "0 ≤ S ≤ Smax  ·  overflow explicitly accounted for", 960, 1016, 20, GREY, 0.85 * ea, "cm", "regular", sym=True)
    ta = smooth(seg(t, 1.0, 1.8)) * (1 - smooth(seg(t, 24.6, 25.4)))
    tag(f, "SYNTHETIC SIMULATION DATA", 1824, 84, AMBER, 19, ta * 0.95, "r")
    return f


def dash_frame(t):
    f = BD.render(t, strength=0.9)
    x0, y0, x1, y1 = DASH_RECT
    t_in = 24.2
    # chapter header
    ha = smooth(seg(t, 25.0, 25.8))
    text(f, "SIMULATION DASHBOARD", 96, 62, 20, CYAN, ha, "lm", "semibold", spacing=5)
    text(f, "Actual output from the supplied simulation run", 96, 92, 24, GREY, ha, "lm", "regular")
    # camera
    cam = OV
    scan = None
    cur = None
    if t < 28.3:
        zoom = lerp(OV[2] * 0.93, OV[2], smooth(seg(t, 24.4, 28.0)))
        cam = (OV[0], OV[1], zoom)
        scan = smooth(seg(t, 25.0, 27.6))
    else:
        for i, (k, st, *_r) in enumerate(TOUR):
            if t >= st:
                cur = i
        # glide between panel cameras
        def pc(i):
            r = PANELS[TOUR[i][0]]
            return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 3.8)
        c1 = pc(cur)
        if cur == 0:
            c0 = OV
        else:
            c0 = pc(cur - 1)
        u = smooth(seg(t, TOUR[cur][1], TOUR[cur][1] + 0.7))
        cam = tuple(lerp(a, b, u) for a, b in zip(c0, c1))
        scan = None
    tf, vr = show_shot(f, DASH, cam, DASH_RECT, glow=CYAN, glow_a=0.55, reflect=0.0, scan=scan)
    # spotlight on the current panel
    if cur is not None:
        r = PANELS[TOUR[cur][0]]
        a_, b_ = tf(r[0], r[1])
        c_, d_ = tf(r[2], r[3])
        spot(f, vr, [(a_, b_, c_, d_)], dim=lerp(1.0, 0.30, smooth(seg(t, TOUR[cur][1], TOUR[cur][1] + 0.6))))
        fx = FX()
        fx.rect(max(x0 + 2, a_), max(y0 + 2, b_), min(x1 - 2, c_), min(y1 - 2, d_), CYAN, 3, 0.9)
        f = fx.apply(f, 0.8)
        k, st, title, l1, l2 = TOUR[cur]
        ca = smooth(seg(t, st + 0.2, st + 0.7))
        glass(f, 1490, 300, 1860, 640, a=0.7, border=CYAN, ba=0.4, r=16)
        text(f, f"{cur + 1} / 6", 1522, 338, 19, CYAN, ca, "lm", "semibold", spacing=3)
        text(f, title, 1522, 392, 31, WHITE, ca, "lm", "semibold")
        text(f, "Reading from the plot", 1522, 440, 19, GREY, ca, "lm", "regular")
        text(f, l1, 1522, 490, 23, ICE, ca, "lm", "regular", sym=True)
        text(f, l2, 1522, 530, 23, ICE, ca, "lm", "regular", sym=True)
        text(f, "approximate values (≈) read from the image", 1522, 596, 17, GREY, 0.8 * ca, "lm", "regular", sym=True)
        for i in range(6):
            cx = 1522 + i * 36
            cv2.circle(f, (cx, 622), 7, CYAN if i == cur else (50, 70, 95), -1 if i == cur else 2, cv2.LINE_AA)
    else:
        ca = smooth(seg(t, 26.0, 26.8))
        glass(f, 1490, 300, 1860, 560, a=0.7, border=CYAN, ba=0.4, r=16)
        text(f, "Six synchronized views", 1522, 352, 29, WHITE, ca, "lm", "semibold")
        text(f, "Original pixels, unaltered.", 1522, 402, 22, ICE, ca, "lm", "regular")
        text(f, "Camera moves, highlights and", 1522, 452, 22, GREY, ca, "lm", "regular")
        text(f, "overlays are added on top.", 1522, 484, 22, GREY, ca, "lm", "regular")
    tag(f, "SYNTHETIC SIMULATION DATA", 1824, 84, AMBER, 19, ha * 0.95, "r")
    return f


TBL_RECT = (790, 170, 1850, 860)


def table_beat(f, t):
    """Close-up of the user's ORIGINAL data-table screenshot (pixels unaltered). No column meanings are asserted."""
    a = smooth(seg(t, 21.9, 22.7))
    if a < 0.01:
        return f
    base = (f.astype(np.float32) * (1 - 0.55 * a)).astype(np.uint8)
    cx, cy, z = [float(v) for v in keys(t, [(21.9, 391, 250, 1.22), (23.0, 391, 165, 1.42), (24.9, 391, 305, 1.42), (26.4, 391, 330, 1.45)])]
    tmp = base.copy()
    tf, vr = show_shot(tmp, TABLE, (cx, cy, z), TBL_RECT, glow=CYAN, glow_a=0.55 * a)
    fx = FX()
    x0, y0, x1, y1 = TBL_RECT
    for r, col, st, en in ((4, CYAN, 22.7, 24.0), (8, AMBER, 23.9, 25.4)):
        ha = smooth(seg(t, st, st + 0.5)) * (1 - smooth(seg(t, en, en + 0.5)))
        if ha > 0.01:
            y = 30 + 39.3 * (r - 1)
            ax, ay = tf(52, y - 19)
            bx, by = tf(768, y + 19)
            fx.rect(ax, ay, bx, by, col, 3, ha * a)
    fx.img[:y0] = 0
    fx.img[y1:] = 0
    fx.img[:, :x0] = 0
    fx.img[:, x1:] = 0
    tmp = fx.apply(tmp, 0.7)
    out = cv2.addWeighted(base, 1 - a, tmp, a, 0)
    # caption column
    glass(out, 72, 250, 740, 760, a=0.72 * a, border=CYAN, ba=0.45 * a, r=18)
    text(out, "RECORDED SIMULATION DATA", 102, 296, 19, CYAN, a, "lm", "semibold", spacing=4)
    text(out, "original screenshot · values unaltered", 102, 330, 18, GREY, a, "lm", "regular")
    for j, ln in enumerate(["Every visible row is labelled “Normal”.", "Numbers are shown exactly as recorded.", "Column headers are not visible here,", "so no column meaning is assumed."]):
        text(out, ln, 102, 392 + 40 * j, 23, WHITE if j < 2 else ICE, a * smooth(seg(t, 22.4 + 0.3 * j, 23.0 + 0.3 * j)), "lm", "regular", sym=True)
    ra = smooth(seg(t, 23.8, 24.5)) * a
    for k, (lab, col) in enumerate((("Recorded rows", CYAN), ("State estimate", ICE), ("Residual check", AMBER))):
        yy = 612 + 52 * k
        cv2.circle(out, (112, yy), 9, tuple(int(c * ra) for c in col), 2, cv2.LINE_AA)
        text(out, lab, 136, yy, 22, WHITE, ra, "lm", "medium")
        if k < 2:
            cv2.line(out, (112, yy + 12), (112, yy + 40), tuple(int(c * 0.6 * ra) for c in GREY), 1, cv2.LINE_AA)
    return out


def world_beat(t):
    return table_beat(world_frame(t), t)


def frame(t):
    if t < 24.2:
        f = world_beat(t)
    elif t < 26.4:
        m = smooth(seg(t, 24.2, 26.4))
        a = world_beat(min(t, 26.3))
        s = lerp(1.0, 1.35, m)
        M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
        a = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
        b = dash_frame(t)
        f = cv2.addWeighted(a, 1 - m, b, m, 0)
    else:
        f = dash_frame(t)
    return finish(f, t, DUR, fin=0.8, fout=0.5)


if __name__ == "__main__":
    scene_main("scene_03_digital_twin", DUR, frame)
