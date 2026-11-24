"""SCENE 5 - Experiment 2: pollution detection and competing hypotheses (40 s).
ILLUSTRATIVE simulation: no Experiment 2 results were supplied, so no metrics, probabilities or rankings are shown.
Sensor traces and the hypothesis/sensor patterns are conceptual schematics and are labelled as such."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *
from engine3d import *

DUR = 40.0
ENV = Env(horizon=(22, 48, 88), zenith=(3, 8, 22), floor=(6, 11, 22), grid=(30, 130, 180), grid_size=4.0, fog=0.011,
          pool=(14, 62, 96), pool_r=40.0, pool_c=(0, 0))
X_SRC, X_ENTRY, X_UP, X_VALVE, X_DN, X_END = -34.0, -24.0, -10.0, 7.0, 19.0, 34.0
Y = 0.8
MAIN = Path([(X_SRC + 3, Y, 0), (X_END, Y, 0)])
SPEED = 4.4
T_ENTRY, T_VALVE = 5.0, 32.4
CAM_KEYS = [(0.0, -44, 18, 30, -26, 1, 0), (6.0, -32, 14, 28, -12, 1, 0), (13.0, -10, 14, 26, 6, 1, 0),
            (21.0, 2, 13, 26, 8, 1, 0), (31.0, 6, 14, 24, 8, 1, -2), (40.0, 14, 14, 26, 12, 1, 0)]
HYPS = [("Upstream contamination", "both sensors rise; upstream first", (1, 1)),
        ("Local downstream contamination", "only the downstream sensor rises", (0, 1)),
        ("Runoff disturbance", "both rise, tied to rainfall", (1, 1)),
        ("Sensor failure", "one sensor deviates alone, implausibly", (0, 2)),
        ("Unresolved cause", "pattern fits no listed explanation", (0, 0))]


def front(t):
    return X_ENTRY + max(0.0, t - T_ENTRY) * SPEED


def arrival(x):
    return T_ENTRY + (x - X_ENTRY) / SPEED


def cam_at(t):
    k = keys(t, CAM_KEYS, ease=smooth)
    return Cam(k[0:3], k[3:6], fov=52)


def build(w):
    t = w.t
    # source tank and houses
    w.cyl((X_SRC, 0.1, 0), 3.6, 7.0, (60, 120, 170), alpha=0.9, edge=(90, 210, 245), cap=(110, 200, 240), bottom=False)
    for i, (x, z) in enumerate([(23, -7), (27, -7), (31, -7), (23, 7), (27, 7), (31, 7)]):
        h = 3.2 + (i % 3) * 0.8
        w.box((x, h / 2, z), (3.2, h, 3.2), (44, 60, 88), edge=(56, 120, 160))
        w.quad([(x - 0.8, 1.0, z + (1.62 if z < 0 else -1.62)), (x + 0.8, 1.0, z + (1.62 if z < 0 else -1.62)),
                (x + 0.8, 2.2, z + (1.62 if z < 0 else -1.62)), (x - 0.8, 2.2, z + (1.62 if z < 0 else -1.62))],
               (150, 205, 225), N=(0, 0, 1 if z < 0 else -1), emissive=True, bias=-0.4)
        w.tube([(x, Y, 0), (x, Y, z * 0.45)], 0.14, (70, 110, 140))
    closed = t > T_VALVE
    # pipes
    w.tube(MAIN.p, 0.4, (70, 112, 142))
    # treatment branch
    w.tube([(X_VALVE, Y, 0), (X_VALVE, Y, -9)], 0.3, (70, 112, 142))
    w.cyl((X_VALVE, 0.1, -11), 1.9, 3.4, (60, 90, 120), edge=(90, 210, 245))
    # clean flow
    n = 38
    u = (np.arange(n) / n + t * 3.4 / MAIN.L) % 1.0
    pts = MAIN.at(u)
    clean = pts[:, 0] < (front(t) - 1.0) if (t > T_ENTRY and t < T_VALVE) else (pts[:, 0] > -999)
    if t > T_ENTRY and t < T_VALVE:
        pts = pts[pts[:, 0] > front(t) + 0.8]
    elif closed:
        pts = pts[(pts[:, 0] < X_VALVE) | (t > 36.5)]
    w.dots(pts, 0.28, (120, 225, 255), bias=-0.3)
    # plume
    if T_ENTRY < t:
        head = min(front(t), X_END)
        if closed:
            tail_x = X_ENTRY + (t - T_VALVE) * SPEED * 0.0
            # inflow upstream of valve is diverted to the treatment branch; downstream plume is flushed out
            flush = clamp((t - T_VALVE) / 3.8)
            xs = np.linspace(X_VALVE + 0.6, X_END, 26)
            xs = xs[xs > X_VALVE + 0.6 + flush * (X_END - X_VALVE)]
            if len(xs):
                w.dots(np.stack([xs, np.full(len(xs), Y), np.zeros(len(xs))], 1), 0.34 * (1 - flush), (255, 140, 60), bias=-0.5)
            d = np.linspace(0, 1, 10)
            ph = (d + t * 0.6) % 1.0
            w.dots(np.stack([np.full(10, X_VALVE), np.full(10, Y), -ph * 9.0], 1), 0.3, (255, 150, 70), bias=-0.4)
        else:
            m = int(clamp((head - X_ENTRY) / 58.0) * 40) + 2
            xs = np.linspace(X_ENTRY, head, m)
            jit = 0.12 * np.sin(xs * 2.1 + t * 5)
            w.dots(np.stack([xs, Y + jit, np.zeros(m)], 1), 0.34, (255, 140, 60), bias=-0.5)
    # pollutant entry
    w.box((X_ENTRY, 3.0, -5.5), (1.2, 6.0, 1.2), (92, 62, 40), edge=(255, 150, 70))
    w.tube([(X_ENTRY, 0.8, -5.5), (X_ENTRY, 0.8, 0)], 0.2, (110, 80, 50))
    if T_ENTRY - 0.3 < t < T_VALVE:
        ph = (t * 1.5 + np.arange(5) / 5.0) % 1.0
        w.dots(np.stack([np.full(5, X_ENTRY), 5.5 - ph * 4.7, np.full(5, -5.5 + 5.5 * (ph > 0.9))], 1), 0.2, (255, 140, 60), bias=-0.4)
    # sensors
    def sens(x, tt):
        arr = arrival(x)
        alert = (t > arr) and not (closed and t > 36.0 and x >= X_DN - 1)
        col = AMBER if alert else CYAN
        if x >= X_DN - 1 and t > 36.0 and closed:
            col = CYAN
        w.tube([(x, Y, 0), (x, 2.4, 0)], 0.08, (120, 160, 190))
        w.sphere((x, 2.8, 0), 0.5 + (0.12 * math.sin(t * 8) if alert else 0), col, glow=True)
    sens(X_UP, t)
    sens(X_DN, t)
    # isolation valve
    vcol = (110, 80, 30) if not closed else (150, 40, 40)
    w.box((X_VALVE, 1.4, 0), (1.6, 1.8, 1.6), vcol, edge=(255, 190, 70) if not closed else (255, 90, 80))
    # labels
    ea = lambda a, b: smooth(seg(t, a, a + .6)) * (1 - smooth(seg(t, b, b + .6)))
    w.label((X_SRC, 7.2, 0), "Source tank", ("simulated supply",), CYAN, 30, -110, ea(0.8, 6.0))
    w.label((X_ENTRY, 6.2, -5.5), "Pollutant entry", ("simulated event · not a real contaminant",), AMBER, 40, -110, ea(3.8, 12.5))
    w.label((X_UP, 3.0, 0), "Upstream sensor", ("water-quality monitoring",), CYAN, -20, -120, ea(5.6, 21.0))
    w.label((X_DN, 3.0, 0), "Downstream sensor", ("water-quality monitoring",), CYAN, 30, -130, ea(8.0, 12.6))
    w.label((X_VALVE, 2.4, 0), "Isolation valve", ("closes the affected zone",), AMBER, -330, -130, ea(31.8, 39.0))
    w.label((X_VALVE, 4.0, -11), "Treatment unit", ("simulated intervention",), CYAN, -330, -60, ea(33.2, 39.0))


def traces(f, t):
    a = smooth(seg(t, 5.4, 6.2))
    if a < 0.01:
        return
    x0, y0, x1, y1 = 72, 690, 800, 1010
    glass(f, x0, y0, x1, y1, a=0.66 * a, border=CYAN, ba=0.4 * a, r=16)
    text(f, "SENSOR TRACES", x0 + 26, y0 + 30, 18, CYAN, a, "lm", "semibold", spacing=4)
    text(f, "illustrative · unitless · not experimental data", x1 - 24, y0 + 30, 16, GREY, a, "rm", "regular")
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.035, 400)
    fx = FX()
    for k, (name, xs_) in enumerate((("UPSTREAM", X_UP), ("DOWNSTREAM", X_DN))):
        gy0 = y0 + 62 + k * 128
        gh = 92
        gx0, gx1 = x0 + 34, x1 - 30
        text(f, name, gx0, gy0 - 6, 16, ICE, a, "lm", "semibold", spacing=3)
        cv2.line(f, (gx0, gy0 + gh), (gx1, gy0 + gh), (60, 80, 110), 1, cv2.LINE_AA)
        T = np.linspace(0, 40, 300)
        arr = arrival(xs_)
        ramp = 1 / (1 + np.exp(-(T - arr - 0.6) * 2.4))
        rec = 1.0
        if xs_ >= X_DN - 1:
            u_ = np.clip((T - 35.0) / 3.5, 0, 1)
            rec = 1 - 0.95 * (u_ * u_ * (3 - 2 * u_))
        v = 0.12 + 0.62 * ramp * rec + noise[:300]
        n = int(clamp((t - 5.0) / 35.0 * 40 / 40) * 300) if False else int(np.sum(T <= t))
        if n > 2:
            pts = np.stack([gx0 + (gx1 - gx0) * T[:n] / 40.0, gy0 + gh - gh * np.clip(v[:n], 0, 1)], 1)
            alert_i = np.where(v[:n] > 0.4)[0]
            fx.poly(pts, CYAN, 3, 0.9)
            if len(alert_i):
                fx.poly(pts[alert_i[0]:], AMBER, 3, 0.95)
            fx.circle(pts[-1], 5, WHITE, -1, 0.9)
    f[:] = fx.apply(f, 0.7)


def hyp_panel(f, t):
    a = smooth(seg(t, 12.9, 13.7)) * (1 - smooth(seg(t, 31.0, 31.8)))
    if a < 0.01:
        return
    x0, y0, x1, y1 = 1000, 110, 1860, 700
    glass(f, x0, y0, x1, y1, a=0.7 * a, border=AMBER, ba=0.45 * a, r=18)
    text(f, "FIVE COMPETING EXPLANATIONS", x0 + 30, y0 + 34, 19, AMBER, a, "lm", "semibold", spacing=4)
    text(f, "conceptual sensor patterns", x1 - 28, y0 + 34, 16, GREY, a, "rm", "regular")
    text(f, "UP", x1 - 148, y0 + 74, 15, GREY, a, "cm", "semibold", spacing=2)
    text(f, "DOWN", x1 - 72, y0 + 74, 15, GREY, a, "cm", "semibold", spacing=2)
    ev = clamp((t - 21.4) / 9.0)
    for i, (name, sub, pat) in enumerate(HYPS):
        st = 13.2 + 1.6 * i
        ra = smooth(seg(t, st, st + 0.6)) * a
        ry = y0 + 96 + i * 88
        glass(f, x0 + 18, ry, x1 - 18, ry + 78, a=0.35 * ra, border=AMBER, ba=0.25 * ra, r=10, bw=1)
        text(f, f"H{i + 1}", x0 + 40, ry + 24, 16, AMBER, ra, "lm", "semibold", spacing=2)
        text(f, name, x0 + 82, ry + 24, 23, WHITE, ra, "lm", "semibold")
        text(f, sub, x0 + 82, ry + 56, 17, GREY, ra, "lm", "regular")
        fx = FX()
        for s_i, cx in enumerate((x1 - 148, x1 - 72)):
            kind = pat[s_i] if i != 3 else (0 if s_i == 0 else 2)
            if i == 0:
                kind = 1
            if i == 1:
                kind = 0 if s_i == 0 else 1
            if i == 3:
                kind = 0 if s_i == 0 else 2
            xs = np.linspace(cx - 26, cx + 26, 24)
            if kind == 0:
                ys = np.full(24, ry + 52.0)
            elif kind == 1:
                ys = ry + 52 - 28 / (1 + np.exp(-(np.linspace(-3, 3, 24) - (s_i * 1.6 - 0.4)) * 2))
            else:
                ys = ry + 52 - 28 * np.abs(np.sin(np.linspace(0, 9, 24))) ** 3
            if i == 4:
                ys = ry + 52 - 7 * np.sin(np.linspace(0, 9, 24) + s_i)
            fx.poly(np.stack([xs, ys], 1), ICE, 2, ra)
        f[:] = fx.apply(f, 0.5)
    # evidence sweep
    if t > 21.4:
        ea = smooth(seg(t, 21.4, 22.2)) * a
        sy = y0 + 96 + ((t - 21.4) * 0.55 % 1.0) * 5 * 88
        cv2.rectangle(f, (x0 + 18, int(sy) - 3), (x1 - 18, int(sy) + 3), tuple(int(c * ea * 0.9) for c in CYAN), -1)
        text(f, "comparing each pattern with what the sensors observed", x0 + 30, y1 - 22, 18, ICE, ea, "lm", "regular")


def conclusion(f, t):
    a = smooth(seg(t, 27.0, 27.8)) * (1 - smooth(seg(t, 31.4, 32.0)))
    if a < 0.01:
        return
    glass(f, 1000, 720, 1860, 930, a=0.72 * a, border=AMBER, ba=0.5 * a, r=18)
    text(f, "THE SYSTEM CAN ALSO CONCLUDE", 1030, 756, 18, AMBER, a, "lm", "semibold", spacing=4)
    text(f, "“Insufficient evidence to distinguish hypotheses.”", 1030, 818, 30, WHITE, a, "lm", "medium")
    text(f, "Uncertainty is reported, not hidden.", 1030, 872, 22, GREY, a, "lm", "regular")


def intervention(f, t):
    a = smooth(seg(t, 32.2, 33.0)) * (1 - smooth(seg(t, 39.2, 39.9)))
    if a < 0.01:
        return
    glass(f, 1000, 110, 1860, 470, a=0.7 * a, border=CYAN, ba=0.45 * a, r=18)
    text(f, "SIMULATED INTERVENTION", 1030, 144, 19, CYAN, a, "lm", "semibold", spacing=4)
    steps = [("1", "Assess the affected region", 32.4), ("2", "Isolate the zone and divert to treatment", 33.8),
             ("3", "Flush, then verify water quality afterwards", 35.6)]
    for i, (k, s, st) in enumerate(steps):
        sa = smooth(seg(t, st, st + 0.6)) * a
        y = 204 + i * 78
        cv2.circle(f, (1048, y), 18, tuple(int(c * sa) for c in CYAN), 2, cv2.LINE_AA)
        text(f, k, 1048, y, 18, WHITE, sa, "cm", "semibold")
        text(f, s, 1086, y, 25, WHITE, sa, "lm", "medium")
    va = smooth(seg(t, 37.2, 38.0)) * a
    text(f, "Verification: downstream sensor returns to normal (illustrative)", 1030, 440, 19, ICE, va, "lm", "regular")


def frame(t):
    cam = cam_at(t)
    f = render_world(cam, ENV, build, t, refl=0.28, bloom=0.85, thr=135)
    ha = smooth(seg(t, 0.3, 1.0))
    text(f, "SCENE 05  ·  EXPERIMENT 2", 96, 62, 20, CYAN, ha, "lm", "semibold", spacing=5)
    text(f, "Pollution Detection", 96, 112, 48, WHITE, ha * (1 - smooth(seg(t, 5.0, 5.8))), "lm", "light")
    text(f, "A simulated pollutant enters the network", 96, 160, 26, GREY, ha * (1 - smooth(seg(t, 5.0, 5.8))), "lm", "regular")
    tg = smooth(seg(t, 0.6, 1.4))
    tag(f, "ILLUSTRATIVE SIMULATION · NOT EXPERIMENTAL RESULTS", 1824, 62, AMBER, 18, tg * 0.95, "r")
    traces(f, t)
    hyp_panel(f, t)
    conclusion(f, t)
    intervention(f, t)
    return finish(f, t, DUR, fin=0.7, fout=0.6)


if __name__ == "__main__":
    scene_main("scene_05_pollution", DUR, frame)
