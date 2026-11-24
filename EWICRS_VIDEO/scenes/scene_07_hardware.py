"""SCENE 7 - Physical implementation pathway (40 s).
Tabletop prototype concept rendered in 3D. Hardware is PLANNED and UNTESTED; only the simulation is complete.
This is a design visualisation, not footage of a built device."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *
from engine3d import *

DUR = 40.0
ENV = Env(horizon=(22, 48, 88), zenith=(3, 8, 22), floor=(6, 11, 22), grid=(30, 130, 180), grid_size=4.0, fog=0.010,
          pool=(14, 62, 96), pool_r=30.0, pool_c=(0, 0))
STEPS = ["Physical\nsensor", "Calibration", "Python\nbackend", "Digital\ntwin", "Intelligence\nengine", "Approved\ncontrol", "Feedback"]
SENS = [("Ultrasonic level", (-11.0, 0.0), 2.0, 12.0), ("Turbidity", (-13.2, 0.0), 3.0, 10.8),
        ("pH", (-11.0, 0.0), 4.0, 10.8), ("Temperature", (-8.8, 0.0), 5.0, 10.8)]
ESP = (-2.0, 1.2, -9.0)
PUMP = (12.0, 1.2, -9.0)
C1, C2 = (-11.0, 0.0, 0.0), (15.0, 0.0, 0.0)
CAM_KEYS = [(0, -16, 26, -62, 0, 3, -2), (8, -10, 19, -44, -6, 4, -2), (11, -4, 13, -30, -10, 5, 0), (19, -4, 13, -30, -10, 5, 0), (24, 0, 22, -54, 0, 3, -4), (40, 12, 24, -58, 4, 3, -3)]


def cam_at(t):
    k = keys(t, CAM_KEYS, ease=smooth)
    return Cam(k[0:3], k[3:6], fov=48)


def build(w):
    t = w.t
    # bench
    w.box((2.0, -0.6, -3.0), (46, 1.2, 24), (30, 40, 58), edge=(56, 120, 160))
    # containers
    for c, lv in ((C1, 0.7), (C2, 0.35)):
        w.cyl(c, 3.2, 8.0, (60, 110, 160), alpha=0.30, edge=(90, 210, 245), cap=(110, 200, 240), bottom=False)
        w.cyl((c[0], 0.02, c[2]), 2.95, 8.0 * lv, (24, 120, 180), alpha=0.85, edge=(24, 120, 180), cap=(110, 200, 240), bottom=False)
    # tubing: container 1 -> pump -> container 2
    p = [(C1[0] + 3.2, 1.0, 0), (6.0, 1.0, 0), (6.0, 1.0, -9), (PUMP[0], 1.0, -9)]
    q = [(PUMP[0], 1.0, -9), (PUMP[0] + 3.0, 1.0, -9), (C2[0], 1.0, -3.4)]
    w.tube(p, 0.26, (70, 112, 142))
    w.tube(q, 0.26, (70, 112, 142))
    run = smooth(seg(t, 27.0, 28.0)) * (1 - smooth(seg(t, 38.5, 39.2)))
    if run > 0.01:
        u = (np.arange(14) / 14 + t * 0.2) % 1.0
        w.dots(Path(p).at(u), 0.2, (120, 225, 255), bias=-0.3)
        u = (np.arange(8) / 8 + t * 0.2) % 1.0
        w.dots(Path(q).at(u), 0.2, (120, 225, 255), bias=-0.3)
    # pump body
    w.box(PUMP, (3.2, 2.4, 3.2), (52, 72, 100), edge=(90, 210, 245))
    # ESP32 board
    w.box(ESP, (6.0, 0.4, 3.6), (22, 70, 58), edge=(70, 224, 186))
    w.box((ESP[0] - 0.4, ESP[1] + 0.35, ESP[2]), (2.2, 0.3, 2.2), (60, 66, 80), edge=(150, 160, 180))
    on_esp = 0.5 + 0.5 * math.sin(t * 6)
    w.sphere((ESP[0] + 2.4, ESP[1] + 0.5, ESP[2] - 1.2), 0.22, CYAN if on_esp > 0.3 else (20, 70, 90), glow=True)
    # sensors, appear with narration (8 s onward)
    for i, (name, (sx, sz), st, top) in enumerate(SENS):
        a = smooth(seg(t, 8.0 + 2.2 * i, 9.0 + 2.2 * i))
        if a < 0.02:
            continue
        col = tuple(int(c * (0.4 + 0.6 * a)) for c in (CYAN if i != 3 else TEAL_))
        if i == 0:
            w.cyl((sx, 8.6, 0), 1.0, 1.2, (60, 70, 90), edge=col)
            w.sphere((sx - 0.5, 8.9, 0.9), 0.30, col, glow=True)
            w.sphere((sx + 0.5, 8.9, 0.9), 0.30, col, glow=True)
        else:
            w.tube([(sx, 9.6, 0), (sx, 2.4, 0)], 0.16, col)
            w.sphere((sx, 2.2, 0), 0.34, col, glow=True)
        # signal cable to ESP32
        w.tube([(sx, 9.6 if i else 9.0, 0), (sx, 10.4, -2.0), (ESP[0] + (i - 1.5) * 1.2, ESP[1] + 0.4, ESP[2] + 1.8)], 0.06, tuple(int(c * 0.6) for c in col))
        pk = ((t * 0.5 + i * 0.25) % 1.0)
        pth = Path([(sx, 9.6 if i else 9.0, 0), (sx, 10.4, -2.0), (ESP[0] + (i - 1.5) * 1.2, ESP[1] + 0.4, ESP[2] + 1.8)])
        w.dots(pth.at(np.array([pk])), 0.16, WHITE, bias=-0.3)
    # optional conductivity: dashed outline
    oa = smooth(seg(t, 16.6, 17.4)) * (1 - smooth(seg(t, 18.6, 19.2)))
    if oa > 0.02:
        w.tube([(-12.0, 9.6, 1.6), (-12.0, 2.4, 1.6)], 0.1, tuple(int(c * oa * 0.7) for c in AMBER))
    ea = lambda a, b: smooth(seg(t, a, a + .6)) * (1 - smooth(seg(t, b, b + .6)))
    w.label((ESP[0], 1.8, ESP[2]), "ESP32", ("microcontroller",), TEAL_, -20, -130, ea(8.2, 19.0))
    w.label((SENS[0][1][0], 9.2, 0), "Ultrasonic", ("water level",), CYAN, -110, -60, ea(10.4, 17.0))
    w.label((SENS[1][1][0], 9.6, 0), "Turbidity", ("clarity",), CYAN, 40, -90, ea(12.6, 17.0))
    w.label((SENS[2][1][0], 9.6, 0), "pH", ("acidity",), CYAN, -60, -150, ea(14.6, 17.0))
    w.label((SENS[3][1][0], 9.6, 0), "Temperature", ("water temp",), CYAN, -230, -90, ea(16.6, 19.0))
    w.label((-12.0, 9.6, 1.6), "Conductivity", ("optional",), AMBER, -30, -30, ea(17.2, 19.4))
    w.label((PUMP[0], 2.8, PUMP[2]), "Low-voltage pump", ("approved commands only",), TEAL_, 30, -110, ea(27.0, 34.0))


TEAL_ = (70, 224, 186)


def pipeline(f, t):
    a = smooth(seg(t, 19.0, 19.8)) * (1 - smooth(seg(t, 34.0, 34.6)))
    if a < 0.01:
        return
    x0, x1, y0, y1 = 72, 1848, 820, 1020
    glass(f, x0, y0, x1, y1, a=0.7 * a, border=CYAN, ba=0.4 * a, r=18)
    text(f, "SENSOR-TO-SOFTWARE PIPELINE", x0 + 28, y0 + 30, 18, CYAN, a, "lm", "semibold", spacing=4)
    n = len(STEPS)
    xs = np.linspace(x0 + 120, x1 - 120, n)
    cy = y0 + 104
    head = clamp((t - 19.6) / 7.4) * (n - 1)
    fx = FX()
    for i in range(n - 1):
        fx.line((xs[i] + 56, cy), (xs[i + 1] - 56, cy), CYAN, 3, (0.25 + 0.7 * clamp(head - i)) * a)
    for i in range(n):
        on = clamp(head - i + 0.7)
        col = TEAL_ if i == 5 else CYAN
        fx.rect(xs[i] - 66, cy - 40, xs[i] + 66, cy + 40, col, 3, (0.3 + 0.7 * on) * a)
    if t > 19.6:
        pu = ((t - 19.6) * 0.14) % 1.0
        fx.circle((lerp(xs[0], xs[-1], pu), cy + 62), 5, WHITE, -1, 0.9 * a)
    f[:] = fx.apply(f, 0.8)
    for i in range(n):
        on = clamp(head - i + 0.7)
        lines = STEPS[i].split("\n")
        for j, ln in enumerate(lines):
            y = cy + (j - (len(lines) - 1) / 2) * 24
            text(f, ln, xs[i], y, 20, WHITE if on > 0.5 else GREY, a * (0.5 + 0.5 * on), "cm", "semibold")


def control_panel(f, t):
    a = smooth(seg(t, 27.8, 28.6)) * (1 - smooth(seg(t, 34.0, 34.6)))
    if a < 0.01:
        return
    glass(f, 1230, 130, 1850, 470, a=0.72 * a, border=TEAL_, ba=0.5 * a, r=18)
    text(f, "SAFETY BOUNDARY", 1260, 166, 18, TEAL_, a, "lm", "semibold", spacing=4)
    items = [("Approved commands only", 28.4), ("Manual override", 30.0), ("Emergency stop", 31.6)]
    for i, (s, st) in enumerate(items):
        sa = smooth(seg(t, st, st + 0.6)) * a
        y = 232 + i * 76
        cv2.circle(f, (1276, y), 14, tuple(int(c * sa) for c in (RED if i == 2 else TEAL_)), -1 if i == 2 else 2, cv2.LINE_AA)
        text(f, s, 1308, y, 27, WHITE, sa, "lm", "medium")
    text(f, "the intelligence engine recommends; it does not act alone", 1260, 440, 19, GREY, a * smooth(seg(t, 32.4, 33.2)), "lm", "regular")


def status(f, t):
    a = smooth(seg(t, 34.2, 35.0))
    if a < 0.01:
        return
    glass(f, 190, 300, 1730, 760, a=0.80 * a, border=AMBER, ba=0.55 * a, r=20)
    text(f, "IMPLEMENTATION STATUS", W / 2, 352, 20, AMBER, a, "cm", "semibold", spacing=6)
    cv2.line(f, (960, 400), (960, 700), tuple(int(c * 0.5 * a) for c in GREY), 1, cv2.LINE_AA)
    sa = smooth(seg(t, 35.0, 35.8)) * a
    text(f, "COMPLETED", 575, 440, 22, CYAN, sa, "cm", "semibold", spacing=5)
    for j, s in enumerate(["Python simulation", "Digital twin and scenarios", "Experiment 1 screenshots"]):
        text(f, s, 575, 520 + 54 * j, 28, WHITE, sa, "cm", "medium")
    pa = smooth(seg(t, 36.2, 37.0)) * a
    text(f, "PLANNED · NOT YET VALIDATED", 1345, 440, 22, AMBER, pa, "cm", "semibold", spacing=5)
    for j, s in enumerate(["ESP32 sensor prototype", "Sensor calibration", "Pump control loop"]):
        text(f, s, 1345, 520 + 54 * j, 28, WHITE, pa, "cm", "medium")
    text(f, "Hardware results will be reported only after testing.", W / 2, 720, 22, GREY, a * smooth(seg(t, 37.4, 38.2)), "cm", "regular")


def frame(t):
    cam = cam_at(t)
    f = render_world(cam, ENV, build, t, refl=0.30, bloom=0.85, thr=135)
    if t > 34.0:
        d = 0.55 * smooth(seg(t, 34.0, 35.0))
        f = (f.astype(np.float32) * (1 - d)).astype(np.uint8)
    ha = smooth(seg(t, 0.3, 1.0))
    text(f, "SCENE 07", 96, 62, 20, CYAN, ha, "lm", "semibold", spacing=5)
    text(f, "Physical Implementation Pathway", 96, 112, 48, WHITE, ha, "lm", "light")
    text(f, "Tabletop prototype · isolated · not connected to any school water system", 96, 160, 24, GREY, ha, "lm", "regular")
    tag(f, "CONCEPT VISUALIZATION · HARDWARE PLANNED, UNTESTED", 1824, 62, AMBER, 18, smooth(seg(t, 0.6, 1.4)) * 0.95, "r")
    pipeline(f, t)
    control_panel(f, t)
    status(f, t)
    return finish(f, t, DUR, fin=0.7, fout=0.6)


if __name__ == "__main__":
    scene_main("scene_07_hardware", DUR, frame)
