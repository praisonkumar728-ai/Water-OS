"""SCENE 2 - Four-layer intelligence architecture (30 s): 3D layer stack + end-to-end pipeline."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *
from engine3d import *

DUR = 30.0
ENV = Env(horizon=(22, 48, 88), zenith=(3, 8, 22), floor=(6, 11, 22), grid=(30, 130, 180), grid_size=4.0, fog=0.012,
          pool=(14, 62, 96), pool_r=34.0, pool_c=(0, 0))
BD = Backdrop(seed=5)
TEAL = (70, 224, 186)
LAYERS = [  # name, accent colour, cue start, question, descriptor
    ("EPISTEMIC INTELLIGENCE", AMBER, 3.6, "What is known, and how reliably?", "uncertainty · sources · competing hypotheses"),
    ("SCIENTIFIC INTELLIGENCE", CYAN, 10.2, "Does the model match physical law?", "conservation · residuals · model revision"),
    ("PREDICTIVE INTELLIGENCE", ICE, 15.0, "What is likely to happen next?", "storage · demand · climate · pollution risk"),
    ("ADAPTIVE DECISION INTELLIGENCE", TEAL, 20.0, "What should be done?", "simulated alternatives · approved actions"),
]
STEPS = ["Environment", "Sensors", "Data\nvalidation", "State\nestimation", "Digital\ntwin", "Epistemic\nengine",
         "Scientific\nvalidation", "Prediction", "Decision", "Control", "Feedback"]
GAP = 4.6
SX, SZ = 24.0, 14.0


def active(t):
    a = -1
    for i, L in enumerate(LAYERS):
        if t >= L[2]:
            a = i
    return a


def cam_at(t):
    ang = lerp(-0.62, 0.30, smooth(seg(t, 0, 30)))
    r = lerp(52, 41, smooth(seg(t, 0, 26)))
    y = lerp(24, 16, smooth(seg(t, 0, 20)))
    return Cam((math.sin(ang) * r, y, -math.cos(ang) * r), (0, 7.6, 0), fov=44)


def node_grid(i):
    return [(-8 + 4 * a, i * GAP + 0.5, -3.6 + 3.6 * b) for a in range(5) for b in range(3)]


def build(w):
    t = w.t
    act = active(t)
    fb = smooth(seg(t, 25.4, 26.6))
    for i, (name, col, st, q, d) in enumerate(LAYERS):
        y = i * GAP
        on = 1.0 if (act == i or (act == 3 and t > 24.8)) else (0.55 if act > i else 0.35)
        if t < LAYERS[0][2]:
            on = 0.55 * smooth(seg(t, 0.5 + 0.4 * i, 1.6 + 0.4 * i))
        c = tuple(int(v * (0.30 + 0.45 * on)) for v in col)
        w.box((0, y, 0), (SX, 0.5, SZ), c, alpha=0.88, edge=tuple(int(v * (0.4 + 0.6 * on)) for v in col), bias=0)
        for k, p in enumerate(node_grid(i)):
            pulse_k = 0.5 + 0.5 * math.sin(t * 3.0 + k * 0.9 + i)
            w.sphere(p, 0.26 + 0.1 * on * pulse_k, col if on > 0.9 else tuple(int(v * 0.55) for v in col), glow=on > 0.9)
        pts = node_grid(i)
        for a in range(4):
            for b in range(3):
                if (a + b + i) % 2 == 0:
                    p0, p1 = pts[a * 3 + b], pts[(a + 1) * 3 + b]
                    w.tube([p0, p1], 0.05, tuple(int(v * 0.5) for v in col))
    # vertical data pathways with flowing particles
    for cx, cz in ((-10.5, -5.5), (10.5, -5.5), (-10.5, 5.5), (10.5, 5.5)):
        top = (len(LAYERS) - 1) * GAP + 0.3
        w.tube([(cx, 0.3, cz), (cx, top, cz)], 0.12, (70, 105, 135))
        n = 12
        u = (np.arange(n) / n + t * 0.16) % 1.0
        ys = 0.3 + u * (top - 0.3)
        w.dots(np.stack([np.full(n, cx), ys, np.full(n, cz)], 1), 0.15, CYAN, bias=-0.3)
        if fb > 0.01:
            u2 = (np.arange(n) / n - t * 0.22) % 1.0
            ys2 = 0.3 + u2 * (top - 0.3)
            w.dots(np.stack([np.full(n, cx + 0.6), ys2, np.full(n, cz)], 1), 0.13 * (0.4 + fb * 0.6), AMBER, bias=-0.3)
    for i, (name, col, st, q, d) in enumerate(LAYERS):
        a = smooth(seg(t, st, st + 0.6)) * (1.0 if (act == i) else 0.0)
        w.label((-SX / 2 + 0.6, i * GAP + 0.3, -SZ / 2 + 1.2), f"{i + 1}  {name.title()}", (q, d), col, -460, -20, a, anchor="r")


def pipeline(f, t):
    a0 = 5.0
    a = smooth(seg(t, a0, a0 + 1.0))
    fb = smooth(seg(t, 25.4, 26.6))
    if a <= 0.01:
        return
    y0, y1 = 892, 1040
    glass(f, 72, y0, 1848, y1, a=0.66 * a, border=CYAN, ba=0.35 * a, r=18)
    text(f, "END-TO-END PIPELINE", 100, y0 + 24, 17, CYAN, a, "lm", "semibold", spacing=4)
    n = len(STEPS)
    xs = np.linspace(150, 1770, n)
    cy = y0 + 84
    head = clamp((t - 6.0) / 17.0) * (n - 1)
    fx = FX()
    for i in range(n - 1):
        on = clamp(head - i)
        fx.line((xs[i] + 22, cy), (xs[i + 1] - 22, cy), CYAN, 2, 0.25 + 0.65 * on)
    for i in range(n):
        on = clamp(head - i + 0.8)
        col = AMBER if STEPS[i].startswith("Epistemic") else (RED if False else CYAN)
        fx.circle((xs[i], cy), 14, col, -1 if on > 0.6 else 2, 0.35 + 0.65 * on)
    # travelling pulse
    if t >= 6.0:
        pu = ((t - 6.0) * 0.28) % 1.0
        px = lerp(xs[0], xs[-1], pu)
        fx.circle((px, cy), 7, WHITE, -1, 0.95)
    # feedback loop arc back to the start
    if fb > 0.01:
        arc = np.array([[xs[-1], cy - 22]] + [[lerp(xs[-1], xs[0], u), cy - 22 - 22 * math.sin(math.pi * u)] for u in np.linspace(0, fb, 40)])
        fx.poly(arc, AMBER, 3, 0.95)
    f[:] = fx.apply(f, 0.8)
    for i in range(n):
        on = clamp(head - i + 0.8)
        for j, ln in enumerate(STEPS[i].split("\n")):
            text(f, ln, xs[i], cy + 38 + 21 * j, 18, WHITE if on > 0.6 else GREY, a * (0.55 + 0.45 * on), "cm", "medium")
    if fb > 0.01:
        text(f, "feedback · validation", (xs[0] + xs[-1]) / 2, y0 + 24, 17, AMBER, fb, "cm", "semibold", spacing=3)


def frame(t):
    cam = cam_at(t)
    f = render_world(cam, ENV, build, t, refl=0.30, bloom=0.85, thr=130)
    ha = smooth(seg(t, 0.3, 1.1)) * (1 - smooth(seg(t, 3.3, 4.0)))
    text(f, "SCENE 02", 96, 92, 20, CYAN, ha, "lm", "semibold", spacing=5)
    text(f, "Four-Layer Intelligence", 96, 144, 58, WHITE, ha, "lm", "light")
    text(f, "From sensing to decision, with knowledge that audits itself", 96, 196, 27, GREY, ha, "lm", "regular")
    # layer counter (right)
    act = active(t)
    if act >= 0 and t < 25.1:
        ca = smooth(seg(t, LAYERS[act][2], LAYERS[act][2] + 0.6))
        name, col, st, q, d = LAYERS[act]
        glass(f, 1400, 130, 1850, 330, a=0.62 * ca, border=col, ba=0.55 * ca, r=16)
        text(f, f"LAYER {act + 1} OF 4", 1428, 166, 18, col, ca, "lm", "semibold", spacing=4)
        text(f, name.split(" INTELLIGENCE")[0].title(), 1428, 214, 38, WHITE, ca, "lm", "semibold")
        text(f, q, 1428, 262, 21, ICE, ca, "lm", "regular")
        text(f, d, 1428, 296, 18, GREY, ca, "lm", "regular")
    if t > 25.2:
        fa = smooth(seg(t, 25.4, 26.4))
        glass(f, 1400, 130, 1850, 270, a=0.62 * fa, border=AMBER, ba=0.55 * fa, r=16)
        text(f, "FEEDBACK CLOSES THE LOOP", 1428, 168, 20, AMBER, fa, "lm", "semibold", spacing=3)
        text(f, "Post-action outcomes return to the", 1428, 214, 21, ICE, fa, "lm", "regular")
        text(f, "knowledge layers for validation.", 1428, 244, 21, ICE, fa, "lm", "regular")
    pipeline(f, t)
    return finish(f, t, DUR, fin=0.8, fout=0.7)


if __name__ == "__main__":
    scene_main("scene_02_architecture", DUR, frame)
