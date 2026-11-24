"""SCENE 4 - Experiment 1: model failure and revision (40 s).
Uses the user's ORIGINAL residual-analysis and dashboard screenshots (pixels unaltered).
No claim is made that the revision improved prediction: no revised-prediction result was supplied."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np, cv2
from core import *

DUR = 40.0
BD = Backdrop(seed=21, tint=(8, 34, 62))
RES = Shot("residual.png", crop=(0, 10, 712, 330))          # crop offset: y - 10
DASH = Shot("dashboard.png", crop=(1, 3, 707, 471))
AVP = (0, 313, 357, 468)                                     # actual-vs-predicted panel (crop coords)
RY = lambda y: y - 10
# measured features (original residual screenshot pixels)
X_DEP, X_MARK, X_PEAK_Y = 331, 375, 43
X_NEG0, X_NEG1, Y_NEG = 436, 607, 293
Y_ZERO = 199
HYP = [("Demand increase", "consumption changed"), ("Sensor error", "measurement is unreliable"),
       ("Unexpected loss", "water leaves the system"), ("Inflow error", "supply input is mis-specified")]
STAGES = ["Original\nmodel", "Unexpected\ndiscrepancy", "Competing\nhypotheses", "Evidence\nevaluation", "Justified\nrevision", "Re-validation"]


def card(f, x0, y0, x1, y1, a, col=CYAN, fill_a=0.62):
    glass(f, x0, y0, x1, y1, a=fill_a * a, border=col, ba=0.45 * a, r=16)


def header(f, t):
    a = smooth(seg(t, 0.2, 0.9))
    text(f, "SCENE 04  ·  EXPERIMENT 1", 96, 62, 20, CYAN, a, "lm", "semibold", spacing=5)
    text(f, "Scientific Model Failure and Revision", 96, 108, 44, WHITE, a, "lm", "light")
    ta = smooth(seg(t, 0.6, 1.4))
    tag(f, "SYNTHETIC SIMULATION DATA", 1824, 70, AMBER, 19, ta * 0.95, "r")


def question(f, t):
    a = smooth(seg(t, 0.8, 1.6)) * (1 - smooth(seg(t, 5.2, 5.8)))
    if a < 0.01:
        return
    text(f, "RESEARCH QUESTION", W / 2, 440, 20, CYAN, a, "cm", "semibold", spacing=6)
    text(f, "Does justified model revision", W / 2, 520, 62, WHITE, a, "cm", "light")
    text(f, "improve subsequent prediction?", W / 2, 592, 62, ICE, a, "cm", "light", glow=3)


def tank(fx, f, cx, base, w, h, level, col, a, leak=False, t=0.0):
    x0, x1 = cx - w / 2, cx + w / 2
    fx.rect(x0, base - h, x1, base, col, 3, a)
    top = base - h * level
    cv2.rectangle(f, (int(x0 + 4), int(top)), (int(x1 - 4), int(base - 3)), tuple(int(c * 0.35 * a) for c in col), -1)
    fx.line((x0 + 4, top), (x1 - 4, top), col, 3, a)


def hidden_loss(f, t):
    a = smooth(seg(t, 5.7, 6.5)) * (1 - smooth(seg(t, 11.4, 12.2)))
    if a < 0.01:
        return
    fx = FX()
    L = smooth(seg(t, 6.0, 9.5))
    base = 760
    for cx, name, col, leak in ((560, "PHYSICAL SIMULATOR", CYAN, True), (1360, "INFERENCE MODEL", AMBER, False)):
        card(f, cx - 330, 260, cx + 330, 880, a)
        text(f, name, cx, 306, 21, col, a, "cm", "semibold", spacing=5)
        lvl = 0.78 - 0.22 * L * (1 if leak else 0.0) - 0.05 * L
        tank(fx, f, cx, base, 220, 330, lvl, col, a)
        # inflow / demand arrows
        fx.line((cx - 200, 470), (cx - 126, 470), ICE, 3, a)
        fx.line((cx + 126, 640), (cx + 200, 640), ICE, 3, a)
        text(f, "inflow + rain", cx - 205, 440, 19, GREY, a, "rm", "regular")
        text(f, "demand", cx + 205, 610, 19, GREY, a, "lm", "regular")
        if leak:
            for k in range(6):
                u = (t * 0.9 + k / 6) % 1.0
                fx.circle((cx - 40 + 12 * u, base + 8 + 60 * u), 5 * (1 - 0.5 * u), RED, -1, a * (1 - u) * smooth(seg(t, 7.0, 8.0)))
            fx.line((cx - 40, base), (cx - 40, base + 40), RED, 3, a * smooth(seg(t, 7.0, 8.0)))
    f[:] = fx.apply(f, 0.9)
    text(f, "extra water loss: present", 560, 842, 22, RED, a * smooth(seg(t, 7.0, 8.0)), "cm", "semibold")
    text(f, "extra water loss: unknown to the model", 1360, 842, 22, AMBER, a * smooth(seg(t, 7.6, 8.6)), "cm", "semibold")
    text(f, "S(t+1) = S(t) + I(t) + R(t) − D(t) − L(t)", W / 2, 220, 26, GREY, a, "cm", "regular", sym=True)
    sa = smooth(seg(t, 9.8, 10.6))
    text(f, "Hidden by design — the model must discover the discrepancy from evidence.", W / 2, 940, 27, ICE, a * sa, "cm", "regular")


def equation(f, t):
    a = smooth(seg(t, 12.0, 12.8)) * (1 - smooth(seg(t, 15.2, 15.9)))
    if a < 0.01:
        return
    card(f, 300, 380, 1620, 700, a)
    text(f, "RESIDUAL", W / 2, 440, 20, CYAN, a, "cm", "semibold", spacing=6)
    parts = [("e(t)", WHITE, 1), (" = ", GREY, 1), ("Observed(t)", ICE, 1), (" − ", GREY, 1), ("Predicted(t)", AMBER, 1)]
    terms(f, parts, W / 2, 540, 66, a, "c", sym=True, w="medium")
    text(f, "A residual alone does not prove the model wrong.", W / 2, 640, 26, GREY, a * smooth(seg(t, 13.6, 14.4)), "cm", "regular")


RES_RECT = (210, 250, 1710, 910)
CAMK = [(15.0, 356, 160, 2.10), (16.2, 345, 160, 3.3), (17.9, 352, 150, 3.3), (18.6, 385, 115, 3.4), (20.2, 395, 120, 3.2),
        (20.8, 520, 195, 2.6), (21.9, 520, 195, 2.6), (22.7, 356, 160, 2.10)]
CUTK = [(15.3, 20), (16.9, 322), (17.7, 348), (18.8, 388), (20.2, 440), (21.0, 640), (21.5, 705)]


def residual_close(f, t):
    a = smooth(seg(t, 14.9, 15.6)) * (1 - smooth(seg(t, 22.4, 23.0)))
    if a < 0.01:
        return
    cx, cy, z = [float(v) for v in keys(t, [(k[0],) + k[1:] for k in CAMK])]
    cut = float(keys(t, CUTK, ease=smooth)[0])
    x0, y0, x1, y1 = RES_RECT
    ow = x1 - x0
    scan = clamp((ow / 2 + (cut - cx) * z) / ow) if t < 21.5 else None
    tmp = f.copy()
    tf, vr = show_shot(tmp, RES, (cx, cy, z), RES_RECT, glow=CYAN, glow_a=0.55, scan=scan)
    fx = FX()
    hi = lambda x, y: tf(x, RY(y))
    # highlights
    ha = smooth(seg(t, 16.3, 16.8)) * (1 - smooth(seg(t, 18.1, 18.5)))
    if ha > 0.01 and t > 16.0:
        px, py = hi(X_DEP, Y_ZERO)
        fx.circle((px, py), 30 + 4 * math.sin(t * 6), CYAN, 3, ha)
    hb = smooth(seg(t, 18.5, 19.0)) * (1 - smooth(seg(t, 20.0, 20.5)))
    if hb > 0.01:
        ax, ay = hi(X_MARK - 9, 32)
        bx, by = hi(X_MARK + 9, 305)
        fx.rect(ax, max(ay, y0 + 4), bx, min(by, y1 - 4), RED, 3, hb)
        px, py = hi(X_MARK, X_PEAK_Y)
        fx.circle((px, py), 22 + 3 * math.sin(t * 6), RED, 3, hb)
    hc = smooth(seg(t, 20.6, 21.1)) * (1 - smooth(seg(t, 22.0, 22.4)))
    if hc > 0.01:
        ax, ay = hi(X_NEG0, Y_NEG - 12)
        bx, by = hi(X_NEG1, Y_NEG + 12)
        fx.rect(ax, ay, bx, by, AMBER, 3, hc)
        zx, zy = hi(X_NEG0 + 40, Y_ZERO)
        fx.line((zx, zy), (zx, hi(0, Y_NEG)[1]), AMBER, 2, hc * 0.8)
    fx.img[:y0] = 0
    fx.img[y1:] = 0
    fx.img[:, :x0] = 0
    fx.img[:, x1:] = 0
    f[:] = fx.apply(tmp, 0.7)
    # callouts
    def call(txt, sub, col, a0, a1):
        ca = smooth(seg(t, a0, a0 + 0.5)) * (1 - smooth(seg(t, a1, a1 + 0.4)))
        if ca < 0.01:
            return
        card(f, 480, 150, 1440, 232, ca, col)
        text(f, txt, 520, 180, 29, WHITE, ca, "lm", "semibold", sym=True)
        text(f, sub, 520, 213, 20, GREY, ca, "lm", "regular", sym=True)
    call("Departs from zero near step 43", "the first visible discrepancy between observed and predicted", CYAN, 16.2, 18.1)
    call("Peaks near the leakage-event marker (t = 50)", "red dotted line = marker from the simulation run", RED, 18.4, 20.1)
    call("Reverses sign: ≈ −35 from t ≈ 60 to t ≈ 88", "values read approximately from the plot", AMBER, 20.6, 22.0)
    tg = smooth(seg(t, 15.6, 16.2)) * (1 - smooth(seg(t, 22.0, 22.6)))
    tag(f, "ORIGINAL SCREENSHOT · VALUES UNALTERED", 210, 960, CYAN, 19, tg * 0.9, "l")


def investigation(f, t):
    a = smooth(seg(t, 22.5, 23.4)) * (1 - smooth(seg(t, 33.0, 33.8)))
    if a < 0.01:
        return
    # left column: original screenshots, small
    tf, vr = show_shot(f, RES, (356, 160, 0.94), (72, 190, 744, 500), glow=CYAN, glow_a=0.5 * a)
    text(f, "RESIDUAL  ·  original screenshot", 72, 176, 18, CYAN, a, "lm", "semibold", spacing=3)
    tf2, vr2 = show_shot(f, DASH, ((AVP[0] + AVP[2]) / 2, (AVP[1] + AVP[3]) / 2, 1.88), (72, 560, 744, 852), glow=CYAN, glow_a=0.5 * a)
    text(f, "ACTUAL VS PREDICTED STORAGE  ·  original dashboard panel", 72, 546, 18, CYAN, a, "lm", "semibold", spacing=3)
    ob = smooth(seg(t, 28.6, 29.4))
    if ob > 0.01:
        fx = FX()
        ax, ay = tf2(165, 436)
        bx, by = tf2(232, 464)
        cv2.rectangle(f, (int(ax), int(ay)), (int(bx), int(by)), (214, 120, 0), 4, cv2.LINE_AA)
        card(f, 72, 872, 744, 988, ob * a, AMBER)
        text(f, "Cross-check worth testing", 92, 900, 21, AMBER, ob * a, "lm", "semibold")
        text(f, "Actual storage holds at 0 while the predicted", 92, 934, 19, ICE, ob * a, "lm", "regular")
        text(f, "curve continues below 0 — a bound-handling candidate.", 92, 960, 19, ICE, ob * a, "lm", "regular")
    # hypothesis cards (2 x 2)
    text(f, "COMPETING HYPOTHESES", 800, 176, 18, AMBER, a, "lm", "semibold", spacing=4)
    for i, (name, sub) in enumerate(HYP):
        st = 23.2 + 1.4 * i
        ca = smooth(seg(t, st, st + 0.6)) * a
        cx0 = 800 + (i % 2) * 540
        cy0 = 200 + (i // 2) * 250
        card(f, cx0, cy0, cx0 + 510, cy0 + 224, ca, AMBER if i != 2 else CYAN)
        text(f, f"H{i + 1}", cx0 + 28, cy0 + 40, 19, AMBER, ca, "lm", "semibold", spacing=3)
        text(f, name, cx0 + 28, cy0 + 86, 33, WHITE, ca, "lm", "semibold")
        text(f, sub, cx0 + 28, cy0 + 126, 21, GREY, ca, "lm", "regular")
        # evidence-evaluation sweep (no outcome is shown)
        by = cy0 + 176
        cv2.rectangle(f, (cx0 + 28, by), (cx0 + 482, by + 8), (40, 56, 82), -1)
        ev = smooth(seg(t, 26.6, 27.4))
        ph = (t * 0.55) % 1.0
        sx0 = int(cx0 + 28 + (454 - 130) * (1 - abs(2 * ph - 1)))
        cv2.rectangle(f, (sx0, by), (sx0 + 130, by + 8), tuple(int(c * ca * ev) for c in ICE), -1)
        text(f, "weighing evidence for and against", cx0 + 28, by + 32, 17, GREY, ca * 0.9, "lm", "regular")
    ba = smooth(seg(t, 29.8, 30.6)) * a
    card(f, 800, 710, 1850, 988, ba)
    text(f, "EVIDENCE EVALUATION", 830, 744, 18, CYAN, ba, "lm", "semibold", spacing=4)
    terms(f, [("P(H|E)", WHITE, 1), (" = ", GREY, 1), ("P(E|H) · P(H)", ICE, 1), (" / ", GREY, 1), ("P(E)", ICE, 1)], 830, 812, 44, ba, "l", sym=True)
    text(f, "Priors and likelihoods must be calibrated or labelled illustrative.", 830, 874, 22, GREY, ba, "lm", "regular")
    text(f, "Permitted conclusion:  “Insufficient evidence to distinguish hypotheses.”", 830, 920, 24, AMBER, ba * smooth(seg(t, 31.2, 32.0)), "lm", "medium")


def revision(f, t):
    a = smooth(seg(t, 33.4, 34.2))
    if a < 0.01:
        return
    n = len(STAGES)
    xs = np.linspace(190, 1730, n)
    cy = 470
    head = clamp((t - 34.0) / 4.8) * (n - 1)
    fx = FX()
    for i in range(n - 1):
        fx.line((xs[i] + 80, cy), (xs[i + 1] - 80, cy), CYAN, 3, (0.25 + 0.7 * clamp(head - i)) * a)
    for i in range(n):
        on = clamp(head - i + 0.7)
        col = AMBER if i in (1, 2) else CYAN
        fx.rect(xs[i] - 76, cy - 56, xs[i] + 76, cy + 56, col, 3, (0.3 + 0.7 * on) * a)
    f[:] = fx.apply(f, 0.8)
    for i in range(n):
        on = clamp(head - i + 0.7)
        for j, ln in enumerate(STAGES[i].split("\n")):
            text(f, ln, xs[i], cy - 12 + 30 * j, 24, WHITE if on > 0.5 else GREY, a * (0.5 + 0.5 * on), "cm", "semibold")
        text(f, str(i + 1), xs[i], cy - 78, 18, CYAN, a * on, "cm", "semibold", spacing=2)
    ba = smooth(seg(t, 37.0, 37.8))
    card(f, 300, 650, 1620, 900, ba, CYAN)
    text(f, "A revision is retained only if evidence supports it", W / 2, 706, 30, WHITE, ba, "cm", "medium")
    text(f, "and predictions on held-out observations agree.", W / 2, 750, 30, WHITE, ba, "cm", "medium")
    text(f, "Effect of a revision on prediction: judged on held-out data", W / 2, 826, 22, AMBER, ba, "cm", "regular", sym=True)
    text(f, "and not asserted in this film.", W / 2, 858, 22, AMBER, ba, "cm", "regular", sym=True)


def frame(t):
    f = BD.render(t, strength=0.9)
    header(f, t)
    question(f, t)
    hidden_loss(f, t)
    equation(f, t)
    residual_close(f, t)
    investigation(f, t)
    revision(f, t)
    return finish(f, t, DUR, fin=0.7, fout=0.6)


if __name__ == "__main__":
    scene_main("scene_04_model_revision", DUR, frame)
