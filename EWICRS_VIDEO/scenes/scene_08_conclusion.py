"""SCENE 8 - Conclusion (14.9 s): wide pull-back over the water-infrastructure world, title, final messages, sign-off."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2
from core import *
from engine3d import *
import scene_01_problem as S1

DUR = 14.9
CAM_KEYS = [(0.0, -56, 24, -2, -2, 4, 36), (14.9, -30, 50, -78, 10, 4, 32)]


def cam_at(t):
    k = keys(t, CAM_KEYS, ease=smooth)
    return Cam(k[0:3], k[3:6], fov=56)


def world_build(w):
    # reuse scene 1 geometry in a calm state (no pollutant, no rain): fake time stays in 4-7 s window
    w.t = 4.0 + w.t * 0.2
    S1.build(w)


def frame(t):
    cam = cam_at(t)
    f = render_world(cam, S1.ENV, world_build, t, refl=0.0, bloom=0.85, thr=140, rain=None)
    d = 0.18 + 0.38 * smooth(seg(t, 11.0, 12.2))
    f = (f.astype(np.float32) * (1 - d)).astype(np.uint8)
    # title block
    ta = smooth(seg(t, 0.2, 1.4))
    text(f, "EWICRS", 96, 120, 74, WHITE, ta, "lm", "light", spacing=16, glow=8)
    text(f, "EPISTEMIC WATER INTELLIGENCE & CLIMATE RESILIENCE SYSTEM", 96, 184, 22, ICE, ta, "lm", "regular", spacing=5, sym=False)
    cv2.line(f, (96, 214), (int(96 + 560 * smooth(seg(t, 0.6, 2.0))), 214), tuple(int(c * 0.9) for c in CYAN), 2, cv2.LINE_AA)
    # message 1
    m1 = smooth(seg(t, 0.8, 1.8)) * (1 - smooth(seg(t, 5.6, 6.4)))
    for i, s in enumerate(["THE FUTURE OF INTELLIGENT WATER MANAGEMENT", "IS NOT JUST PREDICTING WHAT WILL HAPPEN."]):
        text(f, s, W / 2, 470 + 70 * i, 46, WHITE if i == 0 else ICE, m1, "cm", "light", spacing=3, glow=3)
    # message 2
    m2 = smooth(seg(t, 6.5, 7.6))
    lines = ["IT IS UNDERSTANDING HOW RELIABLE OUR KNOWLEDGE IS,", "TESTING IT AGAINST REALITY,", "AND ADAPTING WHEN REALITY PROVES US WRONG."]
    for i, s in enumerate(lines):
        la = smooth(seg(t, 6.5 + 1.1 * i, 7.5 + 1.1 * i))
        text(f, s, W / 2, 380 + 64 * i, 40, WHITE if i < 2 else AMBER, la * m2, "cm", "light", spacing=3, glow=3 if i == 2 else 0)
    # sign-off
    sg = ["UNDERSTAND REALITY.", "VALIDATE KNOWLEDGE.", "ADAPT WITH EVIDENCE."]
    for i, s in enumerate(sg):
        a = smooth(seg(t, 11.6 + 0.8 * i, 12.4 + 0.8 * i)) * (1 - smooth(seg(t, 14.4, 14.9)))
        text(f, s, W / 2, 660 + 66 * i, 44, CYAN if i == 0 else (ICE if i == 1 else WHITE), a, "cm", "medium", spacing=8, glow=5)
    return finish(f, t, DUR, fin=0.9, fout=1.0)


if __name__ == "__main__":
    scene_main("scene_08_conclusion", DUR, frame)
