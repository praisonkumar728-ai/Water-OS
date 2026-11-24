"""Shared 2D toolkit for the EWICRS film: palette, easing, typography, glass panels,
bloom, film finish, screenshot camera and the per-scene ffmpeg writer.
All frames are RGB uint8 arrays, 1920x1080, 30 fps.
"""
import os, sys, math, time, subprocess, functools, argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1920, 1080, 30

# ---------------------------------------------------------------- palette (RGB)
MIDNIGHT = (7, 14, 34)
NAVY = (11, 24, 52)
GRAPHITE = (14, 17, 24)
CYAN = (40, 214, 255)
CYAN_D = (12, 96, 140)
ICE = (176, 238, 255)
WHITE = (240, 246, 255)
GREY = (138, 152, 172)
AMBER = (255, 178, 44)
RED = (255, 72, 72)


# ---------------------------------------------------------------- math helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp(x)
    return x ** 3


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


def seg(t, a, b):
    """progress of t through [a, b], clamped 0..1"""
    return clamp((t - a) / (b - a)) if b > a else float(t >= a)


def lerp(a, b, u):
    return a + (b - a) * u


def mix(c1, c2, u):
    return tuple(lerp(a, b, u) for a, b in zip(c1, c2))


def keys(t, kf, ease=smooth):
    """kf = [(time, v0, v1, ...), ...] -> tuple of eased interpolated values"""
    if t <= kf[0][0]:
        return tuple(kf[0][1:])
    for a, b in zip(kf[:-1], kf[1:]):
        if t <= b[0]:
            u = ease((t - a[0]) / (b[0] - a[0]))
            return tuple(lerp(x, y, u) for x, y in zip(a[1:], b[1:]))
    return tuple(kf[-1][1:])


def pulse(t, period=1.0, lo=0.5, hi=1.0):
    return lo + (hi - lo) * (0.5 + 0.5 * math.sin(2 * math.pi * t / period))


# ---------------------------------------------------------------- typography
@functools.lru_cache(None)
def _fp(spec):
    return subprocess.check_output(["fc-match", "-f", "%{file}", spec]).decode()


@functools.lru_cache(None)
def _font(path, size):
    return ImageFont.truetype(path, size)


_WEIGHT = {"light": "Fira Sans:style=Light", "regular": "Fira Sans:style=Regular", "medium": "Fira Sans:style=Medium",
           "semibold": "Fira Sans:style=SemiBold", "bold": "Fira Sans:style=Bold"}


def font(size, w="medium", sym=False, mono=False):
    if sym:
        spec = "DejaVu Sans:bold" if w in ("bold", "semibold") else "DejaVu Sans"
    elif mono:
        spec = "DejaVu Sans Mono"
    else:
        spec = _WEIGHT[w]
    return _font(_fp(spec), size)


@functools.lru_cache(maxsize=4096)
def sprite(s, size, color, w="medium", spacing=0, glow=0, sym=False, mono=False):
    f = font(size, w, sym, mono)
    if spacing:
        width = sum(f.getlength(ch) + spacing for ch in s)
    else:
        width = f.getlength(s)
    asc, desc = f.getmetrics()
    pad = int(glow * 2.6) + 4
    im = Image.new("RGBA", (int(width) + 2 * pad + 2, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if spacing:
        x = pad
        for ch in s:
            d.text((x, pad), ch, font=f, fill=tuple(color) + (255,))
            x += f.getlength(ch) + spacing
    else:
        d.text((pad, pad), s, font=f, fill=tuple(color) + (255,))
    if glow:
        a = np.array(im)
        al = cv2.GaussianBlur(a[..., 3], (0, 0), glow)
        g = np.zeros_like(a)
        g[..., :3] = color
        g[..., 3] = np.clip(al.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
        im = Image.alpha_composite(Image.fromarray(g), im)
    return np.array(im), pad, int(width)


def blit(frame, rgba, x, y, a=1.0, add=False):
    fh, fw = frame.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0 or a <= 0:
        return
    sp = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]
    al = sp[..., 3:4].astype(np.float32) * (a / 255.0)
    if add:
        roi[:] = np.clip(roi + sp[..., :3] * al, 0, 255).astype(np.uint8)
    else:
        roi[:] = (roi * (1 - al) + sp[..., :3] * al).astype(np.uint8)


def text(frame, s, x, y, size=36, color=WHITE, a=1.0, anchor="lm", w="medium", spacing=0, glow=0, sym=False, mono=False):
    """anchor: first letter l/c/r (horizontal), second t/m/b (vertical) relative to (x, y)."""
    if a <= 0.003 or not s:
        return 0
    arr, pad, width = sprite(s, int(size), tuple(color), w, spacing, glow, sym, mono)
    sh = arr.shape[0] - 2 * pad
    ax, ay = anchor[0], anchor[1]
    X = x - pad - (0 if ax == "l" else width / 2 if ax == "c" else width)
    Y = y - pad - (0 if ay == "t" else sh / 2 if ay == "m" else sh)
    blit(frame, arr, int(X), int(Y), a)
    return width


def text_width(s, size, w="medium", spacing=0, sym=False, mono=False):
    f = font(int(size), w, sym, mono)
    return sum(f.getlength(ch) + spacing for ch in s) if spacing else f.getlength(s)


def terms(frame, parts, x, y, size, a=1.0, anchor="l", sym=True, w="medium"):
    """draw a row of (text, color, alpha) spans, left/center aligned, returns total width"""
    widths = [text_width(p[0], size, w, sym=sym) for p in parts]
    total = sum(widths)
    cx = x - (total / 2 if anchor == "c" else 0)
    for (s, col, al), wd in zip(parts, widths):
        text(frame, s, cx, y, size, col, a * al, "lm", w, sym=sym)
        cx += wd
    return total


def reveal_text(s, p):
    """typewriter: first fraction p of the string"""
    n = int(len(s) * clamp(p) + 1e-6)
    return s[:n]


# ---------------------------------------------------------------- shapes / panels
@functools.lru_cache(maxsize=256)
def _rr(w, h, r):
    r = max(1, min(r, w // 2, h // 2))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return (m.astype(np.float32) / 255)[..., None]


def glass(frame, x0, y0, x1, y1, a=0.62, fill=(8, 16, 34), border=CYAN, ba=0.4, r=14, bw=2, blur=0):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    fh, fw = frame.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
    if cx1 <= cx0 or cy1 <= cy0 or a <= 0:
        return
    w, h = x1 - x0, y1 - y0
    m = _rr(w, h, r)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    roi = frame[cy0:cy1, cx0:cx1]
    base = roi
    if blur:
        base = cv2.GaussianBlur(roi, (0, 0), blur)
    col = np.array(fill, np.float32)
    out = base.astype(np.float32) * (1 - a) + col * a
    if border is not None and ba > 0 and bw > 0 and w > 2 * bw + 4 and h > 2 * bw + 4:
        inner = np.zeros((h, w, 1), np.float32)
        inner[bw:h - bw, bw:w - bw] = _rr(w - 2 * bw, h - 2 * bw, max(1, r - bw))
        ring = (_rr(w, h, r) - inner).clip(0, 1)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        out = out * (1 - ring * ba) + np.array(border, np.float32) * ring * ba
    roi[:] = (roi * (1 - m) + out * m).astype(np.uint8)


def label_box(frame, x, y, title, lines=(), color=CYAN, a=1.0, w=None, tsize=26, bsize=21, anchor="l"):
    """glass callout with a coloured accent bar. returns (x0,y0,x1,y1)"""
    if a <= 0.01:
        return None
    lines = list(lines)
    wt = text_width(title, tsize, "semibold")
    wl = max([text_width(l, bsize) for l in lines] + [0])
    bw = int(w or max(wt, wl) + 44)
    bh = int(22 + tsize * 1.25 + (len(lines) * bsize * 1.4 + 6 if lines else 0))
    x0 = x - (bw if anchor == "r" else bw / 2 if anchor == "c" else 0)
    x0, y0 = int(x0), int(y - bh / 2)
    fa = a
    tmp = frame
    glass(tmp, x0, y0, x0 + bw, y0 + bh, a=0.70 * fa, border=color, ba=0.45 * fa, r=10, bw=2, blur=0)
    cv2.rectangle(tmp, (x0 + 1, y0 + 12), (x0 + 5, y0 + bh - 12), tuple(int(c * (0.35 + 0.65 * fa)) for c in color), -1)
    text(frame, title, x0 + 22, y0 + 14 + tsize * 0.62, tsize, color if color != CYAN else ICE, fa, "lm", "semibold")
    for i, l in enumerate(lines):
        text(frame, l, x0 + 22, y0 + 18 + tsize * 1.25 + (i + 0.5) * bsize * 1.4, bsize, WHITE, 0.88 * fa, "lm", "regular")
    return x0, y0, x0 + bw, y0 + bh


def tag(frame, s, x, y, color=AMBER, size=20, a=1.0, anchor="l"):
    """small outlined caps tag, e.g. SYNTHETIC DATA"""
    wd = text_width(s, size, "semibold", spacing=2) + 28
    x0 = x - (wd if anchor == "r" else wd / 2 if anchor == "c" else 0)
    glass(frame, x0, y - size * 0.95, x0 + wd, y + size * 0.95, a=0.55 * a, border=color, ba=0.8 * a, r=8, bw=2)
    text(frame, s, x0 + 14, y, size, color, a, "lm", "semibold", spacing=2)


class FX:
    """additive layer for glowing 2D graphics; call apply() to bloom it onto a frame"""

    def __init__(self, shape=(H, W)):
        self.img = np.zeros((shape[0], shape[1], 3), np.uint8)

    def line(self, p1, p2, color, w=2, a=1.0):
        cv2.line(self.img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), tuple(int(c * a) for c in color), w, cv2.LINE_AA)

    def poly(self, pts, color, w=2, a=1.0, closed=False):
        cv2.polylines(self.img, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], closed, tuple(int(c * a) for c in color), w, cv2.LINE_AA)

    def circle(self, c, r, color, w=-1, a=1.0):
        cv2.circle(self.img, (int(c[0]), int(c[1])), int(r), tuple(int(v * a) for v in color), w, cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, color, w=2, a=1.0):
        cv2.rectangle(self.img, (int(x0), int(y0)), (int(x1), int(y1)), tuple(int(c * a) for c in color), w, cv2.LINE_AA)

    def apply(self, frame, gain=1.0):
        return bloom_add(frame, self.img, gain)


def bloom_add(frame, fx, gain=1.0):
    fh, fw = fx.shape[:2]
    small = cv2.resize(fx, (fw // 4, fh // 4), interpolation=cv2.INTER_AREA)
    b = cv2.addWeighted(cv2.GaussianBlur(small, (0, 0), 2.2), 0.9, cv2.GaussianBlur(small, (0, 0), 7.5), 1.0, 0)
    up = cv2.resize(b, (fw, fh), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(frame, fx)
    return cv2.add(out, cv2.convertScaleAbs(up, alpha=gain))


def bloom_thresh(frame, thr=150, gain=0.8):
    """bloom from bright pixels of the frame itself (respects occlusion)"""
    fh, fw = frame.shape[:2]
    small = cv2.resize(frame, (fw // 4, fh // 4), interpolation=cv2.INTER_AREA)
    hi = cv2.subtract(small, (thr, thr, thr, 0))
    hi = cv2.convertScaleAbs(hi, alpha=2.2)
    b = cv2.addWeighted(cv2.GaussianBlur(hi, (0, 0), 3), 0.8, cv2.GaussianBlur(hi, (0, 0), 11), 1.0, 0)
    up = cv2.resize(b, (fw, fh), interpolation=cv2.INTER_LINEAR)
    return cv2.add(frame, cv2.convertScaleAbs(up, alpha=gain))


# ---------------------------------------------------------------- backdrop: gradient, light shafts, dust
class Backdrop:
    def __init__(self, seed=3, tint=(10, 40, 70), shafts=True, dust=True, top=(9, 20, 46), bot=(3, 6, 14), cy=0.42):
        rng = np.random.default_rng(seed)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        base = np.array(top, np.float32) * (1 - yy) + np.array(bot, np.float32) * yy
        base = np.broadcast_to(base, (H, W, 3)).copy()
        xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        yv = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        d2 = ((xx - 0.5) / 0.55) ** 2 + ((yv - cy) / 0.7) ** 2
        base += np.exp(-d2 * 2.4)[..., None] * np.array(tint, np.float32)
        self.base = np.clip(base, 0, 255).astype(np.uint8)
        self.use_shafts, self.use_dust = shafts, dust
        sw, sh = 480, 270
        ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
        s = xs * math.cos(0.45) + ys * math.sin(0.45)
        lay = np.zeros((sh, sw), np.float32)
        for c, wd, amp in [(130, 22, .8), (230, 12, .5), (330, 30, .7), (420, 14, .4), (520, 20, .6)]:
            lay += amp * np.exp(-((s - c) / wd) ** 2)
        lay *= np.clip(1.15 - ys / sh, 0, 1)
        self.shaft = cv2.GaussianBlur(lay, (0, 0), 3)
        self.n = 150
        self.p = rng.random((self.n, 2)).astype(np.float32)
        self.v = (rng.random((self.n, 2)) - 0.5).astype(np.float32) * np.array([0.02, 0.015], np.float32)
        self.z = (0.25 + 0.75 * rng.random(self.n)).astype(np.float32)
        self.ph = rng.random(self.n) * 6.28

    def render(self, t, strength=1.0, shafts=1.0, dust=1.0):
        f = self.base.copy()
        if self.use_shafts and shafts > 0:
            M = np.float32([[1, 0, 24 * math.sin(0.15 * t)], [0, 1, 0]])
            sh = cv2.warpAffine(self.shaft, M, (480, 270), borderMode=cv2.BORDER_REFLECT)
            al = (0.55 + 0.45 * math.sin(0.35 * t + 1.0)) * 0.16 * shafts * strength
            up = cv2.resize(sh, (W, H), interpolation=cv2.INTER_LINEAR)
            add = (up * al)[..., None] * np.array([70, 170, 220], np.float32) / 1.0
            f = np.clip(f + add, 0, 255).astype(np.uint8)
        if self.use_dust and dust > 0:
            lay = np.zeros((270, 480), np.float32)
            pos = (self.p + self.v * self.z[:, None] * t * 20) % 1.0
            for (px, py), z, ph in zip(pos, self.z, self.ph):
                br = (0.5 + 0.5 * math.sin(t * 0.9 + ph)) * z
                cv2.circle(lay, (int(px * 480), int(py * 270)), 1 if z < .7 else 2, float(br), -1, cv2.LINE_AA)
            lay = cv2.GaussianBlur(lay, (0, 0), 0.8)
            up = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
            f = np.clip(f + (up * 150 * dust)[..., None] * np.array([0.55, 0.85, 1.0], np.float32), 0, 255).astype(np.uint8)
        return f


# ---------------------------------------------------------------- film finish
_VIG = None
_NOISE = None


def finish(frame, t, dur, fin=0.6, fout=0.6, grain=2.6, vig=0.5, seed=0):
    """vignette + grain + fade in/out. Works on 1920x1080 RGB uint8."""
    global _VIG, _NOISE
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        v = 1 - vig * np.clip(r / 2, 0, 1) ** 1.2
        _VIG = (v * 256).astype(np.uint16)[..., None]
        rng = np.random.default_rng(7)
        _NOISE = [cv2.resize(rng.normal(0, grain, (H // 2, W // 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.int16)[..., None] for _ in range(5)]
    fade = min(smooth(t / fin) if fin > 0 else 1.0, smooth((dur - t) / fout) if fout > 0 else 1.0)
    f = (frame.astype(np.uint16) * _VIG) >> 8
    f = f.astype(np.int16) + _NOISE[int(t * FPS + seed) % 5]
    if fade < 0.999:
        f = (f * int(fade * 256)) >> 8
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- screenshots
class Shot:
    """Original screenshot pixels, upscaled once with Lanczos; view() gives a smooth virtual camera."""

    def __init__(self, name, crop=None, k=3):
        im = np.array(Image.open(ROOT / "assets/screenshots" / name).convert("RGB"))
        if crop:
            x0, y0, x1, y1 = crop
            im = im[y0:y1, x0:x1]
        self.src = im
        self.h, self.w = im.shape[:2]
        self.k = k
        big = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
        bl = cv2.GaussianBlur(big, (0, 0), 1.5)
        self.big = cv2.addWeighted(big, 1.3, bl, -0.3, 0)

    def view(self, cx, cy, zoom, ow, oh, bg=(6, 10, 18)):
        s = zoom / self.k
        M = np.float32([[s, 0, ow / 2 - s * cx * self.k], [0, s, oh / 2 - s * cy * self.k]])
        return cv2.warpAffine(self.big, M, (ow, oh), flags=cv2.INTER_CUBIC if s >= 1.0 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=bg)

    @staticmethod
    def to_out(x, y, cx, cy, zoom, ow, oh):
        return ow / 2 + (x - cx) * zoom, oh / 2 + (y - cy) * zoom


def show_shot(frame, shot, cam, rect, glow=CYAN, glow_a=0.5, shadow=True, reflect=0.0, scan=None, r=10):
    """Draw a camera view of `shot` into rect=(x0,y0,x1,y1) of frame, return (to_frame(x,y), view_rect).
    cam = (cx, cy, zoom).  scan: optional 0..1 -> only left part (by x) is shown with a scan line."""
    x0, y0, x1, y1 = [int(v) for v in rect]
    ow, oh = x1 - x0, y1 - y0
    cx, cy, zoom = cam
    v = shot.view(cx, cy, zoom, ow, oh)
    if shadow:
        glass(frame, x0 - 6, y0 - 6, x1 + 6, y1 + 6, a=0.0, border=glow, ba=glow_a * 0.55, r=r + 4, bw=3)
    m = _rr(ow, oh, r)
    roi = frame[y0:y1, x0:x1]
    if scan is not None:
        sx = int(ow * clamp(scan))
        mm = np.zeros((oh, ow, 1), np.float32)
        mm[:, :sx] = 1
        v = (v.astype(np.float32) * 0.0 + v * mm).astype(np.uint8)
        m = m * mm
        roi[:] = (roi * (1 - m) + v * m).astype(np.uint8)
        if 0 < sx < ow:
            cv2.line(roi, (sx, 0), (sx, oh - 1), tuple(int(c) for c in ICE), 2, cv2.LINE_AA)
            g = np.zeros((oh, 40, 3), np.float32)
            g[:] = np.array(CYAN, np.float32) * 0.32
            for i in range(40):
                g[:, 40 - 1 - i] *= (1 - i / 40) ** 2
            xa = max(0, sx - 40)
            seg_w = sx - xa
            roi[:, xa:sx] = np.clip(roi[:, xa:sx] + g[:, 40 - seg_w:], 0, 255).astype(np.uint8)
    else:
        roi[:] = (roi * (1 - m) + v * m).astype(np.uint8)
    if reflect > 0:
        rh = min(int(oh * 0.35), H - y1 - 2)
        if rh > 8:
            refl = cv2.flip(v[oh - rh:], 0).astype(np.float32)
            fall = np.linspace(reflect, 0, rh, dtype=np.float32)[:, None, None]
            rr = frame[y1 + 4:y1 + 4 + rh, x0:x1]
            rh2 = rr.shape[0]
            rr[:] = (rr * (1 - fall[:rh2]) + refl[:rh2] * fall[:rh2]).astype(np.uint8)
    if glow is not None:
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x1, y1), tuple(int(c * glow_a) for c in glow), 2, cv2.LINE_AA)
    return (lambda X, Y: (x0 + ow / 2 + (X - cx) * zoom, y0 + oh / 2 + (Y - cy) * zoom)), (x0, y0, x1, y1)


def spot(frame, view_rect, holes, dim=0.32, feather=16):
    """darken the view except rectangles `holes` (frame coordinates)"""
    x0, y0, x1, y1 = view_rect
    ow, oh = x1 - x0, y1 - y0
    m = np.zeros((oh, ow), np.float32)
    for (a, b, c, d) in holes:
        a, b, c, d = int(a - x0), int(b - y0), int(c - x0), int(d - y0)
        m[max(0, b):max(0, d), max(0, a):max(0, c)] = 1
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    m = dim + (1 - dim) * np.clip(m * 1.6, 0, 1)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * m[..., None]).astype(np.uint8)


# ---------------------------------------------------------------- scene runner
def _writer(tmp, w=W, h=H):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-f", "mp4", str(tmp)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def scene_main(name, dur, frame_fn):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default="", help="comma separated times (s) -> JPEGs in work/preview")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.preview:
        od = ROOT / "work" / "preview"
        od.mkdir(parents=True, exist_ok=True)
        for ts in a.preview.split(","):
            t = float(ts)
            t0 = time.time()
            fr = frame_fn(t)
            Image.fromarray(fr).save(od / f"{name}_{t:05.1f}.jpg", quality=92)
            print(f"preview {name} t={t}: {time.time() - t0:.2f}s", flush=True)
        return
    out = Path(a.out) if a.out else ROOT / "assets/generated_scenes" / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    n = int(round(dur * FPS))
    p = _writer(tmp)
    t0 = time.time()
    for i in range(n):
        fr = frame_fn(i / FPS)
        assert fr.shape == (H, W, 3) and fr.dtype == np.uint8, fr.shape
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
        if i % 150 == 0:
            print(f"{name} frame {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    p.stdin.close()
    rc = p.wait()
    if rc == 0:
        os.replace(tmp, out)
        print(f"{name} done: {out}  ({n} frames, {time.time() - t0:.0f}s)", flush=True)
    else:
        print(f"{name} FAILED rc={rc}", flush=True)
        sys.exit(1)
