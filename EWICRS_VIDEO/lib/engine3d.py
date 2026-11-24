"""Small software 3D renderer (numpy + OpenCV) used for the cinematic scenes.

World units are metres, y is up. Features: perspective camera with near-plane clipping,
per-pixel sky / glossy grid floor / fog, painter's-algorithm solids with Lambert lighting,
mirrored-geometry floor reflections, tubes, spheres, flowing particles, rain, terrain with
a water level, 3D-anchored labels and threshold bloom.
"""
import math
import numpy as np
import cv2
from core import *

NEAR = 0.3
LIGHT1 = np.array([-0.45, 0.80, 0.40], np.float32)
LIGHT1 /= np.linalg.norm(LIGHT1)
LIGHT2 = np.array([0.65, 0.20, -0.55], np.float32)
LIGHT2 /= np.linalg.norm(LIGHT2)
RIM = np.array([20, 70, 110], np.float32)


class Cam:
    def __init__(self, pos, target, fov=52, size=(W, H)):
        self.pos = np.array(pos, np.float32)
        f = np.array(target, np.float32) - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0, 1, 0], np.float32))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.W, self.H = size
        self.fp = (self.H / 2) / math.tan(math.radians(fov) / 2)

    def to_cam(self, P):
        d = np.asarray(P, np.float32) - self.pos
        return np.stack([d @ self.r, d @ self.u, d @ self.f], -1)

    def proj_cam(self, C):
        z = np.maximum(C[..., 2], 1e-3)
        return np.stack([self.W / 2 + self.fp * C[..., 0] / z, self.H / 2 - self.fp * C[..., 1] / z], -1)

    def project(self, P):
        C = self.to_cam(P)
        return self.proj_cam(C), C[..., 2]


class Env:
    def __init__(self, horizon=(24, 50, 90), zenith=(4, 9, 24), floor=(6, 11, 21), grid=(30, 130, 180), grid_size=4.0,
                 fog=0.010, pool=(14, 62, 96), pool_r=45.0, pool_c=(0, 0), sun=None, sun_col=(255, 205, 140),
                 sun_k=(70.0, 9.0), sun_a=(1.0, 0.35), floor_y=0.0, grid_a=0.8, floor_on=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]


def render_env(cam, env, t=0.0):
    w, h = cam.W // 2, cam.H // 2
    fp = cam.fp / 2
    xs = ((np.arange(w, dtype=np.float32) + 0.5) - w / 2) / fp
    ys = (h / 2 - (np.arange(h, dtype=np.float32) + 0.5)) / fp
    f, r, u = cam.f, cam.r, cam.u
    dx = f[0] + r[0] * xs[None, :] + u[0] * ys[:, None]
    dy = f[1] + r[1] * xs[None, :] + u[1] * ys[:, None]
    dz = f[2] + r[2] * xs[None, :] + u[2] * ys[:, None]
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    hor = np.array(env.horizon, np.float32)
    zen = np.array(env.zenith, np.float32)
    sh = np.clip(dy / n, 0, 1)
    sky = hor + (zen - hor) * (sh ** 0.5)[..., None]
    if env.sun is not None:
        s = np.array(env.sun, np.float32)
        s /= np.linalg.norm(s)
        ca = (dx * s[0] + dy * s[1] + dz * s[2]) / n
        g = env.sun_a[0] * np.exp((ca - 1) * env.sun_k[0]) + env.sun_a[1] * np.exp((ca - 1) * env.sun_k[1])
        sky = sky + g[..., None] * np.array(env.sun_col, np.float32)
    hg = np.exp(-np.abs(dy / n) / 0.06)
    sky = sky + hg[..., None] * hor * 0.35
    img = sky
    if env.floor_on:
        mask = dy < -1e-3
        tt = np.where(mask, (env.floor_y - cam.pos[1]) / np.minimum(dy, -1e-3), 1e4).astype(np.float32)
        tt = np.minimum(tt, 2500.0)
        hx = cam.pos[0] + tt * dx
        hz = cam.pos[2] + tt * dz
        gs = env.grid_size
        wl = 0.035 + 0.0016 * tt
        ax = np.abs((hx / gs + 0.5) % 1 - 0.5) * gs
        az = np.abs((hz / gs + 0.5) % 1 - 0.5) * gs
        gi = np.maximum(np.clip(1 - ax / wl, 0, 1), np.clip(1 - az / wl, 0, 1)) * 0.55
        g5 = gs * 5
        ax5 = np.abs((hx / g5 + 0.5) % 1 - 0.5) * g5
        az5 = np.abs((hz / g5 + 0.5) % 1 - 0.5) * g5
        gi += np.maximum(np.clip(1 - ax5 / (wl * 2.2), 0, 1), np.clip(1 - az5 / (wl * 2.2), 0, 1)) * 0.6
        gi *= np.exp(-tt * 0.010)
        gl = np.exp(-((hx - env.pool_c[0]) ** 2 + (hz - env.pool_c[1]) ** 2) / env.pool_r ** 2)
        col = (np.array(env.floor, np.float32) + gl[..., None] * np.array(env.pool, np.float32)
               + (gi * env.grid_a)[..., None] * np.array(env.grid, np.float32))
        fg = (1 - np.exp(-tt * env.fog))[..., None]
        col = col * (1 - fg) + hor * fg
        img = np.where(mask[..., None], col, sky)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(img, (cam.W, cam.H), interpolation=cv2.INTER_LINEAR)


def _clip_near(C, near=NEAR):
    out = []
    n = len(C)
    for i in range(n):
        a, b = C[i], C[(i + 1) % n]
        ia, ib = a[2] >= near, b[2] >= near
        if ia:
            out.append(a)
        if ia != ib:
            tt = (near - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * tt)
    return np.array(out, np.float32) if len(out) >= 3 else None


class Path:
    def __init__(self, pts):
        self.p = np.array(pts, np.float32)
        d = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.c = np.concatenate([[0], np.cumsum(d)]).astype(np.float32)
        self.L = float(self.c[-1])

    def at(self, u):
        u = np.clip(np.asarray(u, np.float32), 0, 1) * self.L
        i = np.clip(np.searchsorted(self.c, u, side="right") - 1, 0, len(self.p) - 2)
        f = (u - self.c[i]) / (self.c[i + 1] - self.c[i] + 1e-9)
        return self.p[i] + (self.p[i + 1] - self.p[i]) * f[..., None]


class World:
    def __init__(self, cam, env, t=0.0, mirror=False):
        self.cam, self.env, self.t, self.mirror = cam, env, t, mirror
        self.items = []
        self.base = []      # terrain draw callbacks (drawn first)
        self.labels = []
        self.rain_pts = None
        self.hor = np.array(env.horizon, np.float32)

    # -------------------------------------------------------------- helpers
    def _m(self, P):
        P = np.asarray(P, np.float32)
        return P * np.array([1, -1, 1], np.float32) if self.mirror else P

    def _lit(self, color, N, Pc, spec=0.0, emissive=False):
        col = np.array(color, np.float32)
        if not emissive:
            key = max(0.0, float(N @ LIGHT1))
            rim = max(0.0, float(N @ LIGHT2))
            col = col * (0.30 + 0.74 * key) + RIM * rim * 0.45 + spec * 90
        d = float(np.linalg.norm(Pc - self.cam.pos))
        f = 1 - math.exp(-d * self.env.fog)
        if emissive:
            f *= 0.55
        col = col * (1 - f) + self.hor * f
        return tuple(int(clamp(v, 0, 255)) for v in col)

    def _visible(self, Nm, Cm):
        return float(Nm @ (self.cam.pos - Cm)) > 0

    def _poly(self, Pm, col, depth, alpha=1.0, edge=None, ew=1):
        C = self.cam.to_cam(Pm)
        if (C[:, 2] < NEAR).any():
            C = _clip_near(C)
            if C is None:
                return
        xy = self.cam.proj_cam(C)
        if xy[:, 0].max() < -50 or xy[:, 0].min() > self.cam.W + 50 or xy[:, 1].max() < -50 or xy[:, 1].min() > self.cam.H + 50:
            return
        self.items.append((depth, 0, xy.astype(np.int32), col, alpha, edge, ew))

    # -------------------------------------------------------------- primitives
    def quad(self, P, color, N=None, alpha=1.0, edge=None, ew=1, bias=0.0, emissive=False, two_sided=True, spec=0.0):
        P = np.asarray(P, np.float32)
        if N is None:
            N = np.cross(P[1] - P[0], P[2] - P[0])
            N = N / (np.linalg.norm(N) + 1e-9)
        N = np.asarray(N, np.float32)
        Nm = N * np.array([1, -1, 1], np.float32) if self.mirror else N
        Pc = P.mean(0)
        Cm = self._m(Pc)
        if not two_sided and not self._visible(Nm, Cm):
            return
        n_use = N if self._visible(Nm, Cm) else -N
        col = self._lit(color, n_use, Pc, spec, emissive)
        depth = float(self.cam.to_cam(Cm)[2]) + bias
        self._poly(self._m(P), col, depth, alpha, edge, ew)

    def box(self, c, size, color, alpha=1.0, edge=None, emissive=False, bias=0.0):
        c = np.array(c, np.float32)
        hx, hy, hz = np.array(size, np.float32) / 2
        faces = [
            ((0, 1, 0), [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)]),
            ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
            ((0, 0, -1), [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)]),
            ((1, 0, 0), [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)]),
            ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
            ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
        ]
        dc = float(self.cam.to_cam(self._m(c))[2]) + bias
        for N, vs in faces:
            N = np.array(N, np.float32)
            P = c + np.array(vs, np.float32) * np.array([hx, hy, hz], np.float32)
            Nm = N * np.array([1, -1, 1], np.float32) if self.mirror else N
            Cm = self._m(c + N * np.array([hx, hy, hz], np.float32))
            if not self._visible(Nm, Cm):
                continue
            col = self._lit(color, N, c + N * np.array([hx, hy, hz], np.float32), 0, emissive)
            fd = float(np.linalg.norm(Cm - self.cam.pos))
            self._poly(self._m(P), col, dc - 0.001 * (30 - fd), alpha, edge, 1)

    def cyl(self, c, r, h, color, n=26, alpha=1.0, edge=None, cap=None, emissive=False, bias=0.0, bottom=True):
        c = np.array(c, np.float32)
        ang = np.linspace(0, 2 * math.pi, n + 1)
        dc = float(self.cam.to_cam(self._m(c + [0, h / 2, 0]))[2]) + bias
        capcol = cap or color
        for i in range(n):
            a0, a1 = ang[i], ang[i + 1]
            am = (a0 + a1) / 2
            N = np.array([math.cos(am), 0, math.sin(am)], np.float32)
            P = np.array([c + [r * math.cos(a0), 0, r * math.sin(a0)], c + [r * math.cos(a1), 0, r * math.sin(a1)],
                          c + [r * math.cos(a1), h, r * math.sin(a1)], c + [r * math.cos(a0), h, r * math.sin(a0)]], np.float32)
            Pc = c + N * r + [0, h / 2, 0]
            Nm = N * np.array([1, -1, 1], np.float32) if self.mirror else N
            if not self._visible(Nm, self._m(Pc)):
                continue
            col = self._lit(color, N, Pc, 0.0, emissive)
            self._poly(self._m(P), col, dc + 0.01 * float(np.dot(self._m(Pc) - self.cam.pos, self.cam.f)) * 0, alpha, None, 1)
        for yy, N, cc in ((h, np.array([0, 1, 0], np.float32), capcol), (0.0, np.array([0, -1, 0], np.float32), color)):
            if yy == 0.0 and not bottom:
                continue
            Pc = c + [0, yy, 0]
            Nm = N * np.array([1, -1, 1], np.float32) if self.mirror else N
            if not self._visible(Nm, self._m(Pc)):
                continue
            P = np.array([c + [r * math.cos(a), yy, r * math.sin(a)] for a in ang[:-1]], np.float32)
            col = self._lit(cc, N, Pc, 0.05, emissive)
            self._poly(self._m(P), col, dc - 0.05, alpha, edge, 2)

    def tube(self, pts, r, color, core=None, bias=0.0, alpha=1.0):
        """pipe drawn as screen-space thick lines (dark outline + lighter core)"""
        pts = np.asarray(pts, np.float32)
        for a, b in zip(pts[:-1], pts[1:]):
            Ca, Cb = self.cam.to_cam(self._m(a)), self.cam.to_cam(self._m(b))
            if Ca[2] < NEAR or Cb[2] < NEAR:
                continue
            pa, pb = self.cam.proj_cam(Ca), self.cam.proj_cam(Cb)
            z = (Ca[2] + Cb[2]) / 2
            th = max(1, int(2 * r * self.cam.fp / z))
            d = float(np.linalg.norm((a + b) / 2 - self.cam.pos))
            f = 1 - math.exp(-d * self.env.fog)
            dark = tuple(int(v * 0.55 * (1 - f) + h * f) for v, h in zip(color, self.hor))
            lite = tuple(int(min(255, v * 1.25) * (1 - f) + h * f) for v, h in zip(color, self.hor))
            if core is not None:
                lite = core
            self.items.append((float(z) + bias, 1, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), th, dark, lite, alpha))

    def sphere(self, c, r, color, glow=False, bias=0.0):
        Cc = self.cam.to_cam(self._m(c))
        if Cc[2] < NEAR:
            return
        p = self.cam.proj_cam(Cc)
        rad = max(1, int(r * self.cam.fp / Cc[2]))
        d = float(np.linalg.norm(np.asarray(c, np.float32) - self.cam.pos))
        f = 1 - math.exp(-d * self.env.fog) if not glow else 0.0
        col = tuple(int(v * (1 - f) + h * f) for v, h in zip(color, self.hor))
        self.items.append((float(Cc[2]) + bias, 2, (int(p[0]), int(p[1])), rad, col, glow))

    def dots(self, P, px, color, bias=0.0):
        P = np.asarray(P, np.float32).reshape(-1, 3)
        C = self.cam.to_cam(self._m(P))
        xy = self.cam.proj_cam(C)
        for (x, y), z in zip(xy, C[:, 2]):
            if z < NEAR:
                continue
            rad = max(1, int(px * self.cam.fp / z)) if px < 1 else int(px)
            self.items.append((float(z) + bias, 3, (int(x), int(y)), rad, color))

    def label(self, P, title, lines=(), color=CYAN, dx=60, dy=-70, a=1.0, anchor="l"):
        if a <= 0.01 or self.mirror:
            return
        Cc = self.cam.to_cam(np.asarray(P, np.float32))
        if Cc[2] < NEAR:
            return
        p = self.cam.proj_cam(Cc)
        self.labels.append((float(p[0]), float(p[1]), title, tuple(lines), color, dx, dy, a, anchor))

    # -------------------------------------------------------------- terrain with water level
    def terrain(self, X, Z, Y, colors, water=None, water_col=(10, 70, 120), t=0.0):
        """colors: (nz-1, nx-1, 3) base colours. water: y level or None."""
        nz, nx = Y.shape
        PX, PZ = np.meshgrid(X, Z)
        P = np.stack([PX, Y, PZ], -1).astype(np.float32)
        Pm = P * np.array([1, -1, 1], np.float32) if self.mirror else P
        C = self.cam.to_cam(Pm)
        XY = self.cam.proj_cam(C)
        Zc = C[..., 2]
        idx = [(slice(0, -1), slice(0, -1)), (slice(0, -1), slice(1, None)), (slice(1, None), slice(1, None)), (slice(1, None), slice(0, -1))]
        corners = np.stack([XY[a, b] for a, b in idx], 2)          # (nz-1,nx-1,4,2)
        zc = np.stack([Zc[a, b] for a, b in idx], 2)
        ys = np.stack([Y[a, b] for a, b in idx], 2)
        depth = zc.mean(2)
        ok = (zc.min(2) > NEAR)
        # normals for shading
        e1 = P[1:, :-1] - P[:-1, :-1]
        e2 = P[:-1, 1:] - P[:-1, :-1]
        N = np.cross(e1, e2)
        N /= (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9)
        N = np.where(N[..., 1:2] < 0, -N, N)
        key = np.clip(N @ LIGHT1, 0, 1)
        rim = np.clip(N @ LIGHT2, 0, 1)
        col = np.array(colors, np.float32) * (0.32 + 0.78 * key)[..., None] + RIM * rim[..., None] * 0.4
        Pc = P[:-1, :-1] + (P[1:, 1:] - P[:-1, :-1]) * 0.5
        dist = np.linalg.norm(Pc - self.cam.pos, axis=-1)
        fg = (1 - np.exp(-dist * self.env.fog))[..., None]
        col = col * (1 - fg) + self.hor * fg
        wmask = np.zeros(depth.shape, bool)
        if water is not None:
            wmask = ys.max(2) <= water + 1e-3
            wave = np.sin(Pc[..., 0] * 0.35 + t * 1.3) * np.cos(Pc[..., 2] * 0.3 - t * 1.0)
            view = self.cam.pos - Pc
            vn = view / (np.linalg.norm(view, axis=-1, keepdims=True) + 1e-9)
            fres = (1 - np.clip(vn[..., 1], 0, 1)) ** 2.2
            wc = np.array(water_col, np.float32) * (0.85 + 0.25 * wave[..., None])
            wc = wc * (1 - fres[..., None] * 0.65) + self.hor * 1.5 * fres[..., None] * 0.65
            spark = (np.clip(wave, 0.85, 1) - 0.85) / 0.15
            wc = wc + spark[..., None] * 70
            wc = wc * (1 - fg) + self.hor * fg
            col = np.where(wmask[..., None], wc, col)
        order = np.argsort(-depth, axis=None)
        flat = np.unravel_index(order, depth.shape)
        colu = np.clip(col, 0, 255).astype(np.uint8)
        corners_i = corners.astype(np.int32)
        wy = water

        def draw(img, corners_i=corners_i, colu=colu, ok=ok, flat=flat, wmask=wmask, water=water, P=P):
            Wd, Hd = self.cam.W, self.cam.H
            for j, i in zip(*flat):
                if not ok[j, i]:
                    continue
                pts = corners_i[j, i]
                if pts[:, 0].max() < 0 or pts[:, 0].min() > Wd or pts[:, 1].max() < 0 or pts[:, 1].min() > Hd:
                    continue
                if wmask[j, i] and water is not None:
                    # replace by flat water cell at level
                    Q = np.array([[P[j, i, 0], water, P[j, i, 2]], [P[j, i + 1, 0], water, P[j, i + 1, 2]],
                                  [P[j + 1, i + 1, 0], water, P[j + 1, i + 1, 2]], [P[j + 1, i, 0], water, P[j + 1, i, 2]]], np.float32)
                    Qm = Q * np.array([1, -1, 1], np.float32) if self.mirror else Q
                    Cq = self.cam.to_cam(Qm)
                    if Cq[:, 2].min() < NEAR:
                        continue
                    pts = self.cam.proj_cam(Cq).astype(np.int32)
                c = colu[j, i]
                cv2.fillConvexPoly(img, pts, (int(c[0]), int(c[1]), int(c[2])), cv2.LINE_AA)
        self.base.append(draw)

    # -------------------------------------------------------------- drawing
    def draw(self, img):
        for fn in self.base:
            fn(img)
        self.items.sort(key=lambda it: -it[0])
        H_, W_ = img.shape[:2]
        for it in self.items:
            k = it[1]
            if k == 0:
                _, _, pts, col, alpha, edge, ew = it
                if alpha >= 0.999:
                    cv2.fillConvexPoly(img, pts, col, cv2.LINE_AA)
                else:
                    x0, y0 = pts.min(0)
                    x1, y1 = pts.max(0)
                    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W_, x1 + 1), min(H_, y1 + 1)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    cv2.fillConvexPoly(m, pts - [x0, y0], 255, cv2.LINE_AA)
                    roi = img[y0:y1, x0:x1]
                    a = (m.astype(np.float32) * (alpha / 255.0))[..., None]
                    roi[:] = (roi * (1 - a) + np.array(col, np.float32) * a).astype(np.uint8)
                if edge is not None:
                    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, edge, ew, cv2.LINE_AA)
            elif k == 1:
                _, _, pa, pb, th, dark, lite, alpha = it
                if alpha >= 0.999:
                    cv2.line(img, pa, pb, dark, th, cv2.LINE_AA)
                    cv2.line(img, pa, pb, lite, max(1, int(th * 0.42)), cv2.LINE_AA)
                else:
                    ov = img.copy()
                    cv2.line(ov, pa, pb, dark, th, cv2.LINE_AA)
                    cv2.line(ov, pa, pb, lite, max(1, int(th * 0.42)), cv2.LINE_AA)
                    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
            elif k == 2:
                _, _, p, rad, col, glow = it
                cv2.circle(img, p, rad, col, -1, cv2.LINE_AA)
                if rad > 3:
                    hl = tuple(min(255, int(v * 1.5 + 40)) for v in col)
                    cv2.circle(img, (p[0] - rad // 3, p[1] - rad // 3), max(1, rad // 3), hl, -1, cv2.LINE_AA)
            elif k == 3:
                _, _, p, rad, col = it
                cv2.circle(img, p, rad, col, -1, cv2.LINE_AA)

    def draw_labels(self, img, tt=None):
        for (x, y, title, lines, color, dx, dy, a, anchor) in self.labels:
            ex, ey = x + dx, y + dy
            cv2.line(img, (int(x), int(y)), (int(ex), int(ey)), tuple(int(c * (0.4 + 0.6 * a)) for c in color), 2, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 11, color, 2, cv2.LINE_AA)
            label_box(img, ex, ey, title, lines, color, a, anchor=anchor)


def rain_streaks(img, cam, t, center, n=700, spread=60, top=45, speed=38, length=2.6, color=(140, 185, 230), wind=(2.5, 0), alpha=0.55, seed=5):
    rng = np.random.default_rng(seed)
    base = rng.random((n, 3)).astype(np.float32)
    x = center[0] + (base[:, 0] - 0.5) * 2 * spread
    z = center[2] + (base[:, 2] - 0.5) * 2 * spread
    y = (base[:, 1] * top - speed * t) % top
    P0 = np.stack([x + wind[0] * (y / top), y, z + wind[1] * (y / top)], 1)
    P1 = P0 + np.array([wind[0], 1.0, wind[1]], np.float32) * length
    C0, C1 = cam.to_cam(P0), cam.to_cam(P1)
    ok = (C0[:, 2] > NEAR) & (C1[:, 2] > NEAR)
    a, b = cam.proj_cam(C0), cam.proj_cam(C1)
    lay = np.zeros_like(img)
    for i in np.where(ok)[0]:
        z_ = C0[i, 2]
        br = clamp(1.4 - z_ / 90, 0.15, 1)
        cv2.line(lay, (int(a[i, 0]), int(a[i, 1])), (int(b[i, 0]), int(b[i, 1])), tuple(int(c * br) for c in color), 1, cv2.LINE_AA)
    return cv2.add(img, cv2.convertScaleAbs(lay, alpha=alpha))


def render_world(cam, env, build, t, refl=0.30, bloom=0.8, thr=140, rain=None, post=None):
    """build(world) is called once for the mirrored pass and once for the real pass."""
    img = render_env(cam, env, t)
    if refl > 0 and env.floor_on:
        wm = World(cam, env, t, mirror=True)
        build(wm)
        lay = np.zeros_like(img)
        wm.draw(lay)
        m = (lay.max(axis=2) > 0).astype(np.uint8) * 255
        lb = cv2.GaussianBlur(lay, (0, 0), 2.2)
        mb = cv2.GaussianBlur(m, (0, 0), 2.2)
        inv = 255 - cv2.convertScaleAbs(mb, alpha=refl)
        inv3 = cv2.merge([inv, inv, inv])
        img = cv2.add(cv2.multiply(img, inv3, scale=1 / 255.0), cv2.convertScaleAbs(lb, alpha=refl))
    w = World(cam, env, t)
    build(w)
    w.draw(img)
    if rain:
        img = rain_streaks(img, cam, t, **rain)
    if bloom > 0:
        img = bloom_thresh(img, thr, bloom)
    w.draw_labels(img)
    return img
