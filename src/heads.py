"""Realistic heads built around a photograph.

A dense head mesh is sculpted (nose, brow, sockets, cheekbones, lips, chin) so
that its features line up with landmarks measured on a frontal photo; the photo
is then projected onto it from the front and baked into the head texture. Hair
is a second, larger shell whose texture and cut-out mask also come from the
photo (front) and from a painted strand texture (top, sides, back).
"""
import os

import numpy as np
from PIL import Image

from engine import Mesh, from_indexed, norm

HERE = os.path.dirname(os.path.abspath(__file__))
FACES = os.path.join(HERE, '..', 'assets', 'faces')


def _g(x, y, x0, y0, sx, sy):
    return np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2)))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class PhotoHead:
    """Head-local frame: origin between the eyes, x to his left, y up, z forward (metres)."""

    def __init__(self, photo, eyes_px, scale_px_per_m, nose_dx=0.0, mouth_dx=0.0,
                 face_ellipse=(-0.004, -0.042, 0.066, 0.092), hair_len=0.11, hair_volume=1.0,
                 hair_col=(0.10, 0.07, 0.05), seed=0, photo_yaw=0.0, nose_y=-0.037,
                 mouth_y=-0.072, chin_y=-0.117, jaw=0.140, cap=None):
        self.img = np.asarray(Image.open(os.path.join(FACES, photo)).convert('RGB'), np.float64) / 255
        self.ex, self.ey = eyes_px
        self.S = scale_px_per_m
        self.nose_dx, self.mouth_dx = nose_dx, mouth_dx
        self.face_ellipse = face_ellipse
        self.hair_len = hair_len
        self.hv = hair_volume
        self.hair_col = np.asarray(hair_col)
        self.rng = np.random.default_rng(seed)
        self.yaw = photo_yaw
        self.nose_y, self.mouth_y, self.chin_y, self.jaw = nose_y, mouth_y, chin_y, jaw
        self.cap = None if cap is None else np.asarray(cap)   # cap colour: the shell becomes a cap
        self.skin = self._sample_skin()

    # ------------------------------------------------------------ photo access
    def project(self, P):
        """Head-local point -> photo colour, turning the head the way it is turned in the photo."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x = P[..., 0] * c + (P[..., 2] + 0.04) * s
        return self.photo(x, P[..., 1])

    def photo(self, x, y):
        """Colour of the photo at photo-plane (x, y), and whether it is inside the frame."""
        h, w = self.img.shape[:2]
        px = self.ex + x * self.S
        py = self.ey - y * self.S
        inside = (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
        px = np.clip(px, 0, w - 1.001); py = np.clip(py, 0, h - 1.001)
        x0 = px.astype(int); y0 = py.astype(int); fx = (px - x0)[..., None]; fy = (py - y0)[..., None]
        im = self.img
        c = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy) +
             im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)
        return c, inside

    def _sample_skin(self):
        pts = np.array([(-0.04, -0.035, 0.03), (0.04, -0.035, 0.03), (-0.03, -0.06, 0.03), (0.03, -0.06, 0.03)])
        c, _ = self.project(pts)
        return np.median(c, 0)

    def in_face(self, x, y):
        cx, cy, ax, ay = self.face_ellipse
        return ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 < 1.0

    def hair_mask(self, P):
        x, y = P[..., 0], P[..., 1]
        c, inside = self.project(P)
        lum = c @ np.array([0.3, 0.59, 0.11])
        sat = c.max(-1) - c.min(-1)
        blue_bg = (c[..., 2] > c[..., 0] + 0.02) & (lum > 0.35)
        return inside & (lum < 0.36) & ~blue_bg & ~self.in_face(x, y) & (sat < 0.35)

    # ------------------------------------------------------------ geometry
    def head_pos(self, th, ph):
        d = np.stack([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)], -1)
        cx, cy, cz = 0.0, 0.006, -0.040
        lower = smoothstep(0.25, -0.35, d[..., 1])                    # no crease at the equator
        x = cx + (0.079 + 0.005 * lower) * d[..., 0]
        y = cy + (0.112 + (self.jaw - 0.112) * lower) * d[..., 1]
        z = cz + 0.095 * d[..., 2]
        front = np.clip(d[..., 2] * 1.6, 0, 1)
        # jaw and chin narrow toward the front
        k = np.clip((-0.015 - y) / 0.11, 0, 1)
        x = x * (1 - 0.34 * k ** 1.8 * np.sqrt(front))
        z = z - 0.012 * k * front
        # relief, all measured from the photo landmarks
        nx = x - self.nose_dx
        ny = self.nose_y
        t = np.clip((0.012 - y) / (0.012 - ny), 0, 1)
        A = np.where(y >= ny, 0.005 + 0.019 * t ** 1.4, 0.024 * np.exp(-((y - ny) / 0.0065) ** 2))
        A = A * np.exp(-np.maximum(y - 0.012, 0) / 0.008)                  # bridge fades into the brow
        sig = 0.0055 + 0.0075 * t
        dz = A * np.exp(-nx ** 2 / (2 * sig ** 2))
        dz += 0.006 * (_g(nx, y, -0.0135, ny + 0.001, 0.0055, 0.0055) + _g(nx, y, 0.0135, ny + 0.001, 0.0055, 0.0055))
        for s in (-1, 1):
            dz -= 0.0075 * _g(x, y, s * 0.0315, 0.001, 0.015, 0.0095)        # sockets
            dz += 0.0025 * _g(x, y, s * 0.0315, 0.0, 0.009, 0.0045)          # eyeballs
            dz += 0.0045 * _g(x, y, s * 0.028, 0.018, 0.019, 0.0055)         # brow ridge
            dz += 0.0050 * _g(x, y, s * 0.046, -0.028, 0.012, 0.013)         # cheekbones
        mx = x - self.mouth_dx
        my, cy_ = self.mouth_y, self.chin_y
        dz += 0.0065 * _g(mx, y, 0, my, 0.019, 0.010)                         # lips
        dz -= 0.0022 * _g(mx, y, 0, my - 0.0045, 0.021, 0.0018)               # lip line
        dz += 0.0070 * _g(mx, y, 0, cy_, 0.016, 0.013)                        # chin
        dz += 0.0030 * _g(x, y, 0, 0.045, 0.04, 0.02)                         # forehead
        z = z + dz * front
        return np.stack([x, y, z], -1)

    def hair_pos(self, th, ph):
        d = np.stack([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)], -1)
        if self.cap is not None:
            x = 0.091 * d[..., 0]
            y = 0.028 + np.where(d[..., 1] > 0, 0.128, 0.07) * d[..., 1]
            z = -0.036 + 0.106 * d[..., 2]
            return np.stack([x, y, z], -1)
        # hair grows out of the scalp: push the head surface out along its normal,
        # thin at the hairline, full on the crown and over the ears
        def scalp(t, p):
            S0 = self.head_pos(t, p)
            e = 1e-3
            nn = norm(np.cross(self.head_pos(t + e, p) - self.head_pos(t - e, p),
                               self.head_pos(t, p + e) - self.head_pos(t, p - e)))
            out = S0 - np.array([0.0, 0.006, -0.04])
            nn = nn * np.sign(np.sum(nn * out, -1, keepdims=True))
            dz = np.cos(t) * np.cos(p)
            fz = np.clip(dz, 0, 1)
            thick = (0.005 + 0.024 * (1 - fz ** 2)) * self.hv
            wave = 1 + 0.22 * np.sin(t * 11 + p * 5) * np.sin(p * 9 - t * 4) + 0.1 * np.sin(t * 23 + p * 3)
            return S0 + nn * (thick * wave)[..., None]
        P = scalp(th, ph)
        # below the temples the hair falls straight down instead of tucking in under the skull
        ph0 = -0.12
        ring = scalp(th, np.full_like(ph, ph0))
        side = np.where(np.cos(th) < 0, 1.0, smoothstep(0.5, 0.85, np.abs(np.sin(th))))
        w = side * smoothstep(ph0 + 0.25, ph0 - 0.3, ph)
        flare = 1 + 0.15 * np.clip(ph0 - ph, 0, None)
        Ph = np.stack([ring[..., 0] * flare, P[..., 1], (ring[..., 2] + 0.04) * flare - 0.04], -1)
        return P + w[..., None] * (Ph - P)

    def _grid_mesh(self, fn, nth, nph, ph_range, tex):
        th = np.linspace(-np.pi, np.pi, nth + 1)
        ph = np.linspace(ph_range[0], ph_range[1], nph + 1)
        TH, PH = np.meshgrid(th, ph)
        V = fn(TH, PH).reshape(-1, 3)
        UV = np.stack([(TH + np.pi) / (2 * np.pi), (PH - ph_range[0]) / (ph_range[1] - ph_range[0])], -1).reshape(-1, 2)
        F = []
        w = nth + 1
        for j in range(nph):
            for i in range(nth):
                a = j * w + i
                F.append((a, a + 1, a + w + 1)); F.append((a, a + w + 1, a + w))
        F = np.array(F)
        # make sure the faces point outward
        P = V[F]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        if (np.einsum('ij,ij->i', n, P.mean(1) - np.array([0, 0.0, -0.04]))).sum() < 0:
            F = F[:, ::-1]
        return from_indexed(V, F, (1, 1, 1), smooth=True, UV=UV, tex=tex)

    # ------------------------------------------------------------ textures
    def _bake(self, fn, W, H, ph_range):
        u = (np.arange(W) + 0.5) / W
        v = (np.arange(H) + 0.5) / H
        TH = u[None, :] * 2 * np.pi - np.pi + 0 * v[:, None]
        PH = ph_range[0] + v[:, None] * (ph_range[1] - ph_range[0]) + 0 * u[None, :]
        P = fn(TH, PH)
        e = 1e-3
        n = norm(np.cross(fn(TH + e, PH) - fn(TH - e, PH), fn(TH, PH + e) - fn(TH, PH - e)))
        if n[H // 2, W // 2, 2] < 0:
            n = -n
        return P, n

    def head_texture(self, W=1024, H=512):
        P, n = self._bake(self.head_pos, W, H, (-np.pi / 2, np.pi / 2))
        col, inside = self.project(P)
        n = n.copy()
        n[..., 2] = n[..., 2] * np.cos(self.yaw) + n[..., 0] * np.sin(self.yaw)
        w = smoothstep(0.05, 0.35, n[..., 2]) * inside
        # outside the face oval only trust skin-coloured photo pixels (no collars, no backdrop)
        r, g, b = col[..., 0], col[..., 1], col[..., 2]
        lum = col @ np.array([0.3, 0.59, 0.11])
        skinlike = (r > g) & (g > b * 0.9) & (r - b > 0.06) & (lum > 0.22) & (lum < 0.92)
        w = w * np.where(self.in_face(P[..., 0], P[..., 1]), 1.0, skinlike)
        if self.cap is not None:
            fill = np.where(((P[..., 1] > -0.07) & (n[..., 2] < 0.3))[..., None], self.hair_col, self.skin)
        else:
            fill = np.where((P[..., 1] > 0.02)[..., None], self.hair_col, self.skin)
        img = col * w[..., None] + fill * (1 - w[..., None])
        return img[::-1].astype(np.float32)          # texture rows run top-down

    def hair_texture(self, W=1024, H=512):
        P, n = self._bake(self.hair_pos, W, H, (-np.pi / 2, np.pi / 2))
        x, y = P[..., 0], P[..., 1]
        col, inside = self.project(P)
        nz = n[..., 2] * np.cos(self.yaw) + n[..., 0] * np.sin(self.yaw)
        frontish = smoothstep(0.45, 0.7, nz) * inside
        front_face = self.in_face(x, y) & (P[..., 2] > -0.02)
        if self.cap is not None:
            th = (np.arange(W)[None, :] / W) * 2 * np.pi - np.pi
            line = 0.022 - 0.05 * (1 - np.cos(th)) / 2 + 0 * y
            alpha = y > line
            img = col * frontish[..., None] + self.cap * (1 - frontish[..., None])
            return np.concatenate([img, alpha[..., None].astype(float)], -1)[::-1].astype(np.float32)
        mask_photo = self.hair_mask(P)
        # painted strands for everything the photo cannot see
        rng = self.rng
        th = (np.arange(W)[None, :] / W) * 2 * np.pi
        streak = np.repeat(rng.random((1, W)), H, 0) * 0.5 + rng.random((H, W)) * 0.12
        streak += 0.2 * np.sin(th * 40 + np.arange(H)[:, None] * 0.05)
        painted = self.hair_col[None, None] * (0.75 + 0.6 * streak[..., None])
        use_photo = smoothstep(0.6, 0.85, nz) * inside * mask_photo
        img = col * use_photo[..., None] + painted * (1 - use_photo[..., None])
        # cut away wherever the photo shows his skin (forehead, temples, face), plus a ragged hem
        r, g, b = col[..., 0], col[..., 1], col[..., 2]
        lum = col @ np.array([0.3, 0.59, 0.11])
        skin = (r > g) & (g > b * 0.9) & (r - b > 0.06) & (lum > 0.25)
        facing = (nz > 0.3) & inside
        hem = -self.hair_len + 0.014 * np.sin(th * 23) + 0.01 * rng.standard_normal((1, W)).repeat(H, 0)
        alpha = (y > hem) & ~front_face & ~(facing & skin & ~mask_photo)
        out = np.concatenate([img, alpha[..., None].astype(float)], -1)
        return out[::-1].astype(np.float32)

    # ------------------------------------------------------------ public
    def meshes(self, head_tex, hair_tex, pivot=(0.0, -0.10, -0.045)):
        """Head + hair in a frame whose origin is the top of the neck (rotation pivot)."""
        head = self._grid_mesh(self.head_pos, 96, 72, (-np.pi / 2, np.pi / 2), head_tex)
        hair = self._grid_mesh(self.hair_pos, 96, 64, (-np.pi / 2, np.pi / 2), hair_tex)
        if self.cap is not None:
            from engine import loft, T, Rx, merge
            brim = loft([(0.0, 0.088, 0.075), (0.008, 0.088, 0.075)], n=28, color=tuple(self.cap * 0.9))
            brim = brim.xform(T(0, 0.022, 0.025) @ Rx(0.12))
            keep = brim.P[..., 2].mean(1) > 0.03                    # only the front half sticks out
            brim = Mesh(brim.P[keep], brim.N[keep], brim.UV[keep], brim.C[keep], brim.tex[keep],
                        brim.emis[keep], brim.spec[keep])
            hair = merge([hair, brim])
        off = -np.asarray(pivot)
        for m in (head, hair):
            m.P = m.P + off
        # the photo already carries its own lighting: let it read through the scene light a little
        return head.mat(emis=0.35, spec=0.10), hair.mat(emis=0.12, spec=0.008)


_CACHE = {}


def alonso():
    """Fernando Alonso, 2012 (assets/faces/alonso.webp)."""
    if 'alonso' not in _CACHE:
        _CACHE['alonso'] = PhotoHead('alonso.webp', (493, 313), 2220.0, nose_dx=-0.003, mouth_dx=-0.006,
                                     hair_len=0.10, hair_col=(0.11, 0.075, 0.05))
    return _CACHE['alonso']


def vettel():
    """Sebastian Vettel at a 2010 press conference, Red Bull cap on (assets/faces/vettel.jpg)."""
    if 'vettel' not in _CACHE:
        _CACHE['vettel'] = PhotoHead('vettel.jpg', (273, 222), 850.0, photo_yaw=0.25, nose_dx=-0.008,
                                     mouth_dx=-0.023, nose_y=-0.047, mouth_y=-0.089, chin_y=-0.140,
                                     jaw=0.155, face_ellipse=(-0.018, -0.045, 0.055, 0.10),
                                     cap=(0.07, 0.08, 0.17), hair_col=(0.62, 0.47, 0.28))
    return _CACHE['vettel']
