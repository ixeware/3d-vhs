"""Spotlight shading, shadow map, volumetric beam, dust, PS1 dither."""
import numpy as np
from engine import Camera, rasterize, resolve, norm


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Spot:
    def __init__(self, pos, target, inner_deg, outer_deg, intensity, color=(1.0, 0.92, 0.78),
                 shadow_res=480):
        self.pos = np.asarray(pos, float)
        self.target = np.asarray(target, float)
        self.dir = norm(np.asarray(target, float) - self.pos)
        self.cin, self.cout = np.cos(np.radians(inner_deg)), np.cos(np.radians(outer_deg))
        self.I = intensity
        self.color = np.asarray(color)
        up = (0, 0, 1) if abs(self.dir[1]) > 0.95 else (0, 1, 0)
        self.cam = Camera(self.pos, self.pos + self.dir, outer_deg * 2 + 4, shadow_res, shadow_res, up=up)
        self.depth = None

    def build_shadow(self, mesh):
        vis = rasterize(mesh, self.cam, near=0.1, snap=False, cull=False)
        self.depth = np.where(vis[0] >= 0, 1.0 / np.maximum(vis[3], 1e-9), 1e9)

    def shadow(self, Q, bias=0.06):
        if self.depth is None or self.I <= 0:
            return np.ones(len(Q))
        sx, sy, z = self.cam.project(Q)
        n = self.cam.W
        ix = np.clip(sx.astype(int), 0, n - 1); iy = np.clip(sy.astype(int), 0, n - 1)
        return (z <= self.depth[iy, ix] + bias).astype(float)

    def radiance(self, Q):
        """Incoming light (scalar) at points Q ignoring shadow and normal."""
        L = self.pos - Q
        d = np.linalg.norm(L, axis=-1)
        Ld = L / d[..., None]
        cone = smoothstep(self.cout, self.cin, -(Ld @ self.dir))
        return self.I * cone / (1 + (d / 4.0) ** 2), Ld


def shade(G, cam, spot, ambient=0.0025, bounce=0.3):
    H, W = cam.H, cam.W
    out = np.zeros((H, W, 3))
    v = G['valid']
    pos, nrm = G['pos'], G['nrm']
    alb = np.clip(G['alb'], 0, 1) ** 2.2
    rad, Ld = spot.radiance(pos)
    sh = spot.shadow(pos)
    ndl = np.clip((np.einsum('ij,ij->i', nrm, Ld) + 0.25) / 1.25, 0, 1)
    light = (rad * ndl * sh)[:, None] * spot.color
    Vd = norm(cam.eye - pos)
    hv = norm(Ld + Vd)
    sp = np.clip(np.einsum('ij,ij->i', nrm, hv), 0, 1) ** 40 * G['spec'] * rad * sh * 2.0
    # light bouncing back up off the lit floor pool
    pool = spot.target
    dxz = (pos[:, 0] - pool[0]) ** 2 + (pos[:, 2] - pool[2]) ** 2
    up = np.clip(0.55 - 0.45 * nrm[:, 1], 0, 1) * np.exp(-dxz / 3.0) * np.exp(-np.maximum(pos[:, 1] - 0.5, 0) / 2.5)
    fill = (bounce * spot.I / 10.0 * up)[:, None] * spot.color * np.array([1.0, 0.85, 0.75])
    col = alb * (ambient * np.array([0.6, 0.7, 1.0]) + light + fill) + sp[:, None] * spot.color
    col += alb * G['emis'][:, None]
    out[v] = col
    return out


def volume(cam, spot, depth, t_frame, steps=28, density=0.035, seed=0):
    """Ray-marched light shaft through hazy air, shadowed by the shadow map."""
    if spot.I <= 0:
        return np.zeros((cam.H, cam.W, 3))
    d = cam.rays().reshape(-1, 3)
    dl = np.linalg.norm(d, axis=1)
    tmax = np.minimum(depth.reshape(-1), 18.0)
    rng = np.random.default_rng(seed)
    jit = rng.random(len(d))
    acc = np.zeros(len(d))
    dt = tmax / steps
    for k in range(steps):
        t = (k + jit) * dt
        Q = cam.eye + d * t[:, None]
        r, _ = spot.radiance(Q)
        live = r > 1e-4
        if not live.any():
            continue
        s = np.zeros(len(d))
        s[live] = r[live] * spot.shadow(Q[live], bias=0.02)
        # drifting smoke
        q = Q[live]
        n = (np.sin(q[:, 0] * 2.1 + t_frame * 0.31) * np.sin(q[:, 1] * 1.7 - t_frame * 0.23)
             * np.sin(q[:, 2] * 2.6 + t_frame * 0.17))
        s[live] *= 0.75 + 0.45 * n
        acc += s * dt * dl
    return (acc * density)[:, None].reshape(cam.H, cam.W, 1) * spot.color


def dust(img, cam, spot, depth, t, n=500, seed=7, box=((-3, 3), (0, 5), (-3, 3))):
    if spot.I <= 0:
        return
    rng = np.random.default_rng(seed)
    p0 = np.stack([rng.uniform(*box[0], n), rng.uniform(*box[1], n), rng.uniform(*box[2], n)], 1)
    ph = rng.uniform(0, 6.28, (n, 3))
    P = p0 + 0.15 * np.sin(t * np.array([0.3, 0.21, 0.27]) + ph)
    P[:, 1] = (P[:, 1] - t * 0.03 - box[1][0]) % (box[1][1] - box[1][0]) + box[1][0]
    r, _ = spot.radiance(P)
    sx, sy, z = cam.project(P)
    ok = (z > 0.2) & (sx >= 0) & (sx < cam.W) & (sy >= 0) & (sy < cam.H) & (r > 0.01)
    ix, iy = sx[ok].astype(int), sy[ok].astype(int)
    vis = z[ok] < depth[iy, ix]
    b = r[ok][vis] * 0.012 * rng.uniform(0.3, 1.0, ok.sum())[vis]
    np.add.at(img, (iy[vis], ix[vis]), b[:, None] * spot.color)


def bloom(img, thresh=0.8, radius=4, strength=0.35):
    b = np.clip(img - thresh, 0, None)
    for _ in range(2):
        b = _box(b, radius)
    return img + b * strength


def _box(a, r):
    for ax in (0, 1):
        c = np.cumsum(np.pad(a, [(r + 1, r) if i == ax else (0, 0) for i in range(3)], mode='edge'), axis=ax)
        a = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) -
             np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return a


def tonemap(lin, exposure=1.0):
    x = lin * exposure
    x = x * (1 + x / 9.0) / (1 + x)       # gentle shoulder
    return np.clip(x, 0, 1) ** (1 / 2.2)


DITHER = np.array([[-4, 0, -3, 1], [2, -2, 3, -1], [-3, 1, -4, 0], [3, -1, 2, -2]], float)


def ps1_dither(rgb01):
    H, W = rgb01.shape[:2]
    d = np.tile(DITHER, (H // 4 + 1, W // 4 + 1))[:H, :W, None]
    v = np.clip(rgb01 * 255 + d, 0, 255)
    return np.floor(v / 8) * 8 / 248.0
