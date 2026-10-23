"""Tiny PS1-style software rasterizer written with numpy.

Geometry is stored as unindexed triangle soup. Rasterization writes a
visibility buffer (triangle id + screen-space barycentrics); attributes are
then resolved for the whole frame at once. UVs are interpolated affinely
(PS1 texture warp), vertices are snapped to whole pixels (PS1 jitter).
"""
import numpy as np


# ---------------------------------------------------------------- matrices
def T(x, y, z):
    m = np.eye(4); m[:3, 3] = (x, y, z); return m


def S(x, y=None, z=None):
    if y is None: y = z = x
    return np.diag([x, y, z, 1.0])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def R(x=0, y=0, z=0):
    return Ry(y) @ Rx(x) @ Rz(z)


def norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- meshes
class Mesh:
    """Triangle soup. P,N: (m,3,3)  UV: (m,3,2)  C: (m,3,3)  + per-tri material."""

    def __init__(self, P, N, UV, C, tex=-1, emis=0.0, spec=0.0):
        m = len(P)
        self.P, self.N, self.UV, self.C = P, N, UV, C
        self.tex = np.full(m, tex, np.int32) if np.isscalar(tex) else tex
        self.emis = np.full(m, emis, np.float32) if np.isscalar(emis) else emis
        self.spec = np.full(m, spec, np.float32) if np.isscalar(spec) else spec

    def xform(self, M):
        A = M[:3, :3]
        P = self.P @ A.T + M[:3, 3]
        N = norm(self.N @ np.linalg.inv(A))
        return Mesh(P, N, self.UV, self.C, self.tex, self.emis, self.spec)

    def tint(self, rgb, tex=None):
        C = np.empty_like(self.C); C[:] = rgb
        return Mesh(self.P, self.N, self.UV, C,
                    self.tex if tex is None else tex, self.emis, self.spec)

    def mat(self, emis=None, spec=None, tex=None):
        m = Mesh(self.P, self.N, self.UV, self.C, self.tex, self.emis, self.spec)
        n = len(self.P)
        if emis is not None: m.emis = np.full(n, emis, np.float32)
        if spec is not None: m.spec = np.full(n, spec, np.float32)
        if tex is not None: m.tex = np.full(n, tex, np.int32)
        return m


def merge(meshes):
    meshes = [m for m in meshes if m is not None and len(m.P)]
    cat = lambda k: np.concatenate([getattr(m, k) for m in meshes])
    return Mesh(cat('P'), cat('N'), cat('UV'), cat('C'), cat('tex'), cat('emis'), cat('spec'))


def from_indexed(V, F, color, smooth=False, UV=None, **kw):
    V = np.asarray(V, float); F = np.asarray(F, int)
    P = V[F]
    fn = norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]))
    if smooth:
        vn = np.zeros_like(V)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        N = norm(vn)[F]
    else:
        N = np.repeat(fn[:, None], 3, 1)
    uv = np.zeros((len(F), 3, 2)) if UV is None else np.asarray(UV)[F]
    C = np.empty((len(F), 3, 3)); C[:] = color
    return Mesh(P, N, uv, C, **kw)


def box(sx, sy, sz, color, **kw):
    x, y, z = sx / 2, sy / 2, sz / 2
    V = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
         (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return from_indexed(V, F, color, **kw)


def tbox(sx, sy, sz, color, tile=1.0, **kw):
    """Box with per-face UVs in world units / tile (for tiling textures)."""
    x, y, z = sx / 2, sy / 2, sz / 2
    faces = [  # corner order: bl, br, tr, tl seen from outside
        ([(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)], sx, sy),
        ([(x, -y, -z), (-x, -y, -z), (-x, y, -z), (x, y, -z)], sx, sy),
        ([(x, -y, z), (x, -y, -z), (x, y, -z), (x, y, z)], sz, sy),
        ([(-x, -y, -z), (-x, -y, z), (-x, y, z), (-x, y, -z)], sz, sy),
        ([(-x, y, z), (x, y, z), (x, y, -z), (-x, y, -z)], sx, sz),
        ([(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)], sx, sz)]
    V, UV, F = [], [], []
    for corners, w, h in faces:
        b = len(V)
        V += corners
        UV += [(0, 0), (w / tile, 0), (w / tile, h / tile), (0, h / tile)]
        F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return from_indexed(V, F, color, UV=UV, **kw)


def loft(sections, n=8, color=(1, 1, 1), smooth=True, caps=True, power=2.0, deform=None, **kw):
    """Generalised cylinder along +y.

    sections: list of (y, rx, rz[, cx, cz]). Ring angle 0 points to +z
    (character front) and the UV seam sits at the back (-z).
    """
    rings, V, UV = [], [], []
    ys = [s[0] for s in sections]
    y0, y1 = min(ys), max(ys)
    for j, s in enumerate(sections):
        y, rx, rz = s[:3]
        cx, cz = (s[3], s[4]) if len(s) > 3 else (0.0, 0.0)
        ring = []
        for i in range(n + 1):
            a = np.pi + 2 * np.pi * i / n  # start at back
            ca, sa = np.cos(a), np.sin(a)
            e = 2.0 / power
            px = cx + rx * np.sign(sa) * abs(sa) ** e
            pz = cz + rz * np.sign(ca) * abs(ca) ** e
            ring.append(len(V)); V.append((px, y, pz))
            UV.append((i / n, (y - y0) / max(y1 - y0, 1e-6)))
        rings.append(ring)
    F = []
    for j in range(len(rings) - 1):
        a, b = rings[j], rings[j + 1]
        for i in range(n):
            F.append((a[i], a[i + 1], b[i + 1])); F.append((a[i], b[i + 1], b[i]))
    if caps:
        for j, flip in ((0, True), (len(rings) - 1, False)):
            r = rings[j]
            c = len(V); s = sections[j]
            V.append((s[3] if len(s) > 3 else 0, s[0], s[4] if len(s) > 3 else 0))
            UV.append((0.5, 0.0 if flip else 1.0))
            for i in range(n):
                F.append((c, r[i + 1], r[i]) if flip else (c, r[i], r[i + 1]))
    if deform is not None:
        V = deform(np.array(V, float))
    # rings run back -> -x -> front -> +x, so faces point outward when y rises
    F = np.array(F, int)
    if sections[-1][0] < sections[0][0]:
        F = F[:, ::-1]
    return from_indexed(V, F, color, smooth=smooth, UV=UV, **kw)


def sphere(rx, ry, rz, n=10, rings=7, color=(1, 1, 1), **kw):
    secs = []
    for j in range(rings + 1):
        t = -np.pi / 2 + np.pi * j / rings
        r = max(np.cos(t), 1e-3)
        secs.append((ry * np.sin(t), rx * r, rz * r))
    return loft(secs, n=n, color=color, caps=False, **kw)


def quad(w, h, color, uv=((0, 0), (1, 0), (1, 1), (0, 1)), **kw):
    """Quad in the xy plane facing +z."""
    V = [(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)]
    return from_indexed(V, [(0, 1, 2), (0, 2, 3)], color, UV=uv, **kw)


def grid(size, cells, color, y=0.0, tile=None, **kw):
    xs = np.linspace(-size / 2, size / 2, cells + 1)
    V, UV, F = [], [], []
    for j, z in enumerate(xs):
        for i, x in enumerate(xs):
            V.append((x, y, z))
            UV.append((x / tile, -z / tile) if tile else (i / cells * 8, j / cells * 8))
    w = cells + 1
    for j in range(cells):
        for i in range(cells):
            a = j * w + i
            F.append((a, a + w, a + 1)); F.append((a + 1, a + w, a + w + 1))
    return from_indexed(V, F, color, UV=UV, **kw)


# ---------------------------------------------------------------- camera
class Camera:
    def __init__(self, eye, target, fov_deg, W, H, up=(0, 1, 0), roll=0.0):
        self.eye = np.asarray(eye, float)
        f = norm(np.asarray(target, float) - self.eye)
        r = norm(np.cross(f, up)); u = np.cross(r, f)
        if roll:
            c, s = np.cos(roll), np.sin(roll)
            r, u = c * r + s * u, -s * r + c * u
        self.f, self.r, self.u = f, r, u
        self.W, self.H = W, H
        self.fpx = (H / 2) / np.tan(np.radians(fov_deg) / 2)

    def scaled(self, k):
        """Same camera at a different resolution."""
        import copy
        c = copy.copy(self)
        c.W, c.H, c.fpx = int(self.W * k), int(self.H * k), self.fpx * k
        return c

    def view(self, P):
        d = P - self.eye
        return d @ self.r, d @ self.u, d @ self.f

    def project(self, P):
        x, y, z = self.view(P)
        zs = np.maximum(z, 1e-6)
        return self.W / 2 + x / zs * self.fpx, self.H / 2 - y / zs * self.fpx, z

    def rays(self):
        ys, xs = np.mgrid[0:self.H, 0:self.W] + 0.5
        dx = (xs - self.W / 2) / self.fpx; dy = -(ys - self.H / 2) / self.fpx
        d = dx[..., None] * self.r + dy[..., None] * self.u + self.f
        return d  # not normalised: t along it equals view depth


# ---------------------------------------------------------------- raster
def clip_near(mesh, cam, near):
    """Split triangles that cross the near plane (Sutherland-Hodgman)."""
    _, _, z = cam.view(mesh.P.reshape(-1, 3))
    z = z.reshape(-1, 3)
    nb = (z < near).sum(1)
    cross = (nb == 1) | (nb == 2)
    if not cross.any():
        return mesh
    keys = ('P', 'N', 'UV', 'C')
    new = {k: [] for k in keys}
    src = []
    for t in np.nonzero(cross)[0]:
        zs = z[t]
        attrs = [getattr(mesh, k)[t] for k in keys]
        poly = []
        for i in range(3):
            j = (i + 1) % 3
            a_in, b_in = zs[i] >= near, zs[j] >= near
            if a_in:
                poly.append([A[i] for A in attrs])
            if a_in != b_in:
                f = (near - zs[i]) / (zs[j] - zs[i])
                poly.append([A[i] + (A[j] - A[i]) * f for A in attrs])
        for k in range(1, len(poly) - 1):
            for ai, key in enumerate(keys):
                new[key].append(np.stack([poly[0][ai], poly[k][ai], poly[k + 1][ai]]))
            src.append(t)
    keep = np.nonzero(~cross)[0]
    src = np.array(src, int)
    cat = lambda k: np.concatenate([getattr(mesh, k)[keep], np.array(new[k]).reshape(-1, *getattr(mesh, k).shape[1:])])
    return Mesh(cat('P'), cat('N'), cat('UV'), cat('C'),
                np.concatenate([mesh.tex[keep], mesh.tex[src]]),
                np.concatenate([mesh.emis[keep], mesh.emis[src]]),
                np.concatenate([mesh.spec[keep], mesh.spec[src]]))


def rasterize(mesh, cam, near=0.05, snap=True, cull=True, textures=None):
    """Returns (tid, b0, b1, invz, mesh) visibility buffers.

    The returned mesh is the near-clipped one that tid indexes into.
    Textures with an alpha channel are alpha-tested (chain-link, chains).
    """
    W, H = cam.W, cam.H
    mesh = clip_near(mesh, cam, near * 1.01)
    P = mesh.P.reshape(-1, 3)
    sx, sy, z = cam.project(P)
    sx = sx.reshape(-1, 3); sy = sy.reshape(-1, 3); z = z.reshape(-1, 3)
    if snap:
        sx = np.floor(sx) + 0.5; sy = np.floor(sy) + 0.5
    keep = (z > near).all(1)
    keep &= (sx.max(1) >= 0) & (sx.min(1) < W) & (sy.max(1) >= 0) & (sy.min(1) < H)
    if cull:
        g = np.cross(mesh.P[:, 1] - mesh.P[:, 0], mesh.P[:, 2] - mesh.P[:, 0])
        keep &= np.einsum('ij,ij->i', g, cam.eye - mesh.P[:, 0]) > 0
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    keep &= np.abs(area) > 1e-6
    idx = np.nonzero(keep)[0]
    alpha_ids = set()
    if textures is not None:
        alpha_ids = {i for i, im in enumerate(textures) if im.shape[2] == 4}

    tid = np.full((H, W), -1, np.int32)
    b0 = np.zeros((H, W), np.float32); b1 = np.zeros((H, W), np.float32)
    invz = np.zeros((H, W), np.float32)
    if not len(idx):
        return tid, b0, b1, invz, mesh
    X, Y, A = sx[idx], sy[idx], area[idx]
    TEX = mesh.tex[idx]
    UVs = mesh.UV[idx]
    IZ = 1.0 / z[idx]
    # edge function coefficients: w_k = (a_k*x + b_k*y + c_k) / area
    i1, i2 = [1, 2, 0], [2, 0, 1]
    ea = (Y[:, i1] - Y[:, i2]) / A[:, None]
    eb = (X[:, i2] - X[:, i1]) / A[:, None]
    ec = (X[:, i1] * Y[:, i2] - X[:, i2] * Y[:, i1]) / A[:, None]
    x0 = np.clip(np.floor(X.min(1)).astype(int), 0, W); x1 = np.clip(np.ceil(X.max(1)).astype(int) + 1, 0, W)
    y0 = np.clip(np.floor(Y.min(1)).astype(int), 0, H); y1 = np.clip(np.ceil(Y.max(1)).astype(int) + 1, 0, H)
    px = np.arange(W) + 0.5; py = np.arange(H) + 0.5
    eps = -1e-5
    bw, bh = x1 - x0, y1 - y0
    has_alpha = np.isin(TEX, list(alpha_ids)) if alpha_ids else np.zeros(len(idx), bool)
    small = (bw <= 8) & (bh <= 8) & ~has_alpha
    for k in np.nonzero(~small)[0]:
        xa, xb, ya, yb = x0[k], x1[k], y0[k], y1[k]
        if xa >= xb or ya >= yb:
            continue
        cx = px[xa:xb]; cy = py[ya:yb]
        a, b, c = ea[k], eb[k], ec[k]
        w0 = a[0] * cx[None, :] + (b[0] * cy + c[0])[:, None]
        w1 = a[1] * cx[None, :] + (b[1] * cy + c[1])[:, None]
        w2 = 1.0 - w0 - w1
        iz = IZ[k]
        d = w0 * iz[0] + w1 * iz[1] + w2 * iz[2]
        zb = invz[ya:yb, xa:xb]
        m = (w0 >= eps) & (w1 >= eps) & (w2 >= eps) & (d > zb)
        if TEX[k] in alpha_ids and m.any():
            img = textures[TEX[k]]
            h, wd = img.shape[:2]
            uv = UVs[k]
            u = w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0]
            v = w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1]
            m &= img[np.floor((1 - v) * h).astype(int) % h, np.floor(u * wd).astype(int) % wd, 3] > 0.5
        if not m.any():
            continue
        zb[m] = d[m]
        tid[ya:yb, xa:xb][m] = idx[k]
        b0[ya:yb, xa:xb][m] = w0[m]
        b1[ya:yb, xa:xb][m] = w1[m]
    _raster_small(np.nonzero(small)[0], idx, x0, y0, bw, bh, ea, eb, ec, IZ, W, tid, b0, b1, invz, eps)
    return tid, b0, b1, invz, mesh


def _raster_small(ks, idx, x0, y0, bw, bh, ea, eb, ec, IZ, W, tid, b0, b1, invz, eps):
    """Vectorised path for the many tiny triangles of high-poly meshes."""
    if not len(ks):
        return
    P, D, Tr, B0, B1 = [], [], [], [], []
    lo = 0
    for size in (2, 4, 8):
        sel = ks[(np.maximum(bw[ks], bh[ks]) > lo) & (np.maximum(bw[ks], bh[ks]) <= size)]
        lo = size
        if not len(sel):
            continue
        oy, ox = np.mgrid[0:size, 0:size].reshape(2, -1)
        cx = x0[sel, None] + ox[None]
        cy = y0[sel, None] + oy[None]
        ok = (ox[None] < bw[sel, None]) & (oy[None] < bh[sel, None])
        fx, fy = cx + 0.5, cy + 0.5
        w0 = ea[sel, 0, None] * fx + eb[sel, 0, None] * fy + ec[sel, 0, None]
        w1 = ea[sel, 1, None] * fx + eb[sel, 1, None] * fy + ec[sel, 1, None]
        w2 = 1.0 - w0 - w1
        ok &= (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        d = w0 * IZ[sel, 0, None] + w1 * IZ[sel, 1, None] + w2 * IZ[sel, 2, None]
        P.append((cy * W + cx)[ok]); D.append(d[ok])
        Tr.append(np.broadcast_to(idx[sel, None], ok.shape)[ok]); B0.append(w0[ok]); B1.append(w1[ok])
    if not P:
        return
    P = np.concatenate(P); D = np.concatenate(D).astype(np.float32)
    flat = invz.reshape(-1)
    np.maximum.at(flat, P, D)
    win = D >= flat[P]
    tid.reshape(-1)[P[win]] = np.concatenate(Tr)[win]
    b0.reshape(-1)[P[win]] = np.concatenate(B0)[win]
    b1.reshape(-1)[P[win]] = np.concatenate(B1)[win]


def resolve(cam, vis, textures, affine_uv=False):
    """Turn the visibility buffer into a G-buffer."""
    tid, b0, b1, invz, mesh = vis
    valid = tid >= 0
    t = tid[valid]
    w = np.stack([b0[valid], b1[valid], 1 - b0[valid] - b1[valid]], 1)
    _, _, z = cam.view(mesh.P[t].reshape(-1, 3))
    iz = 1.0 / np.maximum(z.reshape(-1, 3), 1e-6)
    pw = w * iz; pw /= pw.sum(1, keepdims=True)   # perspective-correct
    pos = np.einsum('nk,nkc->nc', pw, mesh.P[t])
    nrm = norm(np.einsum('nk,nkc->nc', pw, mesh.N[t]))
    col = np.einsum('nk,nkc->nc', w, mesh.C[t])
    uv = np.einsum('nk,nkc->nc', w if affine_uv else pw, mesh.UV[t])
    tex = mesh.tex[t]
    for ti in np.unique(tex):
        if ti < 0:
            continue
        sel = tex == ti
        img = textures[ti]
        h, wd = img.shape[:2]
        u = np.floor(uv[sel, 0] * wd).astype(int) % wd
        v = np.floor((1 - uv[sel, 1]) * h).astype(int) % h
        col[sel] *= img[v, u, :3]
    G = dict(valid=valid, pos=pos, nrm=nrm, alb=col, emis=mesh.emis[t], spec=mesh.spec[t],
             depth=np.where(valid, 1.0 / np.maximum(invz, 1e-9), 1e9))
    return G
