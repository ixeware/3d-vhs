"""Low-poly characters (simple FK rig), the RB8 and the theatre lamp."""
import numpy as np
from engine import (T, S, R, Rx, Ry, Rz, Mesh, merge, box, loft, sphere, quad, grid,
                    from_indexed, norm)
from textures import HEAD_RY


# ---------------------------------------------------------------- body parts
def _head_deform(V):
    V = V.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    low = np.clip(-y / 0.11, 0, 1)
    x *= 1 - 0.30 * low ** 1.5                   # jaw taper
    front = z > 0
    z[front] *= 1 - 0.10 * low[front]
    ang = np.degrees(np.arctan2(x, z))
    nose = (np.abs(ang) < 8) & (y > -0.045) & (y < 0.02) & front
    z[nose] += 0.018 * (1 - np.abs(y[nose] + 0.015) / 0.035)
    sock = (np.abs(np.abs(ang) - 24) < 12) & (np.abs(y - 0.012) < 0.016) & front
    z[sock] -= 0.006
    brow = (np.abs(ang) < 40) & (np.abs(y - 0.035) < 0.012) & front
    z[brow] += 0.004
    chin = (np.abs(ang) < 20) & (y < -0.085) & front
    z[chin] += 0.006
    return V


def head_mesh(tex, rx=0.082, rz=0.098):
    return sphere(rx, HEAD_RY, rz, n=14, rings=10, tex=tex, deform=_head_deform, smooth=True)


def hair_mesh(color, rng, spiky=0.012, front=0.06, back=-0.07, scale=1.07, top=1.0):
    def deform(V):
        V = V.copy()
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        ang = np.arctan2(x, z)
        line = front + (back - front) * (np.abs(ang) / np.pi) ** 0.8
        below = y < line
        k = np.where(below, 0.86, 1.0)
        V[:, 0] *= k; V[:, 2] *= k
        y[below] = line[below]
        h = np.sin(np.round(V, 4) @ np.array([129.898, 782.33, 377.19])) * 43758.5453
        jit = 1 + spiky * 4 * (h - np.floor(h))      # position hash: seams stay closed
        V[~below] *= jit[~below, None]
        V[:, 1] = np.where(V[:, 1] > 0, V[:, 1] * top, V[:, 1])
        return V
    return sphere(0.082 * scale, HEAD_RY * scale, 0.098 * scale, n=12, rings=8, color=color,
                  deform=deform, smooth=False)


def limb(length, r0, r1, color, n=7, **kw):
    return loft([(0, r0, r0), (-length * 0.5, (r0 + r1) / 2 * 1.04, (r0 + r1) / 2),
                 (-length, r1, r1)], n=n, color=color, **kw)


# ---------------------------------------------------------------- character
def mirror_x(m):
    """Mirror a mesh left/right, keeping its faces pointing outward."""
    M = S(-1, 1, 1)
    out = m.xform(M)
    return Mesh(out.P[:, ::-1], out.N[:, ::-1], out.UV[:, ::-1], out.C[:, ::-1], out.tex, out.emis, out.spec)


def capsule(length, r0, r1, color, n=10, **kw):
    """Rounded limb segment hanging down -y from the joint."""
    secs = [(0.0, r0 * 0.55, r0 * 0.55)]
    for t in np.linspace(0, 1, 6):
        r = r0 + (r1 - r0) * t
        bulge = 1 + 0.06 * np.sin(np.pi * t)
        secs.append((-(r0 * 0.4) - t * (length - (r0 + r1) * 0.4), r * bulge, r * bulge * 0.97))
    secs.append((-length, r1 * 0.55, r1 * 0.55))
    return loft(secs, n=n, color=color, **kw)


def hand_mesh(kind, color, spec=0.05):
    """Left hand hanging down: fingers along -y, palm facing -x (toward the thigh), thumb toward +z."""
    parts = [loft([(0.0, 0.016, 0.040), (-0.045, 0.017, 0.045), (-0.085, 0.015, 0.042)],
                  n=14, color=color, power=2.6)]
    curl = {'hand': (0.35, 0.3), 'fist': (1.5, 1.4), 'point': (1.5, 1.4)}[kind]
    for k, (z, L) in enumerate(((0.030, 0.070), (0.010, 0.080), (-0.010, 0.076), (-0.030, 0.062))):
        c0, c1 = curl
        if kind == 'point' and k == 0:
            c0, c1 = 0.0, 0.0
        seg1 = capsule(L * 0.55, 0.0095, 0.0088, color, n=8)
        seg2 = capsule(L * 0.5, 0.0088, 0.0075, color, n=8)
        M1 = T(0, -0.083, z) @ Rz(c0)
        M2 = M1 @ T(0, -L * 0.52, 0) @ Rz(c1)
        parts += [seg1.xform(M1), seg2.xform(M2)]
    th = 0.25 if kind == 'hand' else 0.9
    parts.append(capsule(0.05, 0.011, 0.009, color, n=8).xform(T(-0.004, -0.02, 0.04) @ Rx(-0.7) @ Rz(th)))
    return merge(parts).mat(spec=spec)


class Look:
    def __init__(self, face, suit_tex, suit_col, limb_col, skin, hair, boots,
                 cap=None, hair_kw=None, gloves=None, seed=0, legs_col=None,
                 sleeve_tex=None, legs_tex=None, photo=None):
        self.face, self.suit_tex, self.suit_col = face, suit_tex, suit_col
        self.limb_col, self.skin, self.hair, self.boots = limb_col, skin, hair, boots
        self.cap, self.gloves = cap, gloves
        self.legs_col = legs_col or limb_col
        self.sleeve_tex, self.legs_tex = sleeve_tex, legs_tex
        self.photo = photo            # (PhotoHead, head texture name, hair texture name)
        self.rng = np.random.default_rng(seed)
        self.hair_kw = hair_kw or {}
        self._cache = {}

    def parts(self, tex_ix, blink=False):
        key = blink
        if key in self._cache:
            return self._cache[key]
        P = {}
        if self.photo is not None:
            ph, ht, hr = self.photo
            P['head'], P['hair'] = ph.meshes(tex_ix[ht], tex_ix[hr])
            skin = tuple(ph.skin)
        else:
            face = self.face + ('_blink' if blink and self.face + '_blink' in tex_ix else '')
            P['head'] = sphere(0.082, HEAD_RY, 0.098, n=32, rings=22, tex=tex_ix[face], deform=_head_deform)
            if self.cap == 'helmet':
                shell = sphere(0.122, 0.14, 0.135, n=36, rings=24, color=(0.07, 0.09, 0.25),
                               deform=lambda V: np.where(V[:, 1:2] < -0.09, np.c_[V[:, 0], np.full(len(V), -0.09), V[:, 2]], V))
                stripe = loft([(-0.002, 0.123, 0.136), (0.03, 0.123, 0.136)], n=36, color=(0.85, 0.08, 0.16), caps=False)
                visor = sphere(0.118, 0.06, 0.128, n=28, rings=8, color=(0.02, 0.02, 0.03),
                               deform=lambda V: V[V[:, 2] > -1])
                keep = visor.P[..., 2].mean(1) > 0.06
                visor = Mesh(visor.P[keep], visor.N[keep], visor.UV[keep], visor.C[keep], visor.tex[keep],
                             visor.emis[keep], visor.spec[keep]).mat(spec=1.2)
                P['hair'] = merge([shell.mat(spec=0.9).xform(T(0, 0.015, -0.005)), stripe.xform(T(0, 0.04, -0.005)),
                                   visor.xform(T(0, 0.0, 0.012))])
            elif self.cap is None:
                P['hair'] = hair_mesh(self.hair, np.random.default_rng(1), **self.hair_kw)
            else:
                cap = sphere(0.082 * 1.1, HEAD_RY * 1.05, 0.098 * 1.1, n=28, rings=16, color=self.cap,
                             deform=lambda V: np.where(V[:, 1:2] < 0.045, np.c_[V[:, 0] * 0.9, np.full(len(V), 0.045), V[:, 2] * 0.9], V))
                brim = loft([(0, 0.075, 0.06), (0.012, 0.075, 0.06)], n=20, color=self.cap).xform(
                    T(0, 0.045, 0.085) @ Rx(0.18) @ S(1, 1, 1.0))
                P['hair'] = merge([cap, brim])
            P['ear'] = sphere(0.012, 0.026, 0.018, n=10, rings=6, color=self.skin)
            skin = self.skin
        sl = dict(tex=tex_ix[self.sleeve_tex]) if self.sleeve_tex else {}
        lg = dict(tex=tex_ix[self.legs_tex]) if self.legs_tex else {}
        limb_c = (1, 1, 1) if sl else self.limb_col
        legs_c = (1, 1, 1) if lg else self.legs_col
        P['neck'] = loft([(-0.02, 0.056, 0.056), (0.04, 0.05, 0.052), (0.09, 0.047, 0.05)], n=18, color=skin)
        P['collar'] = merge([loft([(0, 0.078, 0.072), (0.03, 0.068, 0.064), (0.06, 0.062, 0.06)],
                                  n=24, color=self.suit_col),
                             loft([(0.06, 0.062, 0.06), (0.066, 0.06, 0.058)], n=24, color=(0.05, 0.05, 0.06))])
        P['deltoid'] = sphere(0.056, 0.058, 0.058, n=20, rings=12, color=limb_c, **sl)
        P['torso'] = loft([(0, 0.148, 0.100), (0.08, 0.150, 0.103), (0.16, 0.153, 0.106), (0.24, 0.163, 0.112),
                           (0.31, 0.166, 0.115), (0.37, 0.168, 0.110), (0.42, 0.160, 0.100), (0.46, 0.140, 0.088),
                           (0.49, 0.100, 0.074), (0.505, 0.075, 0.066)],
                          n=32, color=(1, 1, 1), tex=tex_ix[self.suit_tex], power=2.3)
        P['pelvis'] = loft([(-0.11, 0.128, 0.094), (-0.06, 0.145, 0.100), (-0.01, 0.150, 0.102), (0.06, 0.148, 0.100)],
                           n=32, color=legs_c, power=2.3, **lg)
        P['uarm'] = capsule(0.29, 0.056, 0.043, limb_c, n=18, **sl)
        P['farm'] = capsule(0.25, 0.043, 0.032, limb_c, n=18, **sl)
        hc = self.gloves or skin
        for k in ('hand', 'fist', 'point'):
            P[k] = hand_mesh(k, hc)
            P[k + '_R'] = mirror_x(P[k])
        P['thigh'] = capsule(0.43, 0.086, 0.060, legs_c, n=20, **lg)
        P['shin'] = capsule(0.42, 0.060, 0.044, legs_c, n=20, **lg)
        boot = loft([(-0.08, 0.035, 0.03), (-0.04, 0.047, 0.05), (0.06, 0.048, 0.045), (0.15, 0.042, 0.035),
                     (0.19, 0.03, 0.022)], n=16, color=self.boots, power=2.4)
        boot = Mesh(np.stack([boot.P[..., 0], boot.P[..., 2], boot.P[..., 1]], -1)[:, ::-1],
                    np.stack([boot.N[..., 0], boot.N[..., 2], boot.N[..., 1]], -1)[:, ::-1],
                    boot.UV[:, ::-1], boot.C, boot.tex, boot.emis, boot.spec)
        P['foot'] = merge([boot.xform(T(0, -0.04, 0.0)),
                           loft([(0, 0.05, 0.05), (0.08, 0.046, 0.046)], n=16, color=self.boots).xform(T(0, -0.04, -0.01))])
        self._cache[key] = P
        return P


POSE0 = dict(pelvis=(0, 0, 0), spine=(0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0),
             shoulderL=(0, 0, 0.12), shoulderR=(0, 0, 0.12), elbowL=(-0.15, 0, 0), elbowR=(-0.15, 0, 0),
             wristL=(0, 0, 0), wristR=(0, 0, 0), hipL=(0, 0, 0.03), hipR=(0, 0, 0.03),
             kneeL=(0, 0, 0), kneeR=(0, 0, 0), ankleL=(0, 0, 0), ankleR=(0, 0, 0),
             handL='hand', handR='hand')


def character(look, tex_ix, root, pose, blink=False):
    """root: 4x4 world matrix of the feet origin. pose: dict of euler angles."""
    p = dict(POSE0); p.update(pose)
    P = look.parts(tex_ix, blink)
    out = []

    def rot(name, mirror=False):
        x, y, z = p[name]
        return R(x, -y if mirror else y, -z if mirror else z)

    pel = root @ T(0, 0.99 + p.get('lift', 0.0), 0) @ rot('pelvis')
    out.append(P['pelvis'].xform(pel))
    ch = pel @ T(0, 0.06, 0) @ rot('spine')
    out.append(P['torso'].xform(ch))
    nk = ch @ T(0, 0.47, 0.0) @ rot('neck')
    out.append(P['collar'].xform(ch @ T(0, 0.46, 0)))
    out.append(P['neck'].xform(nk))
    if look.photo is not None:
        hd = nk @ T(0, 0.10, 0.03) @ rot('head') @ S(1.08)
    else:
        hd = nk @ T(0, 0.06 + HEAD_RY * 0.85, 0.012) @ rot('head') @ S(1.08)
        for s in (-1, 1):
            out.append(P['ear'].xform(hd @ T(s * 0.079, 0.0, -0.005)))
    out.append(P['head'].xform(hd))
    out.append(P['hair'].xform(hd))
    for side, s in (('L', 1), ('R', -1)):
        sh = ch @ T(s * 0.163, 0.405, 0) @ rot('shoulder' + side, s < 0)
        out.append(P['uarm'].xform(sh))
        out.append(P['deltoid'].xform(sh @ T(0, -0.01, 0)))
        el = sh @ T(0, -0.29, 0) @ rot('elbow' + side, s < 0)
        out.append(P['farm'].xform(el))
        wr = el @ T(0, -0.25, 0) @ rot('wrist' + side, s < 0)
        hand = P[p['hand' + side] + ('' if s > 0 else '_R')]
        out.append(hand.xform(wr))
        hp = pel @ T(s * 0.085, -0.05, 0) @ rot('hip' + side, s < 0)
        out.append(P['thigh'].xform(hp))
        kn = hp @ T(0, -0.43, 0) @ rot('knee' + side, s < 0)
        out.append(P['shin'].xform(kn))
        an = kn @ T(0, -0.42, 0) @ rot('ankle' + side, s < 0)
        out.append(P['foot'].xform(an))
    return merge(out)


# ---------------------------------------------------------------- RB8
def _wheel(r, w, ring):
    tyre = loft([(-w / 2, r * 0.97, r * 0.97), (-w / 2 + 0.02, r, r), (w / 2 - 0.02, r, r),
                 (w / 2, r * 0.97, r * 0.97)], n=14, color=(0.06, 0.06, 0.065), smooth=True)
    tyre.spec[:] = 0.15
    rims = []
    for s in (-1, 1):
        stripe = loft([(s * (w / 2 + 0.001), r * 0.82, r * 0.82), (s * (w / 2 + 0.001), r * 0.77, r * 0.77)],
                      n=14, color=ring, caps=False)
        rims.append(stripe)
        rims.append(loft([(s * (w / 2 - 0.005), r * 0.6, r * 0.6), (s * (w / 2 - 0.004), r * 0.58, r * 0.58)],
                         n=10, color=(0.12, 0.12, 0.13), caps=True))
    return merge([tyre] + rims).xform(Rz(np.pi / 2))


def rb8(tex_ix):
    navy = (0.07, 0.09, 0.25)
    yellow = (1.0, 0.78, 0.05)
    red = (0.85, 0.08, 0.14)
    purple = (0.42, 0.14, 0.52)
    parts = []
    # monocoque + engine cover lofted along z (sections: z, half-width, height, centre-y)
    secs = [(2.45, 0.05, 0.035, 0.30), (2.1, 0.10, 0.07, 0.33), (1.6, 0.14, 0.10, 0.42),
            (1.25, 0.17, 0.13, 0.50), (1.15, 0.24, 0.17, 0.47), (0.7, 0.30, 0.20, 0.45),
            (0.15, 0.30, 0.19, 0.44), (-0.2, 0.26, 0.30, 0.55), (-0.6, 0.22, 0.32, 0.55),
            (-1.2, 0.16, 0.22, 0.45), (-1.7, 0.10, 0.12, 0.38), (-1.95, 0.07, 0.08, 0.36)]
    body = loft([(z, w, h, 0, 0) for z, w, h, cy in secs], n=10, color=navy, power=3.0,
                deform=None, smooth=True)
    # loft builds along +y: remap (x, y, z) -> (x, cy + z_local, y) per section
    P = body.P.copy()
    zs = np.array([s[0] for s in secs]); cys = np.array([s[3] for s in secs])
    yv = P[..., 1]
    cy = np.interp(yv, zs[::-1], cys[::-1])
    newP = np.stack([P[..., 0], cy + P[..., 2], P[..., 1]], -1)
    N = body.N.copy(); newN = np.stack([N[..., 0], N[..., 2], N[..., 1]], -1)
    # axis swap mirrors handedness -> flip winding
    body = Mesh(newP[:, ::-1], newN[:, ::-1], body.UV[:, ::-1], body.C, body.tex, body.emis, body.spec)
    body.spec[:] = 0.6
    nose_tip = body.P[..., 2].max(1) > 2.0
    body.C[nose_tip] = yellow
    parts.append(body)
    # cockpit opening
    parts.append(box(0.34, 0.02, 0.7, (0.02, 0.02, 0.02)).xform(T(0, 0.66, 0.05)))
    # roll hoop / airbox
    parts.append(loft([(0, 0.13, 0.09), (0.16, 0.10, 0.08), (0.2, 0.07, 0.06)], n=8, color=navy)
                 .xform(T(0, 0.78, -0.35) @ Rx(-0.15)))
    parts.append(box(0.10, 0.07, 0.03, (0.02, 0.02, 0.02)).xform(T(0, 0.92, -0.26)))
    # shark fin
    parts.append(from_indexed([(0, 0.95, -0.4), (0, 0.98, -1.5), (0, 0.7, -1.7), (0, 0.72, -0.5)],
                              [(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)], navy))
    # sidepods
    for s in (-1, 1):
        pod = loft([(0.9, 0.17, 0.2), (0.5, 0.19, 0.22), (-0.3, 0.15, 0.18), (-1.0, 0.06, 0.08)],
                   n=8, color=navy, power=3.0)
        Pp = pod.P; Np = pod.N
        newP = np.stack([Pp[..., 0], 0.38 + Pp[..., 2] * 0.9, Pp[..., 1]], -1)
        newN = np.stack([Np[..., 0], Np[..., 2], Np[..., 1]], -1)
        pod = Mesh(newP[:, ::-1], newN[:, ::-1], pod.UV[:, ::-1], pod.C, pod.tex, pod.emis, pod.spec)
        pod.spec[:] = 0.6
        parts.append(pod.xform(T(s * 0.48, 0, 0)))
        parts.append(box(0.012, 0.12, 0.9, purple).xform(T(s * 0.675, 0.40, -0.1)))
        # red bull logo on engine cover
        logo = quad(0.62, 0.30, (1, 1, 1), tex=tex_ix['rb_logo'])
        parts.append(logo.xform(T(s * 0.235, 0.66, -0.75) @ Ry(s * np.pi / 2) @ Rx(-0.0)))
        # endplates
        parts.append(box(0.015, 0.22, 0.45, navy).xform(T(s * 0.86, 0.18, 2.25)))
        parts.append(box(0.015, 0.55, 0.55, navy).xform(T(s * 0.5, 0.75, -2.05)))
        # wishbones
        parts.append(box(0.55, 0.02, 0.04, (0.05, 0.05, 0.06)).xform(T(s * 0.42, 0.33, 1.7)))
        parts.append(box(0.55, 0.02, 0.04, (0.05, 0.05, 0.06)).xform(T(s * 0.42, 0.33, -1.6)))
        # wheels
        parts.append(_wheel(0.33, 0.30, (0.15, 0.75, 0.25)).xform(T(s * 0.72, 0.33, 1.7)))
        parts.append(_wheel(0.33, 0.38, (0.15, 0.75, 0.25)).xform(T(s * 0.72, 0.33, -1.6)))
    # front wing
    fw = box(1.65, 0.025, 0.32, navy).xform(T(0, 0.10, 2.3)); fw.spec[:] = 0.5
    parts.append(fw)
    parts.append(box(1.5, 0.02, 0.16, red).xform(T(0, 0.15, 2.22) @ Rx(-0.35)))
    # rear wing
    parts.append(box(0.98, 0.03, 0.30, navy).xform(T(0, 0.98, -2.1) @ Rx(0.15)))
    parts.append(box(0.98, 0.02, 0.14, red).xform(T(0, 0.88, -2.0) @ Rx(0.4)))
    parts.append(box(0.18, 0.4, 0.08, navy).xform(T(0, 0.55, -2.0)))
    # nose number
    parts.append(quad(0.16, 0.16, (1, 1, 1), tex=tex_ix['num1']).xform(T(0, 0.585, 1.4) @ Rx(-1.35)))
    return merge(parts)


# ---------------------------------------------------------------- lamp
def theatre_lamp(pivot, aim, on=1.0):
    """Fresnel fixture hanging from a pipe. Returns (mesh, lens_centre).

    Built with the lens at the origin facing -y, the yoke pivot 0.3 above it.
    """
    body_c = (0.10, 0.10, 0.11)
    fx = []
    body = loft([(0.0, 0.20, 0.20), (0.05, 0.22, 0.22), (0.45, 0.22, 0.22), (0.55, 0.16, 0.16)],
                n=12, color=body_c)
    body.spec[:] = 0.4
    fx.append(body)
    for k in range(4):   # cooling fins, leaking a little light from inside the housing
        fin = box(0.47, 0.012, 0.47, (1.0, 0.8, 0.55)).xform(T(0, 0.12 + k * 0.09, 0))
        fx.append(fin.mat(emis=0.08 * on))
    hot = (1.0, 0.92, 0.75) if on > 0.5 else (1.0, 0.42, 0.12)   # cooling filament
    lens = loft([(-0.012, 0.175, 0.175), (-0.002, 0.18, 0.18)], n=12, color=hot)
    lens.emis[:] = 25.0 * on + 0.02
    fx.append(lens)
    for k in range(4):   # barn doors
        door = box(0.40, 0.012, 0.20, body_c).xform(Rx(0.45) @ T(0, 0, 0.10))
        fx.append(door.xform(Ry(k * np.pi / 2) @ T(0, -0.01, 0.21)))
    fx = merge(fx)
    aim = norm(np.asarray(aim, float))
    yaw = np.arctan2(aim[0], aim[2])
    h = np.hypot(aim[0], aim[2])
    tilt = np.arctan2(-h, -aim[1])
    M = T(*pivot) @ Ry(yaw) @ Rx(tilt) @ T(0, -0.3, 0)
    yoke = []
    for s in (-1, 1):
        yoke.append(box(0.03, 0.08, 0.06, body_c).xform(Rx(tilt) @ T(s * 0.25, 0, 0)))
        yoke.append(box(0.03, 0.35, 0.06, body_c).xform(T(s * 0.25, 0.17, 0)))
    yoke.append(box(0.53, 0.03, 0.06, body_c).xform(T(0, 0.35, 0)))
    yoke.append(box(0.04, 5.0, 0.04, (0.08, 0.08, 0.08)).xform(T(0, 2.85, 0)))
    yoke = merge(yoke).xform(T(*pivot) @ Ry(yaw))
    lens_c = (M @ np.array([0, -0.02, 0, 1.0]))[:3]
    return merge([fx.xform(M), yoke]), lens_c
