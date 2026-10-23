"""Locations: a foggy Silent Hill style street, and its rusted Otherworld."""
import numpy as np
from engine import T, S, Rx, Ry, Rz, Mesh, merge, box, tbox, loft, quad, grid, norm, from_indexed
from models import _wheel


def beam(p0, p1, w, color, **kw):
    """Thin box from p0 to p1 (wires, poles, arms)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    z = d / L
    x = norm(np.cross((0, 1, 0) if abs(z[1]) < 0.9 else (1, 0, 0), z))
    y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2] = x, y, z; M[:3, 3] = (p0 + p1) / 2
    return box(w, w, L, color, **kw).xform(M)


def two_sided(m):
    return merge([m, m.xform(Ry(np.pi))])


def panel(w, h, tex, tile, color=(1, 1, 1)):
    """Double-sided alpha-tested quad, bottom edge on y=0, spanning x."""
    q = quad(w, h, color, uv=((0, 0), (w / tile, 0), (w / tile, h / tile), (0, h / tile)), tex=tex)
    return two_sided(q.xform(T(0, h / 2, 0)))


def fence(p0, p1, h, tex, post_col):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(d[0], d[2])
    yaw = np.arctan2(d[0], d[2]) - np.pi / 2
    parts = [panel(L, h, tex, 0.5).xform(T(*(p0 + p1) / 2) @ Ry(yaw))]
    n = max(1, int(L / 2.5))
    for k in range(n + 1):
        p = p0 + d * k / n
        parts.append(tbox(0.06, h + 0.1, 0.06, post_col).xform(T(p[0], (h + 0.1) / 2, p[2])))
    parts.append(beam(p0 + (0, h, 0), p1 + (0, h, 0), 0.04, post_col))
    return merge(parts)


def sag_wire(p0, p1, sag, color, segs=4):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    pts = [p0 + (p1 - p0) * s - np.array([0, sag * 4 * s * (1 - s), 0]) for s in np.linspace(0, 1, segs + 1)]
    return merge([beam(pts[i], pts[i + 1], 0.025, color) for i in range(segs)])


# ---------------------------------------------------------------- street furniture
def frustum(wb, db, wt, dt, h, color, **kw):
    """Box whose top face is smaller than its bottom (car cabins, hydrant caps)."""
    V = [(-wb / 2, 0, -db / 2), (wb / 2, 0, -db / 2), (wb / 2, 0, db / 2), (-wb / 2, 0, db / 2),
         (-wt / 2, h, -dt / 2), (wt / 2, h, -dt / 2), (wt / 2, h, dt / 2), (-wt / 2, h, dt / 2)]
    F = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
         (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return from_indexed(V, F, color, **kw)


def sedan(paint):
    """Boxy late-80s sedan left to rot at the kerb. Nose points +z."""
    glass = (0.04, 0.045, 0.05)
    P = [tbox(1.76, 0.58, 4.5, paint).xform(T(0, 0.58, 0)).mat(spec=0.35),
         frustum(1.62, 2.5, 1.38, 1.7, 0.5, glass).xform(T(0, 0.87, -0.25)).mat(spec=0.8),
         box(1.4, 0.05, 1.72, paint).xform(T(0, 1.38, -0.25))]
    for z in (2.27, -2.27):
        P.append(box(1.8, 0.16, 0.1, (0.32, 0.32, 0.33)).xform(T(0, 0.42, z)))           # bumpers
    for x in (-0.6, 0.6):
        P.append(box(0.36, 0.14, 0.02, (0.75, 0.72, 0.6)).xform(T(x, 0.68, 2.26)))       # headlamps
        P.append(box(0.36, 0.12, 0.02, (0.35, 0.04, 0.03)).xform(T(x, 0.72, -2.26)))     # tail lamps
    for x in (-0.8, 0.8):
        for z in (1.45, -1.45):
            P.append(_wheel(0.31, 0.2, (0.2, 0.2, 0.2)).xform(T(x, 0.31, z)))
    return merge(P)


def blade_sign(tex, w, h, frame, emis=0.0):
    """Projecting sign, readable from both directions along the street."""
    face = quad(w, h, (1, 1, 1), tex=tex, emis=emis)
    return merge([box(w + 0.08, h + 0.08, 0.06, frame),
                  face.xform(T(0, 0, 0.035)), face.xform(Ry(np.pi) @ T(0, 0, 0.035))])


def traffic_light(amber):
    house = (0.30, 0.28, 0.12)
    P = [box(0.36, 1.0, 0.3, house)]
    for k, (col, e) in enumerate((((0.5, 0.05, 0.03), 0.05), ((1.0, 0.65, 0.1), 14.0 * amber + 0.05),
                                  ((0.05, 0.4, 0.2), 0.05))):
        lens = box(0.2, 0.2, 0.04, col).mat(emis=e)
        P.append(lens.xform(T(0, 0.3 - k * 0.3, 0.17)))
        P.append(lens.xform(T(0, 0.3 - k * 0.3, -0.17)))
        P.append(box(0.26, 0.04, 0.12, house).xform(T(0, 0.42 - k * 0.3, 0.2)))          # visors
    P.append(box(0.04, 0.5, 0.04, (0.1, 0.1, 0.1)).xform(T(0, 0.75, 0)))
    return merge(P)


def street_lamp(pos, level, side=1):
    """Old sodium street lamp; the bulb is emissive only (it lights nothing but the fog)."""
    x, z = pos
    pole = merge([box(0.14, 6.0, 0.14, (0.18, 0.18, 0.19)).xform(T(x, 3.0, z)),
                  beam((x, 5.9, z), (x + side * 1.2, 6.15, z), 0.08, (0.18, 0.18, 0.19)),
                  box(0.45, 0.12, 0.25, (0.14, 0.14, 0.15)).xform(T(x + side * 1.3, 6.1, z))])
    bulb = box(0.32, 0.04, 0.16, (1.0, 0.62, 0.25)).xform(T(x + side * 1.3, 6.03, z)).mat(emis=30.0 * level + 0.05)
    return merge([pole, bulb])


def building(tx, x_face, side, z0, z1, depth, h, shop, upper, ground_h=3.5):
    """Two-part building: shopfront ground floor + upper floors. Faces the road."""
    d = z0 - z1
    cx = x_face + side * depth / 2
    cz = (z0 + z1) / 2
    P = [tbox(depth, ground_h, d, (1, 1, 1), tile=4.0, tex=tx[shop]).xform(T(cx, ground_h / 2, cz)),
         tbox(depth, h - ground_h, d, (1, 1, 1), tile=4.0, tex=tx[upper]).xform(
             T(cx, ground_h + (h - ground_h) / 2, cz)),
         box(depth + 0.2, 0.25, d + 0.2, (0.2, 0.19, 0.18)).xform(T(cx, h + 0.12, cz)),
         box(0.25, 0.18, d, (0.22, 0.21, 0.2)).xform(T(x_face - side * 0.1, ground_h + 0.05, cz))]
    return merge(P)


# ---------------------------------------------------------------- fog world
EDGE_Z = -7.0


def edge_z(x):
    """Where the road has torn away into the abyss."""
    return EDGE_Z + 0.6 * np.sin(1.3 * x + 0.5) + 0.35 * np.sin(3.7 * x)


def town(tx, t=0.0):
    """South Vale, Silent Hill: Lindsey St runs along z and falls into the abyss behind Alonso.

    Alonso stands on the road at the origin; the camera is on +z looking -z.
    """
    P = []
    g = grid(24, 24, (0.62, 0.62, 0.64), tile=3.0, tex=tx['asphalt']).xform(T(0, 0, 1.0))
    c = g.P.mean(1)
    keep = c[:, 2] > edge_z(c[:, 0])
    P.append(Mesh(g.P[keep], g.N[keep], g.UV[keep], g.C[keep], g.tex[keep], g.emis[keep], g.spec[keep]))
    rng = np.random.default_rng(9)
    for x in np.arange(-4.5, 5.0, 1.6):                               # slabs breaking off the lip
        z = edge_z(x) - 0.2
        slab = tbox(1.3, 0.25, 1.1, (0.62, 0.62, 0.64), tile=3.0, tex=tx['asphalt'])
        P.append(slab.xform(T(x + rng.uniform(-0.3, 0.3), -0.35 - rng.uniform(0, 0.5), z - 0.4)
                            @ Rx(rng.uniform(0.35, 0.8)) @ Rz(rng.uniform(-0.2, 0.2))))
    for z in np.arange(EDGE_Z + 1.0, 10.0, 3.0):                      # faded centre dashes
        P.append(box(0.12, 0.01, 1.3, (0.55, 0.5, 0.3)).xform(T(0.0, 0.004, z + 0.5)))
    for s in (-1, 1):
        P.append(tbox(2.9, 0.16, 17.0, (0.7, 0.7, 0.7), tile=1.5, tex=tx['concrete']).xform(T(s * 5.45, 0.08, 1.5)))

    # left: an old inn, the alley, Neely's Bar
    P.append(building(tx, -6.9, -1, 3.0, -2.6, 5.0, 7.0, 'shop_cafe', 'brick'))
    P.append(fence((-7.0, 0.16, -2.6), (-7.0, 0.16, -3.4), 2.4, tx['chainlink'], (0.35, 0.34, 0.33)))
    P.append(building(tx, -6.9, -1, -3.4, -7.6, 5.0, 8.0, 'shop_neely', 'brick_dark'))
    P.append(quad(4.0, 0.7, (1, 1, 1), tex=tx['sign_neely_h'], emis=0.35).xform(
        T(-6.86, 3.95, -5.5) @ Ry(np.pi / 2)))
    P.append(blade_sign(tx['sign_neely'], 1.4, 0.98, (0.12, 0.1, 0.08), emis=0.6).xform(T(-5.9, 3.7, -3.75)))
    P.append(building(tx, -6.9, -1, 10.0, 3.6, 5.0, 6.5, 'shop_heaven', 'facade1'))

    # right: Cafe 5to2 with its awning, Heaven's Night buzzing pink at the end of the block
    P.append(building(tx, 6.9, 1, 2.0, -3.6, 5.0, 6.0, 'shop_cafe', 'facade3'))
    P.append(tbox(1.3, 0.08, 5.4, (1, 1, 1), tile=1.0, tex=tx['awning']).xform(T(6.3, 3.05, -0.8) @ Rz(-0.35)))
    P.append(quad(3.2, 0.8, (1, 1, 1), tex=tx['sign_cafe'], emis=0.35).xform(T(6.86, 3.85, -0.8) @ Ry(-np.pi / 2)))
    P.append(blade_sign(tx['sign_cafe'], 1.7, 0.42, (0.15, 0.12, 0.1), emis=0.6).xform(T(5.9, 4.0, -3.3)))
    P.append(building(tx, 6.9, 1, -4.2, -7.6, 5.0, 10.0, 'shop_heaven', 'facade2'))
    neon = 0.25 + 5.0 * (np.random.default_rng(int(t * 12) + 3).random() > 0.18)   # buzzing tube
    P.append(blade_sign(tx['sign_heaven'], 1.6, 0.95, (0.1, 0.05, 0.1), emis=neon).xform(T(5.85, 4.3, -5.4)))
    P.append(building(tx, 6.9, 1, 10.0, 2.6, 5.0, 7.5, 'shop_neely', 'facade2'))

    # wooden utility poles, wires, and a traffic light swinging over the road
    wood = dict(tex=tx['wood'])
    poles = [(4.4, 8.0), (4.4, -2.4), (-4.5, 0.2), (-4.5, 6.5), (4.5, -6.3)]
    for x, z in poles:
        P.append(tbox(0.24, 8.5, 0.24, (1, 1, 1), tile=8.5, **wood).xform(T(x, 4.25, z)))
        P.append(box(1.8, 0.12, 0.12, (0.25, 0.2, 0.16)).xform(T(x, 7.9, z)))
    for (x0, z0), (x1, z1) in (((4.4, 8.0), (4.4, -2.4)), ((4.4, -2.4), (4.5, -6.3)), ((-4.5, 6.5), (-4.5, 0.2))):
        for dx in (-0.8, 0.0, 0.8):
            P.append(sag_wire((x0 + dx, 7.95, z0), (x1 + dx, 7.95, z1), 0.8, (0.05, 0.05, 0.05)))
    P.append(sag_wire((4.4, 7.2, -2.4), (-4.5, 7.2, 0.2), 0.9, (0.05, 0.05, 0.05)))
    amber = 1.0 if (t % 1.1) < 0.55 else 0.0
    sway = 0.05 * np.sin(t * 1.3)
    P.append(traffic_light(amber).xform(T(0.0, 6.35, -1.1) @ Rz(sway) @ Ry(0.28)))
    # LINDSEY ST / KATZ ST plates on the corner pole
    P.append(blade_sign(tx['street_sign'], 1.05, 0.22, (0.1, 0.1, 0.1), emis=0.2).xform(
        T(4.4, 2.9, -2.4) @ Ry(np.pi / 2)))
    P.append(blade_sign(tx['street_sign2'], 1.05, 0.22, (0.1, 0.1, 0.1), emis=0.2).xform(
        T(4.4, 3.16, -2.4)))

    # rotting cars, one tipping over the edge
    P.append(sedan((0.55, 0.50, 0.42)).xform(T(3.55, 0, 3.4)))
    P.append(sedan((0.30, 0.07, 0.06)).xform(T(-0.9, -0.15, -7.4) @ Ry(np.pi + 0.25) @ Rx(-0.32)))

    # kerbside clutter
    P.append(loft([(0, 0.13, 0.13), (0.55, 0.11, 0.11), (0.62, 0.07, 0.07)], n=8,
                  color=(0.62, 0.1, 0.07)).xform(T(4.5, 0.16, -0.4)))                    # hydrant
    P.append(tbox(0.5, 1.05, 0.5, (0.12, 0.2, 0.42)).xform(T(-4.55, 0.68, 2.3)))           # mailbox
    P.append(loft([(0, 0.27, 0.27), (0.95, 0.3, 0.3)], n=10, color=(0.25, 0.27, 0.25)).xform(
        T(4.6, 0.16, -4.0)))                                                                   # bin
    P.append(_wheel(0.33, 0.30, (0.6, 0.6, 0.6)).xform(T(-4.3, 0.31, 0.2) @ Rz(np.pi / 2)))
    P.append(_wheel(0.33, 0.30, (0.6, 0.6, 0.6)).xform(T(-4.35, 0.61, 0.25) @ Rz(np.pi / 2) @ Ry(0.3)))

    # barricades in front of the drop
    for x, label in ((-3.4, False), (1.5, True), (3.5, False)):
        z = edge_z(x) + 1.3
        bar = merge([tbox(1.5, 0.25, 0.05, (1, 1, 1), tile=0.5, tex=tx['barricade']).xform(T(0, 0.85, 0)),
                     tbox(1.5, 0.2, 0.05, (1, 1, 1), tile=0.5, tex=tx['barricade']).xform(T(0, 0.45, 0)),
                     beam((-0.6, 0, 0.25), (-0.6, 1.0, 0), 0.06, (0.75, 0.72, 0.65)),
                     beam((-0.6, 0, -0.25), (-0.6, 1.0, 0), 0.06, (0.75, 0.72, 0.65)),
                     beam((0.6, 0, 0.25), (0.6, 1.0, 0), 0.06, (0.75, 0.72, 0.65)),
                     beam((0.6, 0, -0.25), (0.6, 1.0, 0), 0.06, (0.75, 0.72, 0.65))])
        if label:
            bar = merge([bar, quad(1.2, 0.3, (1, 1, 1), tex=tx['road_closed'], emis=0.25).xform(T(0, 1.2, 0.03)),
                         box(0.05, 0.4, 0.05, (0.3, 0.3, 0.3)).xform(T(0, 1.0, 0))])
        P.append(bar.xform(T(x, 0, z) @ Ry(0.1 * x)))
    return merge(P)


# ---------------------------------------------------------------- otherworld
def otherworld(tx, t):
    """Rusted cage around the RB8: grating floor, chain-link, hanging chains, a slow fan."""
    P = []
    P.append(grid(30, 30, (0.55, 0.5, 0.5), tile=1.0, tex=tx['rust_grate']).xform(T(0, 0, 0)))
    fl = tx['chainlink_rust']
    post = (0.22, 0.12, 0.08)
    P.append(fence((7.0, 0, 9.0), (7.0, 0, -7.0), 3.6, fl, post))
    P.append(fence((7.0, 0, -7.0), (-8.0, 0, -7.0), 3.6, fl, post))
    P.append(fence((-8.0, 0, -7.0), (-8.0, 0, 9.0), 3.6, fl, post))
    # rust walls beyond the cage
    P.append(tbox(26, 12, 0.6, (1, 1, 1), tile=4.0, tex=tx['rust_wall']).xform(T(0, 6, -10.5)))
    P.append(tbox(0.6, 12, 26, (1, 1, 1), tile=4.0, tex=tx['rust_wall']).xform(T(10.5, 6, 0)))
    P.append(tbox(0.6, 12, 26, (1, 1, 1), tile=4.0, tex=tx['rust_wall']).xform(T(-11.5, 6, 0)))
    for x, z in ((7.0, -7.0), (7.0, 3.0), (-8.0, -7.0), (-1.0, -7.0)):
        P.append(tbox(0.5, 9, 0.5, (0.8, 0.8, 0.8), tile=2.0, tex=tx['rust_wall']).xform(T(x, 4.5, z)))
    # hanging chains
    rng = np.random.default_rng(12)
    for _ in range(14):
        x, z = rng.uniform(-6.5, 6.0), rng.uniform(-6.0, 5.0)
        if abs(x) < 1.6 and -3.0 < z < 3.5:
            continue                       # keep the car and Vettel clear
        L = rng.uniform(3.0, 6.0)
        sway = 0.04 * np.sin(t * 0.9 + x)
        ch = panel(0.12, L, tx['chain'], 0.5)
        P.append(merge([ch, ch.xform(Ry(np.pi / 2))]).xform(T(x, 8.5, z) @ Rz(sway) @ T(0, -L, 0)))
    # big ventilation fan on the back wall, light spilling through it
    fx, fy = 4.0, 4.2
    glow = loft([(-0.05, 1.4, 1.4), (0.0, 1.4, 1.4)], n=16, color=(1.0, 0.55, 0.25)).mat(emis=5.0)
    P.append(glow.xform(T(fx, fy, -10.15) @ Rx(np.pi / 2)))
    ring = loft([(0, 1.6, 1.6), (0.3, 1.6, 1.6)], n=16, color=(0.12, 0.06, 0.04), caps=False)
    P.append(ring.xform(T(fx, fy, -10.2) @ Rx(np.pi / 2)))
    for k in range(4):
        blade = box(0.45, 1.4, 0.05, (0.08, 0.05, 0.04)).xform(T(0, 0.7, 0))
        P.append(blade.xform(T(fx, fy, -9.95) @ Rz(t * 1.1 + k * np.pi / 2) @ Ry(0.5)))
    P.append(box(0.4, 0.4, 0.12, (0.08, 0.05, 0.04)).xform(T(fx, fy, -9.9)))
    return merge(P)
