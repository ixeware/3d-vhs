"""Procedurally painted low-res textures (faces, race suits, livery, floor)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HEAD_RY = 0.115          # must match the head mesh in characters.py
SS = 4                   # supersampling for painting


class Canvas:
    """Paint in (angle_deg, height_m) space of a cylindrical/spherical UV map."""

    def __init__(self, w, h, base, y0=-HEAD_RY, y1=HEAD_RY):
        self.w, self.h, self.y0, self.y1 = w, h, y0, y1
        self.im = Image.new('RGB', (w * SS, h * SS), rgb(base))
        self.d = ImageDraw.Draw(self.im, 'RGBA')

    def p(self, ang, y):
        return ((0.5 + ang / 360.0) * self.w * SS,
                (1 - (y - self.y0) / (self.y1 - self.y0)) * self.h * SS)

    def ellipse(self, ang, y, rang, ry, fill, alpha=255):
        x0, y0 = self.p(ang - rang, y + ry); x1, y1 = self.p(ang + rang, y - ry)
        self.d.ellipse([x0, y0, x1, y1], fill=rgb(fill) + (alpha,))

    def poly(self, pts, fill, alpha=255):
        self.d.polygon([self.p(a, y) for a, y in pts], fill=rgb(fill) + (alpha,))

    def line(self, pts, fill, width=1.0, alpha=255):
        self.d.line([self.p(a, y) for a, y in pts], fill=rgb(fill) + (alpha,),
                    width=max(1, int(width * SS)), joint='curve')

    def rect(self, a0, y0, a1, y1, fill, alpha=255):
        x0, yy0 = self.p(a0, y1); x1, yy1 = self.p(a1, y0)
        self.d.rectangle([x0, yy0, x1, yy1], fill=rgb(fill) + (alpha,))

    def blur(self, r):
        self.im = self.im.filter(ImageFilter.GaussianBlur(r * SS))
        self.d = ImageDraw.Draw(self.im, 'RGBA')

    def done(self):
        im = self.im.resize((self.w, self.h), Image.BOX)
        return np.asarray(im, np.float32) / 255.0


def rgb(c):
    return tuple(int(max(0, min(1, v)) * 255) for v in c)


def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


# ---------------------------------------------------------------- faces
def face(skin, hair, brow, iris, hairline=0.062, stubble=0.0, mouth='neutral',
         blink=False, lid=0.35, brow_angle=8.0, sideburn=0.0):
    c = Canvas(256, 128, skin)
    dark = mix(skin, (0.25, 0.12, 0.08), 0.35)
    # soft shading: cheeks / temples / under-jaw
    c.ellipse(-48, -0.03, 18, 0.05, dark, 60); c.ellipse(48, -0.03, 18, 0.05, dark, 60)
    c.rect(-180, -0.2, 180, -0.085, mix(skin, (0.2, 0.1, 0.08), 0.5), 255)
    c.blur(1.2)
    # stubble / beard shadow
    if stubble > 0:
        st = mix(skin, (0.18, 0.13, 0.11), 0.85)
        a = int(255 * stubble)
        c.poly([(-75, -0.01), (-58, -0.065), (-30, -0.095), (0, -0.11), (30, -0.095), (58, -0.065),
                (75, -0.01), (62, -0.035), (40, -0.05), (22, -0.05), (12, -0.04), (0, -0.038),
                (-12, -0.04), (-22, -0.05), (-40, -0.05), (-62, -0.035)], st, a)
        c.poly([(-14, -0.037), (14, -0.037), (12, -0.05), (-12, -0.05)], st, a)
        c.blur(0.6)
    # hair: cap + sides + back
    hl = hairline
    c.poly([(-180, 0.2), (180, 0.2), (180, -0.06), (110, -0.05), (85, -0.01 - sideburn),
            (78, 0.02), (62, 0.04), (40, hl - 0.004), (15, hl + 0.004), (0, hl),
            (-15, hl + 0.004), (-40, hl - 0.004), (-62, 0.04), (-78, 0.02),
            (-85, -0.01 - sideburn), (-110, -0.05), (-180, -0.06)], hair)
    # ears
    for s in (-1, 1):
        c.ellipse(s * 90, -0.005, 7, 0.028, mix(skin, (0.6, 0.2, 0.15), 0.25))
    # brows
    for s in (-1, 1):
        a0, a1 = s * 9, s * 36
        c.poly([(a0, 0.033 - brow_angle * 0.0006), (a1, 0.040), (a1, 0.033),
                (a0, 0.025 - brow_angle * 0.0006)], brow)
    # eyes
    for s in (-1, 1):
        ex = s * 22
        if blink:
            c.line([(ex - 10, 0.012), (ex + 10, 0.010)], (0.15, 0.08, 0.06), 1.4)
        else:
            c.poly([(ex - 11, 0.012), (ex - 4, 0.021), (ex + 5, 0.021), (ex + 11, 0.013),
                    (ex + 4, 0.004), (ex - 5, 0.004)], (0.86, 0.82, 0.78))
            c.ellipse(ex, 0.012, 4.6, 0.0085, iris)
            c.ellipse(ex, 0.012, 2.1, 0.0042, (0.02, 0.02, 0.02))
            c.ellipse(ex + 1.5, 0.015, 0.9, 0.0018, (0.95, 0.95, 0.95))
            # heavy upper lid -> the stare
            top = 0.021 - lid * 0.008
            c.poly([(ex - 12, 0.013), (ex - 4, 0.023), (ex + 5, 0.023), (ex + 12, 0.014),
                    (ex + 5, top), (ex - 4, top)], mix(skin, (0.3, 0.15, 0.1), 0.4))
            c.line([(ex - 11, 0.012), (ex - 4, top), (ex + 5, top), (ex + 11, 0.013)],
                   (0.08, 0.05, 0.04), 1.3)
        c.line([(ex - 6, 0.001), (ex + 6, 0.001)], mix(skin, (0.3, 0.15, 0.1), 0.5), 0.8, 120)
    # nose
    c.line([(-5, -0.005), (-6, -0.03), (-3, -0.036)], mix(skin, (0.3, 0.15, 0.1), 0.5), 0.9, 160)
    c.ellipse(-3.5, -0.036, 2.0, 0.003, (0.25, 0.12, 0.1), 200)
    c.ellipse(3.5, -0.036, 2.0, 0.003, (0.25, 0.12, 0.1), 200)
    # mouth
    lip = (0.55, 0.30, 0.27)
    if mouth == 'neutral':
        c.poly([(-12, -0.058), (0, -0.056), (12, -0.058), (0, -0.064)], lip)
        c.line([(-12, -0.059), (0, -0.058), (12, -0.060)], (0.22, 0.1, 0.08), 1.1)
    elif mouth == 'open':
        c.ellipse(0, -0.062, 11, 0.012, (0.25, 0.05, 0.05))
        c.rect(-8, -0.056, 8, -0.052, (0.92, 0.9, 0.85))
    elif mouth == 'smile':
        c.poly([(-15, -0.053), (0, -0.058), (15, -0.053), (8, -0.066), (-8, -0.066)], (0.25, 0.05, 0.05))
        c.poly([(-11, -0.055), (0, -0.059), (11, -0.055), (9, -0.058), (-9, -0.058)], (0.95, 0.92, 0.88))
    return c.done()


# ---------------------------------------------------------------- suits
def suit(kind):
    """Torso texture, cylindrical: angle 0 = chest, +-180 = back. y in [0,1]."""
    if kind == 'ferrari':
        base, trim = (0.62, 0.03, 0.03), (0.97, 0.95, 0.9)
    elif kind == 'redbull':
        base, trim = (0.07, 0.09, 0.24), (0.86, 0.1, 0.18)
    else:  # mechanic shirt
        base, trim = (0.08, 0.10, 0.26), (0.85, 0.12, 0.16)
    c = Canvas(128, 128, base, 0.0, 1.0)
    # side stripes
    for s in (-1, 1):
        c.rect(s * 90 - 6, 0.0, s * 90 + 6, 1.0, trim)
    # belt
    c.rect(-180, 0.0, 180, 0.10, (0.06, 0.06, 0.07))
    c.rect(-180, 0.10, 180, 0.13, trim)
    # collar
    c.rect(-180, 0.93, 180, 1.0, (0.05, 0.05, 0.06))
    if kind == 'ferrari':
        c.poly([(16, 0.80), (32, 0.80), (32, 0.73), (24, 0.68), (16, 0.73)], (1.0, 0.85, 0.0))  # shield
        c.rect(22.5, 0.72, 25.5, 0.78, (0.05, 0.05, 0.05))
        c.rect(-36, 0.73, -14, 0.79, (0.97, 0.97, 0.97))            # sponsor tab
        c.rect(-34, 0.748, -16, 0.772, (0.85, 0.02, 0.02))
        c.rect(-48, 0.40, 48, 0.47, (0.97, 0.97, 0.97))             # belly sponsor band
        c.rect(-40, 0.425, 40, 0.445, (0.85, 0.02, 0.02))
        c.rect(-180, 0.3, -150, 0.4, trim); c.rect(150, 0.3, 180, 0.4, trim)
    else:
        big = kind == 'redbull'
        y = 0.70 if big else 0.72
        r = 22 if big else 14
        c.ellipse(0, y, r * 0.55, 0.06 * r / 22, (1.0, 0.82, 0.05))      # sun
        c.ellipse(-r * 0.55, y - 0.01, r * 0.5, 0.035 * r / 22, (0.88, 0.08, 0.15))  # bulls
        c.ellipse(r * 0.55, y - 0.01, r * 0.5, 0.035 * r / 22, (0.88, 0.08, 0.15))
        c.rect(-40, 0.50, 40, 0.53, (0.55, 0.2, 0.6))             # purple stripe
        if not big:
            c.rect(-180, 0.4, 180, 0.43, trim)
    return c.done()


def flat(color, w=8, h=8):
    im = np.empty((h, w, 3), np.float32); im[:] = color
    return im


def concrete(seed=3):
    rng = np.random.default_rng(seed)
    n = rng.random((64, 64)) * 0.25
    big = np.kron(rng.random((8, 8)), np.ones((8, 8))) * 0.2
    v = 0.55 + n + big
    im = np.stack([v * 0.95, v * 0.93, v * 0.9], -1)
    im[::32, :] *= 0.5; im[:, ::32] *= 0.5   # slab seams
    return im.astype(np.float32)


def decal_number(txt, fg, bg, w=32, h=32):
    from PIL import ImageFont
    im = Image.new('RGB', (w * SS, h * SS), rgb(bg))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(h * SS * 0.8))
    except OSError:
        f = ImageFont.load_default()
    d.text((w * SS / 2, h * SS / 2), txt, fill=rgb(fg), font=f, anchor='mm')
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def redbull_logo(w=64, h=32):
    c = Canvas(w, h, (0.08, 0.10, 0.26), 0.0, 1.0)
    c.ellipse(0, 0.55, 40, 0.3, (1.0, 0.82, 0.05))
    c.poly([(-170, 0.2), (-60, 0.75), (-20, 0.55), (-30, 0.2)], (0.9, 0.08, 0.15))
    c.poly([(170, 0.2), (60, 0.75), (20, 0.55), (30, 0.2)], (0.9, 0.08, 0.15))
    return c.done()


def build_all():
    """Texture table. Returns (list, name->index)."""
    T = {}
    al = dict(skin=(0.80, 0.60, 0.47), hair=(0.09, 0.06, 0.05), brow=(0.07, 0.05, 0.04),
              iris=(0.25, 0.14, 0.07), stubble=0.28, lid=0.45, brow_angle=10, sideburn=0.012)
    T['alonso'] = face(**al)
    T['alonso_blink'] = face(blink=True, **al)
    T['vettel'] = face(skin=(0.93, 0.76, 0.66), hair=(0.80, 0.64, 0.38), brow=(0.55, 0.40, 0.25),
                       iris=(0.30, 0.42, 0.50), stubble=0.25, mouth='open', lid=0.0,
                       brow_angle=-6, hairline=0.07)
    mech = [((0.86, 0.68, 0.56), (0.30, 0.20, 0.12)), ((0.62, 0.44, 0.32), (0.06, 0.05, 0.05)),
            ((0.92, 0.75, 0.65), (0.75, 0.55, 0.30)), ((0.45, 0.30, 0.22), (0.05, 0.04, 0.04)),
            ((0.84, 0.66, 0.55), (0.45, 0.30, 0.18))]
    for i, (sk, hr) in enumerate(mech):
        T[f'mech{i}'] = face(skin=sk, hair=hr, brow=mix(hr, (0, 0, 0), 0.4), iris=(0.2, 0.15, 0.1),
                             stubble=0.3 * (i % 2), mouth='open' if i % 2 == 0 else 'smile',
                             lid=0.0, brow_angle=-8)
    T['suit_ferrari'] = suit('ferrari')
    T['suit_redbull'] = suit('redbull')
    T['suit_mech'] = suit('mech')
    T['concrete'] = concrete()
    T['num1'] = decal_number('1', (0.95, 0.95, 0.95), (0.08, 0.10, 0.26))
    T['rb_logo'] = redbull_logo()
    names = list(T)
    return [T[k] for k in names], {k: i for i, k in enumerate(names)}
