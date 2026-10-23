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


# ---------------------------------------------------------------- locations
def _noise(rng, h, w, octaves=((8, 0.5), (16, 0.3), (64, 0.2))):
    out = np.zeros((h, w))
    for cells, amp in octaves:
        n = rng.random((cells, cells))
        out += amp * np.kron(n, np.ones((h // cells + 1, w // cells + 1)))[:h, :w]
    return out


def asphalt(seed=21):
    rng = np.random.default_rng(seed)
    v = 0.26 + 0.2 * _noise(rng, 64, 64) + 0.1 * rng.random((64, 64))
    im = Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).convert('RGB')
    d = ImageDraw.Draw(im)
    for _ in range(6):   # cracks
        x, y = rng.integers(0, 64, 2)
        pts = [(x, y)]
        for _ in range(5):
            x += rng.integers(-9, 10); y += rng.integers(-9, 10); pts.append((x, y))
        d.line(pts, fill=(28, 28, 30), width=1)
    a = np.asarray(im, np.float32) / 255
    return a * np.array([0.95, 0.96, 1.0], np.float32)


def facade(seed, base, window_lit=False):
    """4m x 4m of wall with one window, grime running down."""
    rng = np.random.default_rng(seed)
    c = Canvas(64, 64, base, 0.0, 1.0)
    c.d.rectangle([0, 0, 64 * SS, 64 * SS], fill=rgb(base))
    im = c.im
    d = ImageDraw.Draw(im, 'RGBA')
    S_ = SS
    d.rectangle([16 * S_, 14 * S_, 48 * S_, 40 * S_], fill=(40, 38, 36))                 # frame
    glass = (120, 105, 70) if window_lit else (18, 20, 24)
    d.rectangle([18 * S_, 16 * S_, 46 * S_, 38 * S_], fill=glass)
    d.rectangle([31 * S_, 16 * S_, 33 * S_, 38 * S_], fill=(40, 38, 36))
    d.rectangle([18 * S_, 26 * S_, 46 * S_, 28 * S_], fill=(40, 38, 36))
    d.rectangle([14 * S_, 40 * S_, 50 * S_, 43 * S_], fill=tuple(int(v * 255 * 0.8) for v in base))
    for _ in range(14):   # grime streaks
        x = rng.integers(0, 64) * S_; y = rng.integers(0, 30) * S_
        d.rectangle([x, y, x + rng.integers(1, 4) * S_, y + rng.integers(10, 40) * S_],
                    fill=(20, 18, 15, int(rng.integers(30, 90))))
    d.rectangle([0, 60 * S_, 64 * S_, 64 * S_], fill=(30, 28, 26, 160))               # floor line
    a = np.asarray(im.resize((64, 64), Image.BOX), np.float32) / 255
    return a * (0.85 + 0.3 * _noise(rng, 64, 64)[..., None])


def chainlink(rust=False):
    """RGBA, 0.5m tile, diamond wire mesh."""
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    a = ((np.abs(((xx + yy) % 16) - 8) < 1.1) | (np.abs(((xx - yy) % 16) - 8) < 1.1)).astype(np.float32)
    col = np.array([0.42, 0.24, 0.14]) if rust else np.array([0.62, 0.62, 0.6])
    im = np.zeros((n, n, 4), np.float32)
    im[..., :3] = col * (0.8 + 0.4 * np.random.default_rng(3).random((n, n, 1)))
    im[..., 3] = a
    return im


def chain():
    """RGBA, one tile = 4 links, vertical."""
    im = Image.new('RGBA', (16 * SS, 64 * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for k in range(4):
        y = k * 16 * SS
        if k % 2 == 0:
            d.ellipse([3 * SS, y, 13 * SS, y + 18 * SS], outline=(110, 70, 45, 255), width=3 * SS)
        else:
            d.rectangle([6 * SS, y - 2 * SS, 10 * SS, y + 18 * SS], fill=(90, 55, 35, 255))
    a = np.asarray(im.resize((16, 64), Image.BOX), np.float32) / 255
    a[..., 3] = (a[..., 3] > 0.4)
    return a


def rust_grate(seed=5):
    rng = np.random.default_rng(seed)
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    bar = ((xx % 8) < 2) | ((yy % 8) < 2)
    v = _noise(rng, n, n, ((4, 0.6), (16, 0.4)))
    metal = np.stack([0.45 + 0.25 * v, 0.24 + 0.1 * v, 0.14 + 0.05 * v], -1)
    hole = np.array([0.04, 0.03, 0.03])
    im = np.where(bar[..., None], metal, hole)
    return im.astype(np.float32)


def rust_wall(seed=8):
    rng = np.random.default_rng(seed)
    c = Canvas(64, 64, (0.36, 0.18, 0.11), 0.0, 1.0)
    d = c.d
    for _ in range(30):
        x, y = rng.integers(0, 64, 2) * SS
        r = rng.integers(3, 12) * SS
        tone = rng.choice([(70, 32, 18), (120, 60, 30), (40, 20, 14)])
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in tone) + (110,))
    for x in (0, 31):   # panel seams + rivets
        d.rectangle([x * SS, 0, (x + 1) * SS, 64 * SS], fill=(25, 12, 8, 255))
        for y in range(4, 64, 8):
            d.ellipse([(x + 3) * SS, y * SS, (x + 5) * SS, (y + 2) * SS], fill=(150, 80, 45, 255))
    for _ in range(3):  # dried blood smears
        x = rng.integers(5, 55) * SS; y = rng.integers(5, 30) * SS
        d.polygon([(x, y), (x + 9 * SS, y + 2 * SS), (x + 6 * SS, y + 30 * SS), (x + 2 * SS, y + 26 * SS)],
                  fill=(70, 6, 6, 200))
    return c.done()


def welcome_sign():
    from PIL import ImageFont
    w, h = 160, 72
    im = Image.new('RGB', (w * SS, h * SS), (196, 186, 150))
    d = ImageDraw.Draw(im, 'RGBA')
    d.rectangle([3 * SS, 3 * SS, (w - 3) * SS, (h - 3) * SS], outline=(40, 62, 40), width=3 * SS)
    try:
        f1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 13 * SS)
        f2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 20 * SS)
        f3 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 10 * SS)
    except OSError:
        f1 = f2 = f3 = ImageFont.load_default()
    ink = (35, 58, 38)
    d.text((w * SS / 2, 15 * SS), 'WELCOME TO', font=f1, fill=ink, anchor='mm')
    d.text((w * SS / 2, 37 * SS), 'INTERLAGOS', font=f2, fill=ink, anchor='mm')
    d.text((w * SS / 2, 58 * SS), 'pop. 0  \u00b7  est. 2012', font=f3, fill=ink, anchor='mm')
    rng = np.random.default_rng(4)
    for _ in range(40):   # rust and rot
        x, y = rng.integers(0, w, 1)[0] * SS, rng.integers(0, h, 1)[0] * SS
        r = rng.integers(2, 9) * SS
        d.ellipse([x - r, y - r, x + r, y + r], fill=(90, 50, 25, int(rng.integers(40, 140))))
    for _ in range(6):
        x = rng.integers(0, w) * SS
        d.rectangle([x, 0, x + 2 * SS, h * SS], fill=(60, 40, 25, 70))
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def wood(seed=6):
    rng = np.random.default_rng(seed)
    v = 0.5 + 0.3 * np.sin(np.arange(16)[None, :] * 1.3 + rng.random((64, 1)) * 0.5) * 0.3
    v = v + 0.2 * rng.random((64, 16))
    return (np.stack([v * 0.45, v * 0.33, v * 0.24], -1)).astype(np.float32)


# ---------------------------------------------------------------- Silent Hill street signage
_FONTS = {'serif': '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
          'sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'}


def board(w, h, lines, fg, bg, border=None, grime=0.4, seed=0):
    """Painted sign board. lines: [(text, size_px, font, y_frac)]."""
    from PIL import ImageFont
    im = Image.new('RGB', (w * SS, h * SS), rgb(bg))
    d = ImageDraw.Draw(im, 'RGBA')
    if border is not None:
        d.rectangle([2 * SS, 2 * SS, (w - 3) * SS, (h - 3) * SS], outline=rgb(border), width=2 * SS)
    for text, size, font, yf in lines:
        try:
            f = ImageFont.truetype(_FONTS[font], int(size * SS))
        except OSError:
            f = ImageFont.load_default()
        d.text((w * SS / 2, yf * h * SS), text, font=f, fill=rgb(fg), anchor='mm')
    rng = np.random.default_rng(seed)
    for _ in range(int(30 * grime)):
        x, y = rng.integers(0, w) * SS, rng.integers(0, h) * SS
        r = int(rng.integers(1, 6)) * SS
        d.ellipse([x - r, y - r, x + r, y + r], fill=(60, 40, 25, int(rng.integers(30, 120))))
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def brick(seed=31, tone=(0.42, 0.22, 0.17)):
    rng = np.random.default_rng(seed)
    im = np.zeros((64, 64, 3), np.float32)
    for row in range(16):
        off = 4 if row % 2 else 0
        for col in range(-1, 8):
            x0 = col * 8 + off
            c = np.array(tone) * (0.8 + 0.35 * rng.random())
            im[row * 4:row * 4 + 3, max(x0, 0):max(min(x0 + 7, 64), 0)] = c
    im[im.sum(-1) == 0] = (0.30, 0.29, 0.27)                              # mortar
    im *= (0.8 + 0.3 * _noise(rng, 64, 64)[..., None])
    return im


def shopfront(seed, wall, frame=(0.12, 0.12, 0.11), lit=False):
    """4m x 3.5m ground floor bay: big shop window + door, kick plate."""
    rng = np.random.default_rng(seed)
    c = Canvas(64, 56, wall, 0.0, 1.0)
    d = c.d
    S_ = SS
    d.rectangle([3 * S_, 10 * S_, 40 * S_, 46 * S_], fill=rgb(frame))
    glass = (0.30, 0.26, 0.18) if lit else (0.05, 0.06, 0.07)
    d.rectangle([5 * S_, 12 * S_, 38 * S_, 44 * S_], fill=rgb(glass))
    for _ in range(5):   # reflections / dirt on the glass
        x = rng.integers(6, 34) * S_
        d.line([(x, 13 * S_), (x + 6 * S_, 43 * S_)], fill=(90, 95, 100, 40), width=2 * S_)
    d.rectangle([46 * S_, 8 * S_, 60 * S_, 56 * S_], fill=rgb(frame))      # door
    d.rectangle([48 * S_, 10 * S_, 58 * S_, 30 * S_], fill=rgb(glass))
    d.rectangle([0, 46 * S_, 64 * S_, 56 * S_], fill=tuple(int(v * 160) for v in wall) + (255,))
    d.rectangle([0, 0, 64 * S_, 6 * S_], fill=(20, 20, 20, 200))         # sign band shadow
    return c.done()


def stripes(c1, c2, n=8):
    im = np.zeros((16, 16 * n // 2, 3), np.float32)
    for k in range(n):
        im[:, k * 8:(k + 1) * 8] = c1 if k % 2 == 0 else c2
    return im


# ---------------------------------------------------------------- HD race suits
class HD:
    """1024px painter in (angle_deg, v) space: angle 0 = chest centre, +-180 = spine."""

    def __init__(self, base, w=1024, h=1024, k=2, xs=0.5):
        self.w, self.h, self.k = w, h, k
        self.xs = xs                  # texels are ~2x wider than tall once wrapped on the body
        self.im = Image.new('RGB', (w * k, h * k), rgb(base))
        self.d = ImageDraw.Draw(self.im, 'RGBA')

    def p(self, ang, v):
        return ((0.5 + ang / 360.0) * self.w * self.k, (1 - v) * self.h * self.k)

    def rect(self, a0, v0, a1, v1, fill, alpha=255, r=0):
        x0, y0 = self.p(a0, v1); x1, y1 = self.p(a1, v0)
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=r * self.k, fill=rgb(fill) + (alpha,))

    def poly(self, pts, fill, alpha=255):
        self.d.polygon([self.p(a, v) for a, v in pts], fill=rgb(fill) + (alpha,))

    def ellipse(self, a, v, ra, rv, fill, alpha=255, outline=None, width=0):
        x0, y0 = self.p(a - ra, v + rv); x1, y1 = self.p(a + ra, v - rv)
        self.d.ellipse([x0, y0, x1, y1], fill=rgb(fill) + (alpha,) if fill is not None else None,
                       outline=rgb(outline) if outline else None, width=width * self.k)

    def line(self, pts, fill, width=1.0, alpha=255):
        self.d.line([self.p(a, v) for a, v in pts], fill=rgb(fill) + (alpha,), width=max(1, int(width * self.k)))

    def text(self, a, v, txt, size, fill, font='sans', stretch=1.0):
        from PIL import ImageFont
        f = ImageFont.truetype(_FONTS[font], int(size * self.k))
        x, y = self.p(a, v)
        stretch *= self.xs
        tw, th = int(f.getlength(txt)) + 4, int(size * self.k * 1.4)
        tmp = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((tw / 2, th / 2), txt, font=f, fill=rgb(fill), anchor='mm')
        tmp = tmp.resize((max(1, int(tw * stretch)), th), Image.LANCZOS)
        self.im.paste(tmp, (int(x - tmp.width / 2), int(y - th / 2)), tmp)

    def quilt(self, col, dv=0.07, dang=22, alpha=90):
        for v in np.arange(0.0, 1.0, dv):
            self.line([(-180, v), (180, v)], col, 1.2, alpha)
        for ang in np.arange(-180, 180, dang):
            self.line([(ang, 0), (ang, 1)], col, 1.0, int(alpha * 0.7))

    def done(self, seed=0, grain=0.06):
        im = np.asarray(self.im.resize((self.w, self.h), Image.LANCZOS), np.float32) / 255
        rng = np.random.default_rng(seed)
        n = rng.standard_normal((self.h // 4, self.w // 4)).repeat(4, 0).repeat(4, 1)
        return np.clip(im * (1 + grain * n[..., None]), 0, 1).astype(np.float32)


def _santander(c, a, v, w, h, fg, bg=None):
    """Flame mark + wordmark in a box w degrees wide, h (v units) tall."""
    if bg is not None:
        c.rect(a - w / 2, v - h / 2, a + w / 2, v + h / 2, bg, r=6)
    fa = a - w * 0.36
    c.poly([(fa, v - h * 0.32), (fa - w * 0.035, v + h * 0.05), (fa, v + h * 0.34), (fa + w * 0.035, v)], fg)
    c.text(a + w * 0.07, v, 'Santander', h * 1024 * 0.55, fg, 'serif', stretch=0.85)


def _shell(c, a, v, w, h):
    c.rect(a - w / 2, v - h / 2, a + w / 2, v + h / 2, (0.97, 0.97, 0.95), r=6)
    c.ellipse(a, v + h * 0.02, w * 0.36, h * 0.36, (0.98, 0.78, 0.05), outline=(0.85, 0.12, 0.05), width=3)
    c.rect(a - w * 0.4, v - h * 0.4, a + w * 0.4, v - h * 0.18, (0.97, 0.97, 0.95))
    for k in range(-3, 4):
        c.line([(a, v - h * 0.2), (a + k * w * 0.09, v + h * 0.33)], (0.85, 0.12, 0.05), 2)


def _ferrari_panel(c, a, v, w, h):
    c.poly([(a - w / 2, v - h / 2), (a + w / 2, v - h / 2), (a + w * 0.42, v + h / 2), (a - w * 0.42, v + h / 2)],
           (0.97, 0.97, 0.95))
    sa, sv, sw, sh = a - w * 0.22, v + h * 0.1, w * 0.13, h * 0.32
    c.poly([(sa - sw, sv + sh), (sa + sw, sv + sh), (sa + sw, sv - sh * 0.3), (sa, sv - sh), (sa - sw, sv - sh * 0.3)],
           (1.0, 0.82, 0.0))
    c.rect(sa - sw, sv + sh * 0.75, sa + sw, sv + sh, (0.1, 0.55, 0.25))
    c.poly([(sa - sw * 0.3, sv - sh * 0.6), (sa + sw * 0.25, sv + sh * 0.55), (sa + sw * 0.5, sv + sh * 0.4),
            (sa + sw * 0.05, sv - sh * 0.6)], (0.05, 0.05, 0.05))
    c.line([(a - w * 0.02, v + h * 0.12), (a + w * 0.38, v + h * 0.3)], (0.85, 0.05, 0.05), 3)
    c.text(a + w * 0.16, v - h * 0.22, 'SCUDERIA FERRARI', h * 1024 * 0.16, (0.3, 0.3, 0.3), 'sans', stretch=0.75)


def _redbull(c, a, v, w, h, red=(0.86, 0.06, 0.16), sun=(1.0, 0.78, 0.0)):
    c.ellipse(a, v + h * 0.12, w * 0.2, h * 0.36, sun)
    for sgn in (-1, 1):
        x = a + sgn * w * 0.25
        c.ellipse(x, v - h * 0.08, w * 0.2, h * 0.24, red)                                   # body
        c.ellipse(a + sgn * w * 0.08, v + h * 0.02, w * 0.075, h * 0.17, red)              # lowered head
        c.poly([(a + sgn * w * 0.06, v + h * 0.14), (a + sgn * w * 0.02, v + h * 0.33),
                (a + sgn * w * 0.11, v + h * 0.15)], red)                                    # horn
        for lx in (0.12, 0.38):
            c.rect(a + sgn * w * lx - w * 0.025, v - h * 0.45, a + sgn * w * lx + w * 0.025, v - h * 0.2, red)
        c.poly([(a + sgn * w * 0.44, v - h * 0.05), (a + sgn * w * 0.5, v + h * 0.12),
                (a + sgn * w * 0.46, v - h * 0.1)], red)                                     # tail


def suit_torso(kind):
    if kind == 'ferrari':
        c = HD((0.80, 0.06, 0.05))
        c.quilt((0.45, 0.02, 0.02))
        for sgn in (-1, 1):                                       # Italian flag piping down the sides
            for k, col in enumerate(((0.0, 0.55, 0.25), (0.97, 0.97, 0.95), (0.0, 0.55, 0.25))):
                a0 = sgn * (70 + k * 3.2)
                c.rect(min(a0, a0 + sgn * 3.2), 0.18, max(a0, a0 + sgn * 3.2), 0.86, col)
            c.line([(sgn * 30, 0.985), (sgn * 70, 0.93), (sgn * 95, 0.90)], (0.97, 0.97, 0.95), 3)   # shoulder piping
            c.line([(sgn * 32, 0.97), (sgn * 72, 0.915), (sgn * 96, 0.885)], (0.97, 0.97, 0.95), 2)
        _shell(c, -24, 0.80, 26, 0.15)
        _ferrari_panel(c, 26, 0.80, 36, 0.14)
        _santander(c, 0, 0.635, 92, 0.095, (0.85, 0.05, 0.05), bg=(0.98, 0.98, 0.97))
        c.rect(-180, 0.03, 180, 0.12, (0.62, 0.03, 0.03))                                       # belt
        c.rect(-6, 0.065, 8, 0.095, (0.85, 0.05, 0.05)); c.rect(-6, 0.073, 8, 0.087, (1.0, 0.8, 0.0))   # Spain
        c.text(26, 0.08, 'FERNANDO', 30, (0.98, 0.98, 0.98), 'sans', stretch=0.8)
        c.text(-26, 0.08, 'PUMA', 30, (0.98, 0.98, 0.98), 'sans', stretch=0.8)
        c.text(0, 0.965, 'Santander', 26, (0.98, 0.98, 0.98), 'serif')
        for a0 in (180, -180):
            _santander(c, a0, 0.74, 100, 0.10, (0.98, 0.98, 0.97))
            c.text(a0, 0.55, 'ALONSO', 60, (0.98, 0.98, 0.97), 'sans')
        return c.done(1, grain=0.025)
    if kind == 'redbull':
        c = HD((0.06, 0.08, 0.24))
        c.quilt((0.02, 0.03, 0.10))
        for sgn in (-1, 1):
            c.poly([(sgn * 60, 0.1), (sgn * 100, 0.1), (sgn * 100, 0.5), (sgn * 75, 0.35)], (0.75, 0.06, 0.14))
            c.poly([(sgn * 66, 0.1), (sgn * 78, 0.1), (sgn * 96, 0.42), (sgn * 90, 0.44)], (1.0, 0.78, 0.0))
        _redbull(c, 0, 0.70, 80, 0.16)
        c.text(0, 0.86, 'INFINITI', 46, (0.97, 0.97, 0.97), 'sans', stretch=1.1)
        c.text(-26, 0.55, 'TOTAL', 34, (0.97, 0.3, 0.1), 'sans')
        c.text(26, 0.55, 'Pepe Jeans', 28, (0.97, 0.97, 0.97), 'serif')
        c.rect(-180, 0.03, 180, 0.12, (0.04, 0.05, 0.16))
        c.text(0, 0.08, 'SEBASTIAN', 30, (0.97, 0.97, 0.97), 'sans', stretch=0.8)
        c.text(0, 0.965, 'RED BULL RACING', 22, (0.97, 0.97, 0.97), 'sans')
        for a0 in (180, -180):
            _redbull(c, a0, 0.7, 80, 0.16)
        return c.done(2, grain=0.025)
    # mechanic team shirt
    c = HD((0.07, 0.09, 0.25))
    for sgn in (-1, 1):
        c.rect(sgn * 80 - 4, 0.0, sgn * 80 + 4, 1.0, (0.75, 0.06, 0.14))
    _redbull(c, -22, 0.76, 34, 0.08)
    c.text(24, 0.78, 'INFINITI', 30, (0.97, 0.97, 0.97), 'sans')
    c.text(0, 0.62, 'RED BULL RACING', 30, (0.97, 0.97, 0.97), 'sans')
    for a0 in (180, -180):
        c.text(a0, 0.72, 'RED BULL', 64, (0.97, 0.97, 0.97), 'sans')
    return c.done(3, grain=0.02)


def suit_limb(kind, part):
    """Sleeves and legs: u around the limb, v along it (0 = far end)."""
    base = {'ferrari': (0.80, 0.06, 0.05), 'redbull': (0.06, 0.08, 0.24), 'mech': (0.07, 0.09, 0.25),
            'mech_legs': (0.05, 0.06, 0.16)}[kind]
    c = HD(base, 256, 512)
    if kind in ('ferrari', 'redbull'):
        c.quilt(tuple(np.array(base) * 0.55), dv=0.09, dang=45)
    if kind == 'ferrari' and part == 'sleeve':
        c.rect(-180, 0.0, 180, 0.05, (0.62, 0.03, 0.03))
        c.rect(150, 0.2, 180, 0.95, (0.97, 0.97, 0.95)); c.rect(-180, 0.2, -150, 0.95, (0.97, 0.97, 0.95))
    if kind == 'redbull':
        c.rect(-180, 0.02, 180, 0.08, (0.75, 0.06, 0.14)); c.rect(-180, 0.08, 180, 0.11, (1.0, 0.78, 0.0))
    if kind == 'mech' and part == 'sleeve':
        c.rect(-180, 0.0, 180, 0.04, (0.75, 0.06, 0.14))
    return c.done(4, grain=0.04)


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
    T['suit_ferrari'] = suit_torso('ferrari')
    T['suit_redbull'] = suit_torso('redbull')
    T['suit_mech'] = suit_torso('mech')
    T['sleeve_ferrari'] = suit_limb('ferrari', 'sleeve')
    T['legs_ferrari'] = suit_limb('ferrari', 'legs')
    T['sleeve_redbull'] = suit_limb('redbull', 'sleeve')
    T['legs_redbull'] = suit_limb('redbull', 'legs')
    T['sleeve_mech'] = suit_limb('mech', 'sleeve')
    T['legs_mech'] = suit_limb('mech_legs', 'legs')
    T['concrete'] = concrete()
    T['num1'] = decal_number('1', (0.95, 0.95, 0.95), (0.08, 0.10, 0.26))
    T['rb_logo'] = redbull_logo()
    T['asphalt'] = asphalt()
    T['facade1'] = facade(1, (0.55, 0.52, 0.47))
    T['facade2'] = facade(2, (0.47, 0.42, 0.38))
    T['facade3'] = facade(3, (0.40, 0.42, 0.44), window_lit=True)
    T['chainlink'] = chainlink()
    T['chainlink_rust'] = chainlink(rust=True)
    T['chain'] = chain()
    T['rust_grate'] = rust_grate()
    T['rust_wall'] = rust_wall()
    T['welcome'] = welcome_sign()
    T['brick'] = brick()
    T['brick_dark'] = brick(32, (0.25, 0.20, 0.18))
    T['shop_neely'] = shopfront(41, (0.18, 0.24, 0.18))
    T['shop_heaven'] = shopfront(42, (0.06, 0.05, 0.06), frame=(0.25, 0.08, 0.15))
    T['shop_cafe'] = shopfront(43, (0.70, 0.66, 0.58), lit=True)
    T['sign_neely'] = board(80, 56, [("NEELY'S", 14, 'serif', 0.36), ('BAR', 17, 'serif', 0.74)],
                            (0.95, 0.88, 0.70), (0.10, 0.22, 0.14), border=(0.85, 0.75, 0.5), seed=1)
    T['sign_neely_h'] = board(192, 32, [("NEELY'S BAR", 22, 'serif', 0.52)], (0.95, 0.88, 0.70),
                              (0.10, 0.22, 0.14), border=(0.85, 0.75, 0.5), seed=2)
    T['sign_heaven'] = board(96, 56, [("HEAVEN'S", 14, 'sans', 0.34), ('NIGHT', 19, 'sans', 0.72)],
                             (1.0, 0.35, 0.75), (0.04, 0.02, 0.04), border=(0.6, 0.2, 0.9),
                             grime=0.1, seed=3)
    T['sign_cafe'] = board(160, 40, [('CAFE 5to2', 26, 'sans', 0.5)], (0.95, 0.92, 0.85),
                           (0.62, 0.08, 0.06), border=(0.95, 0.92, 0.85), seed=4)
    T['street_sign'] = board(96, 20, [('LINDSEY ST', 11, 'sans', 0.52)], (0.95, 0.95, 0.92),
                             (0.08, 0.36, 0.18), border=(0.95, 0.95, 0.92), grime=0.2, seed=5)
    T['street_sign2'] = board(96, 20, [('KATZ ST', 11, 'sans', 0.52)], (0.95, 0.95, 0.92),
                              (0.08, 0.36, 0.18), border=(0.95, 0.95, 0.92), grime=0.2, seed=6)
    T['awning'] = stripes((0.55, 0.08, 0.06), (0.85, 0.82, 0.75))
    T['barricade'] = stripes((0.75, 0.08, 0.06), (0.9, 0.9, 0.88), n=10)
    T['road_closed'] = board(96, 24, [('ROAD CLOSED', 12, 'sans', 0.52)], (0.08, 0.08, 0.08),
                             (0.92, 0.82, 0.25), grime=0.6, seed=7)
    T['wood'] = wood()
    import heads
    al = heads.alonso()
    T['alonso_head'] = al.head_texture()
    T['alonso_hair'] = al.hair_texture()
    sv = heads.vettel()
    T['vettel_head'] = sv.head_texture()
    T['vettel_hair'] = sv.hair_texture()
    names = list(T)
    return [T[k] for k in names], {k: i for i, k in enumerate(names)}
