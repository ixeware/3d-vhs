"""Analog VHS degradation applied to the 320x240 PS1 frame."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OW, OH = 640, 480
_FONT = None


def _font(size):
    global _FONT
    if _FONT is None:
        for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
                  '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'):
            try:
                _FONT = ImageFont.truetype(p, size); break
            except OSError:
                pass
        _FONT = _FONT or ImageFont.load_default()
    return _FONT


def osd(img, t, lines):
    """Camcorder style on-screen text (drawn before the tape smears it)."""
    if not lines:
        return img
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = _font(26)
    for (x, y, txt) in lines:
        for dx, dy in ((2, 2), (-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((x + dx, y + dy), txt, font=f, fill=(0, 0, 0))
        d.text((x, y), txt, font=f, fill=(235, 235, 235))
    return np.asarray(im, np.float32) / 255


def snow(rng, h=OH, w=OW):
    n = rng.random((h // 2, w // 2))
    n = np.kron(n, np.ones((2, 2)))
    streak = rng.random((h, 1)) ** 6
    v = np.clip(n * 0.85 + streak * 0.4, 0, 1)
    return np.repeat(v[..., None], 3, 2)


def _hblur(a, k):
    if k <= 1:
        return a
    c = np.cumsum(np.pad(a, [(0, 0), (k // 2 + 1, k - k // 2 - 1)] + [(0, 0)] * (a.ndim - 2), mode='edge'), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def _shift_rows(a, offs):
    """Shift each row horizontally by a (fractional) number of pixels."""
    h, w = a.shape[:2]
    x = np.arange(w)[None, :] - offs[:, None]
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    f = (x - np.floor(x))
    rows = np.arange(h)[:, None]
    if a.ndim == 3:
        f = f[..., None]
    return a[rows, x0] * (1 - f) + a[rows, x1] * f


def degrade(frame, i, t, glitch=0.0, osd_lines=None, snow_amt=0.0):
    """frame: (240,320,3) in [0,1]. glitch: 0..1 tracking trouble."""
    rng = np.random.default_rng(1000 + i)
    img = np.kron(frame, np.ones((2, 2, 1)))
    img = osd(img, t, osd_lines)
    if snow_amt > 0:
        img = img * (1 - snow_amt) + snow(rng) * snow_amt

    # RGB -> YIQ
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    I = 0.596 * img[..., 0] - 0.274 * img[..., 1] - 0.322 * img[..., 2]
    Q = 0.211 * img[..., 0] - 0.523 * img[..., 1] + 0.312 * img[..., 2]

    # luma: soft + edge ringing
    Yb = _hblur(Y, 3)
    Y = Yb + 0.45 * (Yb - _hblur(Yb, 9))
    # chroma: low bandwidth, delayed, a little noisy
    I = np.roll(_hblur(I, 13), 4, axis=1) * 0.9
    Q = np.roll(_hblur(Q, 17), 5, axis=1) * 0.9
    cn = np.repeat(rng.standard_normal((OH, OW // 16)), 16, axis=1)
    I += _hblur(cn, 9) * 0.012; Q += _hblur(cn[::-1], 9) * 0.012

    # line jitter / tracking
    offs = rng.standard_normal(OH) * 0.35
    offs += 1.2 * np.sin(np.arange(OH) * 0.05 + t * 3.0) * 0.3
    if glitch > 0:
        band = (np.arange(OH) - (OH * (1 - (t * 1.7) % 1.0))) / 40.0
        offs += glitch * 25 * np.exp(-band ** 2) * rng.standard_normal(OH)
        offs += glitch * 6 * rng.standard_normal(OH)
    hs = 10 + int(4 * rng.random())                # head-switching noise
    offs[-hs:] += np.linspace(0, 18, hs) + rng.standard_normal(hs) * 3
    Y = _shift_rows(Y, offs); I = _shift_rows(I, offs); Q = _shift_rows(Q, offs)
    Y[-hs:] += rng.random((hs, OW)) * 0.15

    # tape noise
    Y += rng.standard_normal((OH, OW)) * 0.022
    Y += np.repeat(rng.standard_normal((OH, 1)), OW, 1) * 0.01
    n_drop = rng.poisson(1.5 + 30 * glitch)
    for _ in range(n_drop):
        y = rng.integers(0, OH); x = rng.integers(0, OW); L = rng.integers(8, 120)
        Y[y, x:x + L] += 0.5 + 0.4 * rng.random()

    # YIQ -> RGB
    r = Y + 0.956 * I + 0.621 * Q
    g = Y - 0.272 * I - 0.647 * Q
    b = Y - 1.106 * I + 1.703 * Q
    out = np.stack([r, g, b], -1)
    out = 0.025 + out * 0.95                          # lifted tape black
    out[1::2] *= 0.88                                 # scanlines
    yy, xx = np.mgrid[0:OH, 0:OW]
    vig = 1 - 0.35 * (((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2) ** 1.5
    out *= vig[..., None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
