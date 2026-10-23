"""Soundtrack: VHS-mangled anime voice lines, lamp clunks/hum, drone, tape noise.

Voice lines come from ElevenLabs (see tts.py) and are stored in assets/.
Everything else is synthesised here.
"""
import argparse
import subprocess
import wave

import numpy as np

import timeline as TL

SR = 44100


def load(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def band(x, lo, hi, order=2.0):
    """Soft FFT band-pass."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = 1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** (2 * order)) if lo else 1.0
    g = g / np.sqrt(1 + (f / hi) ** (2 * order)) if hi else g
    return np.fft.irfft(np.fft.rfft(x) * g, n)


def resample_var(x, rate):
    """Read x with a time-varying playback rate (wow/flutter, pitch shift)."""
    pos = np.cumsum(rate)
    pos = pos[pos < len(x) - 1]
    i = pos.astype(int); f = pos - i
    return x[i] * (1 - f) + x[i + 1] * f


def reverb(x, secs=1.2, mix=0.25, seed=0, damp=3000):
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 6.5))
    ir = band(ir, 200, damp)
    ir /= np.sqrt((ir ** 2).sum())
    wet = np.fft.irfft(np.fft.rfft(x, len(x) + n) * np.fft.rfft(ir, len(x) + n), len(x) + n)
    out = np.zeros(len(x) + n); out[:len(x)] += x * (1 - mix)
    return out + wet * mix


def vhs_voice(x, seed, pitch=1.07, sad=False):
    rng = np.random.default_rng(seed)
    n = int(len(x) / pitch) + SR
    t = np.arange(n) / SR
    wow = 0.0035 * np.sin(2 * np.pi * 0.55 * t + rng.random() * 6)
    flutter = 0.0012 * np.sin(2 * np.pi * 8.7 * t)
    y = resample_var(x, pitch * (1 + wow + flutter))
    y = band(y, 280, 4200, 2.5)                       # narrow tape/TV speaker band
    y = np.tanh(2.2 * y / (np.abs(y).max() + 1e-9)) / np.tanh(2.2)
    y = np.round(y * 512) / 512                       # crunchy
    # dropouts: brief level dips where the oxide is worn
    env = np.ones(len(y))
    for _ in range(rng.integers(3, 6)):
        c = rng.integers(0, len(y)); w = rng.integers(800, 3500)
        env[c:c + w] *= np.hanning(min(w, len(y) - c)) * -0.75 + 1
    y *= env
    y = reverb(y, 1.6 if sad else 0.9, 0.32 if sad else 0.18, seed)
    return y / np.abs(y).max()


def place(buf, x, t, gain=1.0):
    i = int(t * SR)
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[:j - i] * gain


def clunk(rng, big=True):
    """Heavy relay / breaker thunk of a theatre lamp striking."""
    n = int(0.6 * SR); t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 62 * t) * np.exp(-t / 0.09) * 0.9
    body += np.sin(2 * np.pi * 118 * t) * np.exp(-t / 0.05) * 0.4
    hit = band(rng.standard_normal(n), 1200, 9000) * np.exp(-t / 0.006) * 1.4
    ring = np.sin(2 * np.pi * 1730 * t) * np.exp(-t / 0.04) * 0.15
    y = body + hit + ring
    if big:
        y = reverb(y, 2.4, 0.4, 5, 2500)
    return y


def click(rng):
    """Small hard switch click."""
    n = int(0.25 * SR); t = np.arange(n) / SR
    tr = band(rng.standard_normal(n), 2000, 12000) * np.exp(-t / 0.0025) * 1.2
    tr += np.sin(2 * np.pi * 2900 * t) * np.exp(-t / 0.012) * 0.35
    tr += np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.02) * 0.4
    return reverb(tr, 1.8, 0.3, 9, 4000)


def lamp_hum(n, rng):
    t = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * 100 * k * t + rng.random() * 6) / k ** 1.3 for k in range(1, 9))
    y += band(rng.standard_normal(n), 3000, 8000) * 0.08    # filament fizz
    return y * 0.05


def tape_snow(n, rng):
    return band(rng.standard_normal(n), 400, 9000) * 0.35


def build(out):
    rng = np.random.default_rng(42)
    N = int(TL.DUR * SR) + SR
    mix = np.zeros(N)
    t = np.arange(N) / SR

    # tape bed: hiss + mains hum, all the way through
    mix += band(rng.standard_normal(N), 1500, 12000) * 0.012
    mix += np.sin(2 * np.pi * 50 * t) * 0.004 + np.sin(2 * np.pi * 150 * t) * 0.002

    # snow + mechanism at start, glitch at cut, snow at the end
    place(mix, tape_snow(int(0.45 * SR), rng) * np.linspace(1, 0.6, int(0.45 * SR)), 0.0, 0.4)
    place(mix, clunk(rng, big=False), 0.02, 0.35)
    place(mix, tape_snow(int(0.2 * SR), rng) * np.hanning(int(0.2 * SR)), TL.CUT - 0.1, 0.5)
    place(mix, tape_snow(int(0.6 * SR), rng), TL.TAPE_END, 0.4)

    # dark drone (silent-hill-ish), duck when the lamp is off
    drone = (np.sin(2 * np.pi * 41.2 * t) + 0.7 * np.sin(2 * np.pi * 41.9 * t)
             + 0.35 * np.sin(2 * np.pi * 82.4 * t + np.sin(t * 0.3) * 2))
    drone += band(rng.standard_normal(N), 60, 400) * 0.8
    drone *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t) ** 2
    live = ((t > TL.LAMP1_ON) & (t < TL.CLICK)).astype(float)
    live = np.convolve(live, np.ones(4410) / 4410, 'same')
    mix += drone * live * 0.045

    # lamps: strike, hum, click
    for t_on, t_off in ((TL.LAMP1_ON, TL.CUT), (TL.LAMP2_ON, TL.CLICK)):
        place(mix, clunk(rng), t_on, 0.9)
        n = int((t_off - t_on) * SR)
        hum = lamp_hum(n, rng)
        hum[:int(0.45 * SR)] *= (rng.random(int(0.45 * SR) // 1323 + 1) > 0.3).repeat(1323)[:int(0.45 * SR)]
        place(mix, hum, t_on)
    place(mix, click(rng), TL.CLICK, 2.2)
    tick = np.arange(int(0.4 * SR)) / SR                 # filament ping as it cools
    place(mix, np.sin(2 * np.pi * 3400 * tick) * np.exp(-tick / 0.08) * 0.03, TL.CLICK + 0.12)

    # far-off muffled crowd under the Red Bull shot
    crowd = band(rng.standard_normal(N), 300, 1800)
    crowd *= 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 0.8 * t + np.sin(t * 2.3)))
    gate = ((t > TL.LAMP2_ON + 0.2) & (t < TL.CLICK)).astype(float)
    gate = np.convolve(gate, np.ones(22050) / 22050, 'same')
    mix += band(crowd, 0, 900) * gate * 0.05

    # voices
    v1 = vhs_voice(load('assets/line1.mp3'), 1)
    v2 = vhs_voice(load('assets/line2.mp3'), 2, sad=True, pitch=1.05)
    place(mix, v1, TL.VOICE1, 0.62)
    place(mix, v2, TL.VOICE2, 0.62)

    # silence the room once the light is gone (only tape left)
    after = t > TL.CLICK + 0.02
    mix[after] *= np.where(t[after] < TL.TAPE_END, 0.6, 1.0)

    mix = mix[:int(TL.DUR * SR)]
    mix = band(mix, 40, 11000)
    mix /= np.abs(mix).max() / 0.89
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/soundtrack.wav')
    build(ap.parse_args().out)
