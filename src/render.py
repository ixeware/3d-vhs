"""Render every frame and pipe it into ffmpeg together with the soundtrack."""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

import timeline as TL
import textures
from engine import rasterize, resolve
from scenes import Stage
from shade import shade, volume, dust, bloom, tonemap, ps1_dither
from vhs import degrade, OW, OH

_STAGE = None
_TEX = None


def _init():
    global _STAGE, _TEX
    _TEX, ix = textures.build_all()
    _STAGE = Stage(ix)


def clock(t):
    s = int(17 * 3600 + 42 * 60 + 7 + t)
    return f'PM {s // 3600 - 12}:{s // 60 % 60:02d}:{s % 60:02d}'


def render_frame(i):
    t = i / TL.FPS
    frame = np.zeros((240, 320, 3))
    if TL.LAMP1_ON - 0.2 <= t < TL.CUT:
        shot = _STAGE.shot_alonso(t)
    elif TL.CUT <= t < TL.TAPE_END:
        shot = _STAGE.shot_redbull(t)
    else:
        shot = None
    if shot is not None:
        scene, occ, cam, spot, glow, dbox = shot
        if spot.I > 0:
            spot.build_shadow(occ)
        vis = rasterize(scene, cam)
        G = resolve(scene, cam, vis, _TEX)
        lin = shade(G, cam, spot)
        lin += volume(cam, spot, G['depth'], t, density=0.0065, seed=i)
        dust(lin, cam, spot, G['depth'], t, box=dbox)
        lin = bloom(lin, thresh=1.6, radius=3, strength=0.4)
        frame = ps1_dither(tonemap(lin))

    # tape events
    glitch, snow_amt = 0.0, 0.0
    if t < TL.LAMP1_ON - 0.2:
        snow_amt = 1.0 if t < 0.45 else 0.0
        glitch = 1.0 if t < 0.6 else 0.3
    if abs(t - TL.CUT) < 0.18:
        glitch = 1.0
        snow_amt = 0.55 if t >= TL.CUT and t < TL.CUT + 0.07 else 0.0
    if t >= TL.TAPE_END:
        dt = t - TL.TAPE_END
        snow_amt = 1.0 if dt < 0.55 else 0.0
        glitch = 1.0 if dt < 0.7 else 0.0
    osd_lines = []
    if t < 3.2 and int(t * 2) % 2 == 0 or t < 1.0:
        osd_lines.append((40, 32, 'PLAY ▶'))
    if TL.CUT <= t < TL.CUT + 0.25 or TL.TAPE_END + 0.6 > t >= TL.TAPE_END:
        osd_lines.append((40, 32, 'TRACKING'))
    if t < TL.TAPE_END:
        osd_lines.append((420, 404, 'NOV.25 2012'))
        osd_lines.append((420, 432, clock(t)))
    if t >= TL.TAPE_END + 0.7:
        osd_lines = []
        frame[:] = 0
    return degrade(frame, i, t, glitch=glitch, osd_lines=osd_lines, snow_amt=snow_amt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/alonso_2012_vhs.mp4')
    ap.add_argument('--audio', default='out/soundtrack.wav')
    ap.add_argument('--frames', default=None, help='a:b frame range for previews')
    ap.add_argument('--png', default=None, help='write PNGs to this dir instead of video')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args()

    n = int(round(TL.DUR * TL.FPS))
    rng = range(n)
    if a.frames:
        lo, hi = map(int, a.frames.split(':')); rng = range(lo, hi)
    t0 = time.time()
    if a.png:
        from PIL import Image
        os.makedirs(a.png, exist_ok=True)
        with Pool(a.jobs, _init) as pool:
            for k, img in zip(rng, pool.imap(render_frame, rng)):
                Image.fromarray(img).save(f'{a.png}/f{k:04d}.png')
        return
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(TL.FPS), '-i', '-',
           '-i', a.audio,
           '-vf', 'scale=960:720:flags=bicubic', '-c:v', 'libx264', '-preset', 'slow', '-crf', '23',
           '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart',
           a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(a.jobs, _init) as pool:
        for k, img in enumerate(pool.imap(render_frame, rng, chunksize=2)):
            ff.stdin.write(img.tobytes())
            if k % 30 == 0:
                print(f'frame {k}/{len(rng)}  {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    ff.stdin.close()
    ff.wait()
    print('done', a.out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
