"""The two shots: Alonso's stare, and Red Bull celebrating on the RB8."""
import numpy as np
from engine import T, Ry, Camera, merge
from models import Look, character, rb8, theatre_lamp
from sets import town, street_lamp, otherworld
from shade import Spot, Env
import timeline as TL

W, H = 320, 240

FERRARI_RED = (0.62, 0.03, 0.03)
RB_NAVY = (0.07, 0.09, 0.24)

FOG_WORLD = Env(fog=(0.30, 0.31, 0.33), fog_dist=11.0, ambient=(0.52, 0.53, 0.57))
RUST_WORLD = Env(fog=(0.20, 0.07, 0.045), fog_dist=8.0, ambient=(0.20, 0.09, 0.06), tint=(1.0, 0.62, 0.5))
BLACK = Env(fog=(0, 0, 0), fog_dist=8.0, ambient=(0, 0, 0), tint=(1.0, 0.62, 0.5))


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def lamp_level(t, t_on, t_off=None, seed=0):
    """Theatre lamp: dead -> flickering strike -> steady; hard cut at t_off."""
    if t < t_on:
        return 0.0
    if t_off is not None and t >= t_off:
        return 0.0
    dt = t - t_on
    if dt < 0.45:
        pattern = [1, 0, 0.15, 1, 0.05, 0.6, 1, 0.3, 1, 1, 0.8, 1, 1, 1]
        return pattern[min(int(dt * 30), len(pattern) - 1)]
    rng = np.random.default_rng(int(t * 30) + seed)
    return 1.0 - 0.025 * rng.random()


def street_flicker(t):
    """Dying sodium lamp: mostly on, with stutters."""
    rng = np.random.default_rng(int(t * 15) + 77)
    if (t % 3.7) < 0.35:
        return 0.15 if rng.random() < 0.6 else 1.0
    return 0.9 + 0.1 * rng.random()


def afterglow(t, t_off):
    if t < t_off:
        return 0.0
    return float(np.exp(-(t - t_off) / (TL.AFTERGLOW / 3)))


class Stage:
    def __init__(self, tex_ix):
        self.tx = tex_ix
        self.alonso = Look('alonso', 'suit_ferrari', FERRARI_RED, FERRARI_RED, (0.80, 0.60, 0.47),
                           (0.09, 0.06, 0.05), (0.08, 0.08, 0.09),
                           hair_kw=dict(spiky=0.03, front=0.064, back=-0.075))
        self.vettel = Look('vettel', 'suit_redbull', RB_NAVY, RB_NAVY, (0.93, 0.76, 0.66),
                           (0.80, 0.64, 0.38), (0.06, 0.06, 0.08),
                           hair_kw=dict(spiky=0.01, front=0.07, back=-0.07))
        skins = [(0.86, 0.68, 0.56), (0.62, 0.44, 0.32), (0.92, 0.75, 0.65), (0.45, 0.30, 0.22),
                 (0.84, 0.66, 0.55)]
        hairs = [(0.30, 0.20, 0.12), (0.06, 0.05, 0.05), (0.75, 0.55, 0.30), (0.05, 0.04, 0.04),
                 (0.45, 0.30, 0.18)]
        caps = [RB_NAVY, None, None, RB_NAVY, None]
        self.mechs = [Look(f'mech{i}', 'suit_mech', RB_NAVY, RB_NAVY, skins[i], hairs[i],
                           (0.05, 0.05, 0.06), cap=caps[i], legs_col=(0.06, 0.07, 0.16))
                      for i in range(5)]
        self.car = rb8(tex_ix)

    # ------------------------------------------------------------ shot 1
    def shot_alonso(self, t):
        lvl = lamp_level(t, TL.LAMP1_ON, seed=1)
        pivot = (0.25, 4.1, 2.3)
        aim_at = np.array([0.0, 1.05, 0.0])
        lamp, lens = theatre_lamp(pivot, aim_at - pivot, on=lvl)
        spot = Spot(lens, aim_at, 9, 15, 7.0 * lvl)

        # stare: body three-quarter, head comes round to the lens
        turn = ease((t - TL.VOICE1 - 3.2) / 2.2)
        breath = np.sin(t * 2 * np.pi / 4.2)
        blink = any(abs(t - b) < 0.06 for b in (6.1, 11.6))
        pose = dict(
            pelvis=(0, 0.0, 0.0), spine=(0.03 + 0.01 * breath, 0.05, 0.0),
            neck=(0.05, -0.15 - 0.15 * turn, 0.0),
            head=(0.13 - 0.04 * turn, -0.25 - 0.30 * turn, 0.05 * (1 - turn)),
            shoulderL=(0.05, 0, 0.10), shoulderR=(-0.02, 0, 0.12),
            elbowL=(-0.25, 0, 0), elbowR=(-0.18, 0, 0),
            hipL=(0, 0, 0.05), hipR=(0, 0, 0.03), kneeL=(0.04, 0, 0))
        root = T(0, 0, 0) @ Ry(0.62)
        alonso = character(self.alonso, self.tx, root, pose, blink=blink)

        if t < TL.LAMP1_ON:
            # establishing: standing on the corner of Lindsey St, panning off Neely's sign
            # down the street to the figure waiting in the fog
            k = ease((t - TL.ESTABLISH) / (TL.LAMP1_ON - TL.ESTABLISH - 0.15))
            eye = lerp((-3.0, 1.5, 1.9), (-2.5, 1.6, 4.3), k)
            tgt = lerp((-5.9, 3.5, -3.75), (1.0, 1.3, -4.5), k)
            cam = Camera(eye, tgt, 50, W, H, roll=0.03 * np.sin(t * 0.8))
        else:
            # wide reveal with the lamp in frame, slow push to a close-up
            k = ease((t - TL.LAMP1_ON - 0.75) / 7.6)
            eye = lerp((0.9, 1.25, 6.4), (0.40, 1.66, 1.18), k)
            tgt = lerp((0.0, 2.6, 0.0), (0.02, 1.52, 0.0), k)
            eye = eye + 0.01 * np.array([np.sin(t * 0.7), np.sin(t * 0.9 + 1), 0])
            fov = float(lerp(56, 34, k))
            cam = Camera(eye, tgt, fov, W, H, roll=0.02 * np.sin(t * 0.4))

        parts = [alonso, town(self.tx, t), street_lamp((4.75, -5.3), street_flicker(t), side=-1)]
        # the one who won it, watching from the fog
        flick = np.random.default_rng(int(t * 30) + 5).random()
        edge = min(t - TL.APP_IN, TL.APP_OUT - t)
        if TL.APP_IN <= t < TL.APP_OUT and (edge > 0.4 or flick < 0.55):
            ghost = dict(shoulderR=(-2.95, 0, 0.05), elbowR=(-0.1, 0, 0), handR='point',
                         head=(0.05, 0.0, 0.0), shoulderL=(0, 0, 0.08))
            parts.append(character(self.vettel, self.tx, T(-2.4, 0, -3.7) @ Ry(0.5), ghost))

        k_other = ease((t - TL.OTHER_START) / (TL.CUT - TL.OTHER_START))
        env = Env.lerp(FOG_WORLD, RUST_WORLD, k_other)
        scene = merge(parts + [lamp])
        occ = merge(parts)
        dust_box = ((-1.5, 1.5), (0.0, 5.0), (-1.5, 2.5))
        return dict(scene=scene, occ=occ, cam=cam, spot=spot, env=env, dust=dust_box, ember=False)

    # ------------------------------------------------------------ shot 2
    def shot_redbull(self, t):
        lvl = lamp_level(t, TL.LAMP2_ON, TL.CLICK, seed=2)
        glow = afterglow(t, TL.CLICK)
        pivot = (-1.0, 6.2, 2.8)
        aim_at = np.array([-0.1, 0.9, 0.9])
        lamp, lens = theatre_lamp(pivot, aim_at - pivot, on=max(lvl, glow * 0.25))
        spot = Spot(lens, aim_at, 18, 26, 12.0 * lvl)

        dead = t >= TL.CLICK
        fan = 1.0 if not dead else glow
        world = otherworld(self.tx, t)
        world.emis = world.emis * fan
        parts = [world, lamp, self.car]
        # Vettel on top of the car, finger up
        bounce = abs(np.sin(t * 5.2)) * 0.03
        vpose = dict(
            spine=(-0.05, 0.1 * np.sin(t * 2.6), 0), head=(-0.25, 0.2 * np.sin(t * 1.3), 0),
            shoulderR=(-2.95, 0, 0.05), elbowR=(-0.1, 0, 0), handR='point',
            shoulderL=(-1.2 + 0.5 * np.sin(t * 5.2), 0, 0.55), elbowL=(-1.6, 0, 0), handL='fist',
            hipL=(0, 0, 0.18), hipR=(0, 0, 0.18), kneeL=(0.15, 0, 0), kneeR=(0.15, 0, 0),
            lift=bounce - 0.04)
        parts.append(character(self.vettel, self.tx, T(0, 0.68, -0.05) @ Ry(-0.55), vpose))

        cam_xz = np.array([-4.0, 5.2])
        spots = [(-1.65, 1.25), (-0.95, 2.85), (0.35, 3.25), (1.35, 2.35), (-1.85, -0.45)]
        for i, (x, z) in enumerate(spots):
            ph = i * 1.37
            jump = max(0.0, np.sin(t * 6.0 + ph)) * (0.12 if i % 2 == 0 else 0.06)
            wave = np.sin(t * 7.0 + ph)
            if i in (1, 3):     # arms up shaking fists
                arms = dict(shoulderL=(-2.7 + 0.2 * wave, 0, 0.35), shoulderR=(-2.7 - 0.2 * wave, 0, 0.35),
                            elbowL=(-0.5, 0, 0), elbowR=(-0.5, 0, 0), handL='fist', handR='fist')
            elif i == 2:        # clapping
                c = 0.5 + 0.5 * np.sin(t * 14 + ph)
                arms = dict(shoulderL=(-1.1, 0, 0.25 * c), shoulderR=(-1.1, 0, 0.25 * c),
                            elbowL=(-0.9, 0, 0), elbowR=(-0.9, 0, 0))
            else:               # one arm punching the air
                arms = dict(shoulderR=(-2.9, 0, 0.1 + 0.15 * wave), elbowR=(-0.3, 0, 0), handR='fist',
                            shoulderL=(-0.4, 0, 0.5), elbowL=(-1.3, 0, 0), handL='fist')
            pose = dict(head=(-0.35, 0.15 * np.sin(t * 2 + ph), 0), spine=(-0.08, 0, 0),
                        kneeL=(0.25 * (1 - jump * 6), 0, 0), kneeR=(0.25 * (1 - jump * 6), 0, 0),
                        hipL=(-0.12 * (1 - jump * 6), 0, 0.05), hipR=(-0.12 * (1 - jump * 6), 0, 0.05),
                        lift=jump - 0.02, **arms)
            to_cam = np.arctan2(cam_xz[0] - x, cam_xz[1] - z)
            to_car = np.arctan2(-x, 0.6 - z)
            yaw_to = to_cam + 0.35 * np.angle(np.exp(1j * (to_car - to_cam))) + 0.3 * np.sin(ph)
            parts.append(character(self.mechs[i], self.tx, T(x, 0, z) @ Ry(yaw_to), pose))

        # wide on the team, then creep in on Vettel as his name is said
        k = ease((t - TL.CUT) / (TL.CLICK - TL.CUT))
        ang = lerp(-0.72, -0.48, k)
        r = float(lerp(6.4, 4.4, k))
        eye = np.array([np.sin(ang) * r, float(lerp(1.75, 2.0, k)), np.cos(ang) * r + 0.4])
        tgt = lerp((0.0, 1.05, 0.5), (0.0, 1.5, 0.25), k)
        cam = Camera(eye, tgt, float(lerp(44, 40, k)), W, H, roll=-0.015)
        scene = merge(parts)
        occ = merge([p for p in parts if p is not lamp])
        dust_box = ((-2.5, 2.5), (0.0, 6.0), (-2.0, 3.5))
        # when the lamp dies, the whole Otherworld dies with it
        env = BLACK if dead else RUST_WORLD
        return dict(scene=scene, occ=occ, cam=cam, spot=spot, env=env, dust=dust_box,
                    ember=not dead)
