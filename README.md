# 3d-vhs — "Perfection was not enough" (2012)

A 25-second PS1 / Silent Hill style 3D clip on a worn VHS tape, made entirely in
code: a numpy software rasterizer, no Blender and no 3D packages.

Output: [`out/alonso_2012_vhs.mp4`](out/alonso_2012_vhs.mp4) (960×720, 30 fps, H.264 + AAC)

## Story
1. Tape snow, then `PLAY ▶`. The camera stands at a corner of Lindsey St in South Vale,
   Silent Hill, under the green NEELY'S BAR sign, and pans down the foggy street: ash
   falling, the amber traffic light blinking on its wire, HEAVEN'S NIGHT buzzing pink, a
   rusted sedan sliding into the abyss where the road has torn away behind the
   ROAD CLOSED barricades. Fernando Alonso stands in the middle of the road in his
   Ferrari suit.
2. A theatre lamp strikes on above him with a clunk. The camera pushes in while he
   turns to stare into the lens (the Brazil 2012 stare).
   A bright anime voice says: *"And yet again, perfection was not enough for Fernando
   Alonso, to win his third championship."*
   Pocket-radio static rises: Vettel's silhouette stands in the fog behind him, finger
   up, and vanishes on a tape glitch.
3. The air-raid siren winds up and the street rusts over into the Otherworld.
   A tracking glitch cuts to a rusted cage with a grated floor, hanging chains, embers
   and a turning fan. A second lamp strikes on over Red Bull mechanics celebrating
   around the RB8, with Sebastian Vettel standing on the car pointing his finger up.
   The same voice, now sad: *"It went to Sebastian and Red Bull in 2012..."*
4. A switch clicks, the lamp dies, the Otherworld goes black with it, the filament glows
   orange for a moment, then tape snow.

## How it works
| file | what it does |
|---|---|
| `src/engine.py` | triangle soup meshes, primitives (box, loft, sphere), camera, near-plane clipping, rasterizer with alpha-tested textures that writes a visibility buffer and then resolves it. Vertices snap to whole pixels and UVs are interpolated affinely, which gives the PS1 wobble and texture warp |
| `src/models.py` | low-poly characters on a small FK rig, the RB8, the Fresnel theatre lamp |
| `src/sets.py` | the two locations: the Silent Hill street (shopfronts and signs, utility poles and wires, traffic light, parked cars, the torn road and abyss) and the rusted Otherworld cage |
| `src/textures.py` | painted faces (with a blink frame), race suits, livery, concrete floor |
| `src/shade.py` | spotlight with a shadow map, light bouncing off the floor, distance fog and sky light, ray-marched volumetric beam, dust, falling ash and embers, bloom, PS1 15-bit ordered dither |
| `src/vhs.py` | YIQ chroma bleed and delay, luma ringing, line jitter, head-switching noise, dropouts, snow, camcorder date overlay |
| `src/scenes.py` | the two shots: animation, camera moves, lamp flicker |
| `src/audio.py` | VHS treatment of the voice (pitch, wow/flutter, band-limit, crunch, dropouts, reverb), plus lamp clunks, hum, the switch click, the air-raid siren, radio static, wind, traffic-light relay ticks, distant metal clangs, a drone and tape hiss |
| `src/tts.py` | fetches the two voice lines from ElevenLabs (`eleven_v3`, voice "Jessica") into `assets/` |
| `src/timeline.py` | shared timings |

The image is rendered at 320×240, degraded at 640×480 and scaled to 960×720.

## Rebuild
Needs Python 3 with numpy and Pillow, and ffmpeg.

```sh
# only if you change the lines (the mp3s are committed)
ELEVENLABS_API_KEY=... python3 src/tts.py

python3 src/audio.py      # -> out/soundtrack.wav
python3 src/render.py     # -> out/alonso_2012_vhs.mp4  (~4 min on 4 cores)
```
