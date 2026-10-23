# 3d-vhs — "Perfection was not enough" (2012)

A 25-second Silent Hill style 3D clip on a worn VHS tape, made entirely in code: a
numpy software rasterizer, no Blender and no 3D packages. The drivers are realistic
high-poly models whose faces are real photographs projected onto sculpted heads.

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
| `src/engine.py` | triangle soup meshes, primitives (box, loft, sphere), camera, near-plane clipping, a rasterizer with a vectorised path for the many tiny triangles of high-poly meshes and alpha-tested textures; it writes a visibility buffer that is then resolved with perspective-correct UVs |
| `src/heads.py` | photo heads: a dense head mesh sculpted to the landmarks of a frontal photo (nose, brow, sockets, cheekbones, lips, chin), the photo projected on it and baked into its texture; hair grown out of the scalp with its cut-out taken from the photo, or a cap for Vettel |
| `src/models.py` | smooth high-poly characters on a small FK rig (hands with fingers, boots, pit-crew helmets), the RB8, the Fresnel theatre lamp |
| `src/sets.py` | the two locations: the Silent Hill street (shopfronts and signs, utility poles and wires, traffic light, parked cars, the torn road and abyss) and the rusted Otherworld cage |
| `src/textures.py` | 1024px race suits (Ferrari 2012: Santander, Shell, Scuderia Ferrari panel, Italian piping, FERNANDO belt; Red Bull: bulls, Infiniti, Total), sleeves and legs, signage, livery, street and Otherworld surfaces |
| `src/shade.py` | spotlight with a PCF shadow map, light bouncing off the floor, distance fog and sky light, ray-marched volumetric beam (half resolution), dust, falling ash and embers, bloom |
| `src/vhs.py` | YIQ chroma bleed and delay, luma ringing, line jitter, head-switching noise, dropouts, snow, camcorder date overlay |
| `src/scenes.py` | the two shots: animation, camera moves, lamp flicker |
| `src/audio.py` | VHS treatment of the voice (pitch, wow/flutter, band-limit, crunch, dropouts, reverb), plus lamp clunks, hum, the switch click, the air-raid siren, radio static, wind, traffic-light relay ticks, distant metal clangs, a drone and tape hiss |
| `src/tts.py` | fetches the two voice lines from ElevenLabs (`eleven_v3`, voice "Jessica") into `assets/` |
| `src/timeline.py` | shared timings |

The image is rendered at 640×480, degraded like a VHS tape and scaled to 960×720.

## Faces
- `assets/faces/alonso.webp`: the photo of Alonso supplied for this project (not committed;
  put it there to rebuild).
- `assets/faces/vettel.jpg`: [Sebastian Vettel - Press conference, 2010 German Grand Prix
  (cropped)](https://commons.wikimedia.org/wiki/File:Sebastian_Vettel_-_Press_conference,_2010_German_Grand_Prix_(cropped).jpeg),
  original by LG전자, derivative by Danyele, CC BY 2.0.

## Rebuild
Needs Python 3 with numpy and Pillow, and ffmpeg.

```sh
# only if you change the lines (the mp3s are committed)
ELEVENLABS_API_KEY=... python3 src/tts.py

python3 src/audio.py      # -> out/soundtrack.wav
python3 src/render.py     # -> out/alonso_2012_vhs.mp4  (~4 min on 4 cores)
```
