"""Fetch the two voice lines from ElevenLabs into assets/.

Needs ELEVENLABS_API_KEY in the environment. The generated mp3s are
committed, so this only has to run when the lines change.
"""
import json
import os
import urllib.request

VOICE = 'cgSgspJ2msm6clMCkdW9'   # "Jessica - playful, bright, warm"
MODEL = 'eleven_v3'              # supports [excited] / [sad] audio tags

LINES = {
    'line1': '[excited] And yet again, perfection was not enough for Fernando Alonso... '
             '[cheerfully] to win his third championship!',
    'line2': '[sad] It went to Sebastian... and Red Bull... in 2012...',
}


def fetch(name, text, key):
    req = urllib.request.Request(
        f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128',
        data=json.dumps({'text': text, 'model_id': MODEL, 'voice_settings': {'stability': 0.5}}).encode(),
        headers={'xi-api-key': key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as r, open(f'assets/{name}.mp3', 'wb') as f:
        f.write(r.read())


if __name__ == '__main__':
    key = os.environ['ELEVENLABS_API_KEY']
    for name, text in LINES.items():
        fetch(name, text, key)
        print('wrote', f'assets/{name}.mp3')
