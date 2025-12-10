#!/bin/python
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
import os

model_path = os.path.expanduser('~/models/en_US-john-medium.onnx')
voice = PiperVoice.load(model_path)
stream = sd.OutputStream(
    samplerate=voice.config.sample_rate, 
    channels=1, 
    dtype='int16',
)

stream.start()
text = input()

for audio_bytes in voice.synthesize(text):
    audio_bytes = audio_bytes.audio_int16_bytes
    int_data = np.frombuffer(audio_bytes, dtype=np.int16)
    stream.write(int_data)

stream.stop()
stream.close()
