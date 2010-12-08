#!/usr/bin/env python -u

import midi
from decode import *
from soundutils import *

midistr = streamcopy(wav_to_rawpcm(midi_to_wav(open(sample_mid_file))))
mp3str = streamcopy(wav_to_rawpcm(ffmpeg_to_wav(open(sample_mp3_file))))

print "loaded data"
print calcDiff(midistr, mp3str, 10)
