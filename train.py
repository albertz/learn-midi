#!/usr/bin/env python -u

import midi
from decode import *
from soundutils import *

from better_exchook import *
sys.excepthook = better_exchook

midistr = streamcopy(midi_to_rawpcm(open(sample_mid_file)))
mp3str = streamcopy(ffmpeg_to_rawpcm(open(sample_mp3_file)))

print "loaded data"
print calcDiff(midistr, mp3str, 10)
