#!/usr/bin/python

from decode import *
from soundutils import *
import train_simple as t

import sys
if len(sys.argv) > 1:
	millisecs = int(sys.argv[1])
else:
	millisecs = 10000

midistate_seq = t.generate_random_midistate_seq(millisecs)
midievents_seq = t.midistates_to_midievents(midistate_seq)


def midiEventHook(stream):
	for ev in stream:
		print ev
		yield ev

midievents_seq = midiEventHook(midievents_seq)
rawpcm = midievents_to_rawpcm(midievents_seq)

for data in rawpcm:
	play(data)
