#!/usr/bin/python

from decode import *
from soundutils import *
import sys

if len(sys.argv) > 1:
	midifile = sys.argv[1]
else:
	midifile = sample_mid_file

def midiEventHook(stream):
	for ev in stream:
		print ev
		yield ev

midievents = midi_to_midievents(midifile)
midievents = midiEventHook(midievents)
rawpcm = midievents_to_rawpcm(midievents)

for data in rawpcm:
	play(data)
