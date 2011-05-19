#!/usr/bin/python -u

import better_exchook
better_exchook.install()

from decode import *
from soundutils import *
import sys

if len(sys.argv) > 1:
	midifile = sys.argv[1]
else:
	midifile = sample_mid_file

print "loading ...",
midievents = midi_to_midievents(midifile)
rawpcm = midievents_to_rawpcm(midievents, gain=4.0)
rawpcm = streamcopy(rawpcm)
print "done"

from numpy.fft import rfft
import numpy as np
import math

RATE = 44100
N_window = RATE / 10
window = np.blackman(N_window)
fdata = []

def show_which_freq(freqs):
	which = freqs[1:].argmax() + 1
	# use quadratic interpolation around the max
	if which != len(freqs)-1:
		y0,y1,y2 = np.log(freqs[which-1:which+2:])
		x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
		# find the frequency and output it
		thefreq = (which+x1)*RATE/N_window
		print "The freq is %f Hz." % (thefreq)
	else:
		thefreq = which*RATE/N_window
		print "The freq is %f Hz." % (thefreq)

def resample(data, outdim):
	indim = len(data)
	outdata = [0.0] * outdim
	outentrylen = indim * 1.0 / outdim
	for outidx in xrange(outdim):
		x = 0.0
		startinidx = outidx * indim * 1.0 / outdim
		endinidx = (outidx + 1) * indim * 1.0 / outdim
		if not startinidx.is_integer():
			x += (1.0 - math.fmod(startinidx, 1)) * data[int(math.floor(startinidx))]
			inidx = int(math.ceil(startinidx))
		else:
			inidx = int(startinidx)
		while inidx < math.floor(endinidx):
			x += data[inidx]
			inidx += 1
		if not endinidx.is_integer() and int(math.floor(endinidx)) < len(data):
			x += math.fmod(endinidx, 1) * data[int(math.floor(endinidx))]
		x /= outentrylen
		outdata[outidx] = x
	return outdata

if __name__ == '__main__':

	while rawpcm.tell() < len(rawpcm.getvalue()):
		data = arrayFromPCMStream(rawpcm, N_window/10)
		fdata += list(data)
		if len(fdata) > N_window:
			fdata = fdata[len(fdata) - N_window:]
			
			freqs = rfft(window * fdata)
			freqs = abs(freqs) ** 2
			show_which_freqs(freqs)
		
		play(data)
	
	print
	print "finished"
