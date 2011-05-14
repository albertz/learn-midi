import decode
from cStringIO import StringIO
import numpy
from array import array
from fluidsynth import raw_audio_string

def streamcopy(stream):
	s = StringIO()
	c = 0
	if hasattr(stream, "read"):
		while True:
			buf = stream.read(2048)
			if len(buf) == 0: break
			s.write(buf)
			c += len(buf)
	else:
		for data in stream:
			if type(data) is numpy.ndarray:
				buf = raw_audio_string(data)
			elif type(data) is str:
				buf = str(data)
			else:
				assert False, "cannot handle " + repr(data)
			s.write(buf)
			c += len(buf)
	return s

# kind of the reverse of fluidsynth.raw_audio_string
def arrayFromPCMStream(str, n):
	data = array('h')
	#oldpos = str.tell()
	data.fromstring(str.read(n * 2))
	#str.seek(oldpos) -- this would reset the pos. but actually, we want to go forward in the stream. so keep this commented out
	data = numpy.array(data, numpy.int16)
	return data
	
def calcDiff(str1, str2, dt):
	numsamples = int(decode.sampleRate * dt)
	diffdata = arrayFromPCMStream(str1, numsamples) - arrayFromPCMStream(str2, numsamples)
	return numpy.linalg.norm(diffdata)

import pyaudio
pa = pyaudio.PyAudio()
strm = pa.open(
			   format = pyaudio.paInt16,
			   channels = 1, 
			   rate = 44100, 
			   output = True)

def play(s):
	if type(s) is numpy.ndarray: s = raw_audio_string(s)
	elif type(s) is not str: s = s.getvalue()
	strm.write(s)
