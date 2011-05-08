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
			buf = raw_audio_string(data)
			s.write(buf)
			c += len(buf)
	print "read", c, "bytes"
	return s

# kind of the reverse of fluidsynth.raw_audio_string
def arrayFromPCMStream(str, n):
	data = array('h')
	oldpos = str.tell()
	data.fromstring(str.read(n * 2))
	str.seek(oldpos)
	data = numpy.array(data, numpy.int16)
	return data
	
def calcDiff(str1, str2, dt):
	numsamples = int(decode.sampleRate * dt)
	diffdata = arrayFromPCMStream(str1, numsamples) - arrayFromPCMStream(str2, numsamples)
	return numpy.linalg.norm(diffdata)

def play(s):
	import pyaudio
	pa = pyaudio.PyAudio()
	strm = pa.open(
				   format = pyaudio.paInt16,
				   channels = 2, 
				   rate = 44100, 
				   output = True)
	if type(s) is not str: s = s.getvalue()
	strm.write(s)
