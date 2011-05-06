import decode
from cStringIO import StringIO
import numpy
from array import array

def streamcopy(stream):
	#return stream
	s = StringIO()
	if hasattr(stream, "read"):
		c = 0
		while True:
			buf = stream.read(2048)
			if len(buf) == 0: break
			s.write(buf)
			c += len(buf)
	else:
		for data in stream:
			print type(data)
	print "read", c, "bytes"
	return s


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

