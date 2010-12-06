# code under GPLv3
# Albert Zeyer - 2010-12-06

from subprocess import Popen, PIPE
import os
from struct import *

sample_input_file = "/Users/az/Music/Classic/Glenn Gould Plays Bach/French Suites, BWV812-7 - Gould/Bach, French Suite 1 in d, BWV812 - 1 Allemande.mp3"

def pcm_stream(filename):
	sampleRate = 44100
	
	p = Popen([
		"ffmpeg",
		"-i", filename,
		"-vn", "-acodec", "pcm_s16le",
		"-ac", "1", "-ar", str(sampleRate),
		"-f", "wav", "-"],
		stdin = open(os.devnull), stdout = PIPE, stderr = open(os.devnull, "w"))
	
	h_chunkid, h_chunksize, h_rifftype = unpack("<4sI4s", p.stdout.read(12))
	if h_chunkid != "RIFF":
		raise Exception, "stream is not in RIFF format"
	if h_rifftype != "WAVE":
		raise Exception, "stream is not in WAVE format"
		
	if p.stdout.read(4) != "fmt ": raise Exception, "fmt section expected"
	if unpack("<L", p.stdout.read(4))[0] != 16: raise Exception, "fmt section should be of size 16"
	wFormatTag = unpack("<H", p.stdout.read(2))[0]
	if wFormatTag != 1: raise Exception, "PCM format expected but got " + hex(wFormatTag)
	wChannels = unpack("<H", p.stdout.read(2))[0]
	dwSamplesPerSec = unpack("<L", p.stdout.read(4))[0]
	dwAvgBytesPerSec = unpack("<L", p.stdout.read(4))[0]
	wBlockAlign = unpack("<H", p.stdout.read(2))[0]
	wBitsPerSample = unpack("<H", p.stdout.read(2))[0]
	
	if dwSamplesPerSec != sampleRate: raise Exception, "we expected " + str(sampleRate) + " Hz but got " + str(dwSamplesPerSec)
	if wBitsPerSample != 16: raise Exception, "we expected 16 bits per sample but got " + str(wBitsPerSample)
	if wChannels != 1: raise Exception, "we expected 1 channel but got " + str(wChannels)
	
	if p.stdout.read(4) != "data": raise Exception, "data section expected"
	datalen = unpack("<L", p.stdout.read(4))[0]

	# and now the data begins ...
	return p.stdout, wBlockAlign

	