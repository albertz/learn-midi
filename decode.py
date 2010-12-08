# code under GPLv3
# Albert Zeyer - 2010-12-06

from subprocess import Popen, PIPE
import os
from struct import *

sample_mp3_file = "/Users/az/Music/Classic/Glenn Gould Plays Bach/French Suites, BWV812-7 - Gould/Bach, French Suite 1 in d, BWV812 - 1 Allemande.mp3"
sample_mid_file = "/Users/az/Programmierung/MidiWriter/mf_BachPrelude2.mid"

sampleRate = 44100

def ffmpeg_to_wav(stream):	
	p = Popen([
		"ffmpeg",
		"-i", "-",
		"-vn", "-acodec", "pcm_s16le",
		"-ac", "1", "-ar", str(sampleRate),
		"-f", "wav", "-"],
		stdin = stream, stdout = PIPE, stderr = open(os.devnull, "w"))
	return p.stdout
	
def midi_to_wav(stream):
	p = Popen([
		"timidity", "-",
		"-A90", "-a",
		"-Ow", "--output-mono", "--output-signed", "--output-16bit",
		"-o", "-"],
		stdin = stream, stdout = PIPE, stderr = open(os.devnull, "w"))	
	return p.stdout
	
def wav_to_rawpcm(stream):
	h_chunkid, h_chunksize, h_rifftype = unpack("<4sI4s", stream.read(12))
	if h_chunkid != "RIFF":
		raise Exception, "stream is not in RIFF format"
	if h_rifftype != "WAVE":
		raise Exception, "stream is not in WAVE format"
		
	if stream.read(4) != "fmt ": raise Exception, "fmt section expected"
	if unpack("<L", stream.read(4))[0] != 16: raise Exception, "fmt section should be of size 16"
	wFormatTag = unpack("<H", stream.read(2))[0]
	if wFormatTag != 1: raise Exception, "PCM format expected but got " + hex(wFormatTag)
	wChannels = unpack("<H", stream.read(2))[0]
	dwSamplesPerSec = unpack("<L", stream.read(4))[0]
	dwAvgBytesPerSec = unpack("<L", stream.read(4))[0]
	wBlockAlign = unpack("<H", stream.read(2))[0]
	wBitsPerSample = unpack("<H", stream.read(2))[0]
	
	if dwSamplesPerSec != sampleRate: raise Exception, "we expected " + str(sampleRate) + " Hz but got " + str(dwSamplesPerSec)
	if wBitsPerSample != 16: raise Exception, "we expected 16 bits per sample but got " + str(wBitsPerSample)
	if wBlockAlign != 2: raise Exception, "we expected block align 2 but got " + str(wBlockAlign)
	if wChannels != 1: raise Exception, "we expected 1 channel but got " + str(wChannels)
	
	if stream.read(4) != "data": raise Exception, "data section expected"
	datalen = unpack("<L", stream.read(4))[0]

	# and now the data begins ...
	return stream

	