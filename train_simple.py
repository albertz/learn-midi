#!/usr/bin/env python -u
# by Albert Zeyer, www.az2000.de
# 2011-05-14

import midi
from decode import *
from soundutils import *

from better_exchook import *
sys.excepthook = better_exchook

#midistr = streamcopy(midi_to_rawpcm(open(sample_mid_file)))
#mp3str = streamcopy(ffmpeg_to_rawpcm(open(sample_mp3_file)))


import pybrain
import pybrain.tools.shortcuts as bs
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer, SoftmaxLayer
import pybrain.structure.networks as bn
import pybrain.structure.connections as bc
import pybrain.rl.learners.valuebased as bl
import pybrain.datasets.sequential as bd


MIDINOTENUM = 128

print "preparing network ...",
nn = bn.RecurrentNetwork()
nn_in_origaudio = LinearLayer(1, name="audioin") # audio input, mono signal
nn_out_midi = LinearLayer(MIDINOTENUM * 2, name="outmidi")
nn_hidden_in = LSTMLayer(5, name="hidden")
nn_hidden_out = nn_hidden_in

nn.addModule(nn_hidden_in)
if nn_hidden_out is not nn_hidden_in: nn.addModule(nn_hidden_out)

nn.addRecurrentConnection(bc.FullConnection(nn_hidden_out, nn_hidden_in, name="recurrent_conn"))


AudioSamplesPerSecond = 44100
TicksPerSecond = 100
AudioSamplesPerTick = AudioSamplesPerSecond / TicksPerSecond
MillisecsPerTick = 1000 / TicksPerSecond

class AudioIn:
	def __init__(self):
		self.stream = None
	def load(self, filename):
		self.pcmStream = streamcopy(ffmpeg_to_rawpcm(open(filename)))
	def getSamples(self, num):
		return arrayFromPCMStream(self.pcmStream, num)
		
audioIn = AudioIn()

def midistates_to_midievents(midistate_seq, oldMidiKeysState = (False,) * MIDINOTENUM):
	for midiKeysState, midiKeysVelocity in midistate_seq:
		for note,oldstate,newstate,velocity in izip(count(), oldMidiKeysState, midiKeysState, midiKeysVelocity):
			if not oldstate and newstate:
				yield ("noteon", 0, note, max(0, min(127, int(round(velocity)))))
			elif oldstate and not newstate:
				yield ("noteoff", 0, note)
		yield ("play", MillisecsPerTick)		
		oldMidiKeysState = tuple(midiKeysState)

class MidiSampler:
	def __init__(self):
		self.midiKeysState = [False] * MIDINOTENUM
		self.oldMidiKeysState = tuple(self.midiKeysState)
		self.midiKeysVelocity = [0.0] * MIDINOTENUM
		self.midiEventStream = []
		self.pcmStream = midievents_to_rawpcm(self.midiEventStream)		
	def tick(self):
		self.midiEventStream += list(midistates_to_midievents([(self.midiKeysState, self.midiKeysVelocity)], self.oldMidiKeysState))
		self.oldMidiKeysState = tuple(self.midiKeysState)
	def getSamples(self, num):
		return arrayFromPCMStream(self.stream, num)

midiSampler = MidiSampler()

def audioSamplesAsNetInput(audioSamples):
	audioSamples = map(lambda s: float(s) / 2**15, audioSamples)
	return sum(audioSamples) / len(audioSamples)

def getAudioIn_netInput():
	return audioSamplesAsNetInput(audioIn.getSamples(AudioSamplesPerTick))

def interpretOutMidiKeys(lastMidiKeysState, vec):
	newState = list(lastMidiKeysState)
	assert len(vec) == MIDINOTENUM
	for i in xrange(MIDINOTENUM):
		if vec[i] < 0.3 and lastMidiKeysState[i]:
			newState[i] = False
		elif vec[i] > 0.7 and not lastMidiKeysState[i]:
			newState[i] = True
	return newState

def readMidiKeys_netOutput(vec): pass

def interpretOutMidiKeyVelocities(vec):
	return list(vec)

def readMidiKeyVelocities_netOutput(vec): pass


def getCurMidiKeyVelocities_netInput():
	return list(midiSampler.midiKeysVelocity)

def midiKeysAsNetInput(midiKeysState):
	return map(lambda k: 1.0 if k else 0.0, midiKeysState)

def getCurMidiKeys_netInput():
	return midiKeysAsNetInput(midiSampler.midiKeysState)

def getMidiSamplerAudio_netInput():
	return audioSamplesAsNetInput(midiSampler.getSamples(AudioSamplesPerTick))

NetInputs = [
	(nn_in_origaudio, getAudioIn_netInput),
]

NetOutputs = [
	(nn_out_midi, None),
]

for i,(module,_) in enumerate(NetInputs):
	nn.addInputModule(module)
	nn.addConnection(bc.FullConnection(module, nn_hidden_in, name = "in_c" + str(i)))
for i,(module,_) in enumerate(NetOutputs):
	nn.addOutputModule(module)
	nn.addConnection(bc.FullConnection(nn_hidden_out, module, name = "out_c" + str(i)))

nn.sortModules()
print "done"

def netSetInputs():
	for module,inputFunc in NetInputs:
		module.activate(inputFunc())

def netReadOutputs():
	for module,outFunc in NetOutputs:
		outFunc(module.outputbuffer[module.offset])

def tick():
	netSetInputs()
	midiSampler.tick()
	nn.activate(())
	netReadOutputs()

import pybrain.supervised as bt
from numpy.random import normal
import random
from itertools import *
import os

def normal_only_pos(mean, std):
	y = normal(mean, std)
	if y < 0.0: y = -y
	return y

def normal_limit(mean, std, min, max):
	y = normal(mean, std)
	if y < min: y = 2 * min - y
	elif y > max: y = 2 * max - y
	if y < min or y > max: y = mean
	return y

def random_note():
	return int(round(normal_limit(MIDINOTENUM/2, std=MIDINOTENUM/4, min=0, max=MIDINOTENUM-1)))

def random_note_vel():
	return normal_only_pos(48, std=25)
	
def random_note_time():
	return int(round(normal_only_pos(200, std=500))) + 1

def generate_random_midistate_seq(millisecs):
	midiKeysState = [0] * MIDINOTENUM # or millisecs time of holding
	midiKeysVelocity = [0.0] * MIDINOTENUM
	for i in xrange(TicksPerSecond * millisecs / 1000):
		midiKeysState = map(lambda k: k - MillisecsPerTick, midiKeysState)
		
		numdown = len([k for k in midiKeysState if k >= 0])
		numdown_good = int(round(normal_only_pos(1, 4)))
		for _ in xrange(random.randint(0, max(numdown_good - numdown, 0))):
			note = random_note()
			midiKeysState[note] = random_note_time()
			midiKeysVelocity[note] = random_note_vel()

		midiKeysVelocity = map(lambda (s,v): v if s >= 0 else 0.0, izip(midiKeysState, midiKeysVelocity))
		yield (map(lambda s: s >= 0, midiKeysState), midiKeysVelocity)

def generate_silent_midistate_seq(millisecs):
	midiKeysState = [False] * MIDINOTENUM
	midiKeysVelocity = [0.0] * MIDINOTENUM
	for i in xrange(TicksPerSecond * millisecs / 1000):
		yield (midiKeysState, midiKeysVelocity)

def generate_seq(maxtime):
	millisecs = maxtime #random.randint(1,20) * 1000
	midistate_seq = list(generate_random_midistate_seq(millisecs))
	midievents_seq = list(midistates_to_midievents(midistate_seq))
	pcm_stream = streamcopy(midievents_to_rawpcm(midievents_seq))
	
	delaytime = 100
	# add delaytime ms silence at beginning so that the NN can operate a bit on the data
	midistate_seq = list(generate_silent_midistate_seq(delaytime)) + midistate_seq
	# add delaytime ms silence at ending (chr(0)*2 for int16(0))
	pcm_stream.seek(0, os.SEEK_END)
	pcm_stream.write(chr(0) * 2 * (AudioSamplesPerSecond * delaytime / 1000))
	pcm_stream.seek(0)
	millisecs += delaytime
	
	for tick in xrange(TicksPerSecond * millisecs / 1000):
		#print "XXX", tick, pcm_stream.tell(), len(pcm_stream.getvalue()), millisecs, TicksPerSecond * millisecs / 1000
		audio = audioSamplesAsNetInput(arrayFromPCMStream(pcm_stream, AudioSamplesPerTick))
		midikeystate,midikeyvel = midistate_seq[tick]
		midikeystate = map(lambda s: 1.0 if s else 0.0, midikeystate)		
		yield (audio, midikeystate + midikeyvel)

def addSequence(dataset, maxtime):
    dataset.newSequence()
    for i,o in generate_seq(maxtime):
        dataset.addSample(i, o)

def generateData(nseq, maxtime):
    dataset = bd.SequentialDataSet(1, MIDINOTENUM*2)
    for i in xrange(nseq): addSequence(dataset, maxtime)
    return dataset





if __name__ == '__main__':
	import thread
	def userthread():
		from IPython.Shell import IPShellEmbed
		ipshell = IPShellEmbed()
		ipshell()
	#thread.start_new_thread(userthread, ())
	
	maxtime = 50
	nseq = 10
	
	import pickle
	try:
		maxtime, nn.params[:] = pickle.load(open("nn_params.dump"))
	except Exception, e:
		print e
		print "ignoring and continuing..."

	from pybrain.tools.validation import ModuleValidator
	import pybrain.supervised as bt
	#trainer = bt.BackpropTrainer(nn, learningrate=0.0001, momentum=0.1)
	#trainer = bt.RPropMinusTrainer(nn)
	
	def eval_nn(params):
		global nn, trndata
		nn.reset()
		nn.params[:] = params
		return ModuleValidator.MSE(nn, trndata)
	
	import pybrain.optimization as bo
	theparams = nn.params
	thetask = eval_nn
	maxEvals = 1000
	print "len params:", len(nn.params)
	
	tstresults = []
	# carry out the training
	while True:
		print "generating data (maxtime = " + str(maxtime) + ") ...",
		trndata = generateData(nseq = nseq, maxtime = maxtime)
		tstdata = generateData(nseq = nseq, maxtime = maxtime)
		#trainer.setData(trndata)
		print "done"
		#trainer.train()
		
		params, besterror = bo.ExactNES(thetask, theparams, maxEvaluations=maxEvals, minimize=True).learn()
		nn.params[:] = params
		print "best error:", besterror
		
		print "max param:", max(map(abs, nn.params))
		nn.params[:] = map(lambda x: max(-1.0, min(1.0, x)), nn.params)
		trnresult = 100. * (ModuleValidator.MSE(nn, trndata))
		tstresult = 100. * (ModuleValidator.MSE(nn, tstdata))
		print "train error: %5.2f%%" % trnresult, ",  test error: %5.2f%%" % tstresult
		
		tstresults += [tstresult]
		if len(tstresults) > 10: tstresults.pop(0)
		if len(tstresults) >= 10:
			print "test error sum of last 10 episodes:", sum(tstresults)
			if sum(tstresults) < 50.0:
				pickle.dump(nn.params, open("nn_params.dump", "w"))
				tstresults = []
				maxtime += 100
				
		#s = getRandomSeq(100, ratevarlimit=random.uniform(0.0,1.0))
		#print " real:", seqStr(s)
		#print "   nn:", getSeqOutputFromNN(nn, s)
