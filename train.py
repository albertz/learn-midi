#!/usr/bin/env python -u
# by Albert Zeyer, www.az2000.de
# 2011-05-08

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
import pybrain.supervised as bt
import pybrain.datasets.sequential as bd


MIDINOTENUM = 128

print "preparing network ...",
nn = bn.RecurrentNetwork()
nn_in_origaudio = LinearLayer(1, name="audioin") # audio input, mono signal
nn_in_sampleraudio = LinearLayer(1, name="sampleraudio") # audio from midi sampler
nn_in_curmidikeys = LinearLayer(MIDINOTENUM, name="curmidikeys")
nn_in_curmidikeyvels = LinearLayer(MIDINOTENUM, name="curmidikeyvels")
nn_out_midikeys = LinearLayer(MIDINOTENUM, name="outmidikeys")
nn_out_midikeyvels = LinearLayer(MIDINOTENUM, name="outmidikeyvels")
nn_hidden_in = LSTMLayer(6, name="hidden")
nn_hidden_out = nn_hidden_in

nn.addModule(nn_hidden_in)
if nn_hidden_out is not nn_hidden_in: nn.addModule(nn_hidden_out)

nn.addRecurrentConnection(bc.FullConnection(nn_hidden_out, nn_hidden_in, name="recurrent_conn"))


TicksPerSecond = 100
AudioSamplesPerTick = 44100 / TicksPerSecond
MillisecsPerTick = 1000 / TicksPerSecond

class AudioIn:
	def __init__(self):
		self.stream = None
	def load(self, filename):
		self.pcmStream = streamcopy(ffmpeg_to_rawpcm(open(filename)))
	def getSamples(self, num):
		return arrayFromPCMStream(self.pcmStream, num)
		
audioIn = AudioIn()

class MidiSampler:
	def __init__(self):
		self.midiKeysState = [False] * MIDINOTENUM
		self.oldMidiKeysState = list(self.midiKeysState)
		self.midiKeysVelocity = [0.0] * MIDINOTENUM
		self.midiEventStream = []
		self.pcmStream = midievents_to_rawpcm(self.midiEventStream)		
	def tick(self):
		for note,oldstate,newstate,velocity in izip(count(), self.oldMidiKeysState, self.midiKeysState, self.midiKeysVelocity):
			if not oldstate and newstate:
				self.midiEventStream += [("noteon", 0, note, ev.velocity)]
			elif oldstate and not newstate:
				self.midiEventStream += [("noteoff", 0, note)]
		self.midiEventStream += [("play", MillisecsPerTick)]
		self.oldMidiKeysState = list(self.midiKeysState)
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
	(nn_in_sampleraudio, getMidiSamplerAudio_netInput),
	(nn_in_curmidikeys, getCurMidiKeys_netInput),
	(nn_in_curmidikeyvels, getCurMidiKeyVelocities_netInput),
]

NetOutputs = [
	(nn_out_midikeys, readMidiKeys_netOutput),
	(nn_out_midikeyvels, readMidiKeyVelocities_netOutput),
]

for i,(module,_) in enumerate(NetInputs):
	nn.addModule(module)
	nn.addConnection(bc.FullConnection(module, nn_hidden_in, name = "in_c" + str(i)))
for i,(module,_) in enumerate(NetOutputs):
	nn.addModule(module)
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


# The magic is happening here!
# See the code about how we are calculating the error.
# We just use bt.BackpropTrainer as a base.
# We ignore the target of the dataset though.
class ReinforcedTrainer(bt.BackpropTrainer):
	def __init__(self, module, rewarder, *args, **kwargs):
		bt.BackpropTrainer.__init__(self, module, *args, **kwargs)
		self.rewarder = rewarder # func (seq,last module-output) -> reward in [0,1]

	def _calcDerivs(self, seq):
		"""Calculate error function and backpropagate output errors to yield the gradient."""
		self.module.reset()
		for sample in seq:
			self.module.activate(sample[0])
		error = 0.
		ponderation = 0.
		for offset, sample in reversed(list(enumerate(seq))):
			subseq = itertools.imap(operator.itemgetter(0), seq[:offset+1])
			reward = self.rewarder(subseq, self.module.outputbuffer[offset])
			
			target = sample[1]
			outerr = target - self.module.outputbuffer[offset] # real err. if we are reinforcing, we are not allowed to use this

			# NOTE: We use the information/knowledge that the output must be in {0,1}.
			# This is a very strict assumption and the whole trick might not work when we generalize it.
			# normalize NN l,r output to {0.0,1.0}
			nl,nr = self.module.outputbuffer[offset]
			nl,nr = nl > 0.5, nr > 0.5
			nl,nr = nl and 1.0 or 0.0, nr and 1.0 or 0.0
			# guess target l,r
			gl = nl * reward + (1.0-nl) * (1.0-reward)
			gr = nr * reward + (1.0-nr) * (1.0-reward)
			
			outerr2 = (gl,gr) - self.module.outputbuffer[offset]
			#print "derivs:", offset, ":", outerr, outerr2
			outerr = outerr2
			
			error += 0.5 * sum(outerr ** 2)
			ponderation += len(target)
			# FIXME: the next line keeps arac from producing NaNs. I don't
			# know why that is, but somehow the __str__ method of the
			# ndarray class fixes something,
			str(outerr)
			self.module.backActivate(outerr)
		
		return error, ponderation


def rewardFunc(seq, nnoutput):
    seq = [ "123ABCXYZ"[pybrain.utilities.n_to_one(sample)] for sample in seq ]
    cl,cr = outputAsVec(SeqGenerator().nextSeq(seq)[-1])
    nl,nr = nnoutput
    reward = 0.0
    if nl > 0.5 and cl > 0.5: reward += 0.5
    if nl < 0.5 and cl < 0.5: reward += 0.5
    if nr > 0.5 and cr > 0.5: reward += 0.5
    if nr < 0.5 and cr < 0.5: reward += 0.5
    return reward

trainer = ReinforcedTrainer(module=nn, rewarder=rewardFunc)


import random
from itertools import *

def _highestBit(n):
	c = 0
	while n > 0:
		n /= 2
		c += 1
	return c

def _numFromBinaryVec(vec, indexStart, indexEnd):
	num = 0
	c = 1
	for i in xrange(indexStart,indexEnd):
		if vec[i] > 0.0: num += c
		c *= 2
	return num

def _numToBinaryVec(num, bits):
	vec = ()
	while num > 0 and len(vec) < bits:
		vec += (num % 2,)
		num /= 2
	if len(vec) < bits: vec += (0,) * (bits - len(vec))
	return vec




print "loaded data"
print calcDiff(midistr, mp3str, 10)
