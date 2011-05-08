#!/usr/bin/env python -u
# by Albert Zeyer, www.az2000.de
# 2011-05-08

import midi
from decode import *
from soundutils import *

from better_exchook import *
sys.excepthook = better_exchook

midistr = streamcopy(midi_to_rawpcm(open(sample_mid_file)))
mp3str = streamcopy(ffmpeg_to_rawpcm(open(sample_mp3_file)))


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
nn_out_midikeys = LinearLayer(MIDINOTENUM, name="outmidikeys")
nn_hidden_in = LSTMLayer(6, name="hidden")
nn_hidden_out = nn_hidden_in

nn.addInputModule(nn_in_origaudio) 
nn.addModule(nn_in_sampleraudio)
nn.addModule(nn_hidden_in)
if nn_hidden_out is not nn_hidden_in: nn.addModule(nn_hidden_out)
nn.addOutputModule(nn_inout_midikeys)

nn.addConnection(bc.FullConnection(nn_in_origaudio, nn_hidden_in, name="in_c1"))
nn.addConnection(bc.FullConnection(nn_in_sampleraudio, nn_hidden_in, name="in_c2"))
nn.addConnection(bc.FullConnection(nn_in_curmidikeys, nn_hidden_in, name="in_c3"))
nn.addConnection(bc.FullConnection(nn_hidden_out, nn_out_midikeys, name="out_c4"))
nn.addRecurrentConnection(bc.FullConnection(nn_hidden_out, nn_hidden_in, name="recurrent_conn"))

nn.sortModules()
print "done"


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


# a neural interface has some input and some output
# which can be used to communicate with a neural network
class NeuralInterface:
	class IO(LinearLayer):
		dim = 0
		def __init__(self, parent, **kwargs):
			LinearLayer.__init__(self, self.dim, **kwargs)
			self.parent = parent
		def update(self): pass
	Input = IO
	Output = IO
	DummyIO = IO
	def __init__(self):
		self.input = self.Input(self)
		self.output = self.Output(self)
	def update(self): self.output.update()

class OutputOnlyNeuralInterface(NeuralInterface):
	Input = NeuralInterface.DummyIO
	class Output(NeuralInterface.Output):
		def __init__(self, parent):
			self.dim = parent.outdim
			NeuralInterface.Output.__init__(self, parent)
	def __init__(self, outdim):
		self.outdim = outdim
		NeuralInterface.__init__(self)

class MemoryNeuralInterface(NeuralInterface):
	class Input(NeuralInterface.Input):
		dim = ActionDim
		def _forwardImplementation(self, inbuf, outbuf):
			super(Input, self)._forwardImplementation(inbuf, outbuf)
			action = netOutToAction(inbuf)
			memoryOut = action(self.parent.memory)
			self.parent.output.activate(memoryOutToNetIn(*memoryOut))
	class Output(NeuralInterface.Output):
		dim = MemoryActivationDim
	def __init__(self, memory):
		NeuralInterface.__init__(self)
		self.memory = memory


# General NN IO. slots:
# obj <->
#   NN:
#     id(obj)
#     id(left obj) or 0
#     id(right obj) or 0
#     additionalInfo(obj)

class GenericNeuralInterface(NeuralInterface):
	IdVecLen = 32
	@classmethod
	def idToVec(cls, id):
		if id is None: id = 0
		v = _numToBinaryVec(id, cls.IdVecLen)
		v = map(float, v)
		return tuple(v)
	@classmethod
	def vecToId(cls, v):
		id = _numFromBinaryVec(v)
		return id
	@classmethod
	def objToVec(cls, obj):
		if obj is None: return cls.idToVec(None)
		else: return cls.idToVec(id(obj))
	
	class Input(NeuralInterface.Input):
		def __init__(self, parent):
			self.dim = parent.inputVecLen()
			NeuralInterface.Input.__init__(self, parent)
		def _forwardImplementation(self, inbuf, outbuf):
			NeuralInterface.Input._forwardImplementation(self, inbuf, outbuf)
			levelInputs = [None] * self.parent.levelCount
			vecOffset = 0
			for level in xrange(self.parent.levelCount):
				newVecOffset = vecOffset + self.parent.inputVecLenOfLevel(level)
				levelInputs[level] = self.parent.vecToId(inbuf[vecOffset:newVecOffset])
				vecOffset = newVecOffset
			for level in xrange(self.parent.levelCount):
				if not self.parent.selectObjById(level, levelInputs[level]):
					# it means the objId is invalid
					# just ignore the rest
					break
			self.parent.update()
	# Output is activated through updateLevelOutput
	class Output(NeuralInterface.Output):
		def __init__(self, parent):
			self.dim = parent.outputVecLen()
			NeuralInterface.Output.__init__(self, parent)
	class LevelRepr:
		objList = []
		objDict = {} # id(obj) -> objList index
		curObj = None
		curObjIndex = 0
		def reset(self, newObjList):
			self.objList = newObjList
			self.objDict = dict(map(lambda (index,obj): (id(obj),index), enumerate(newObjList)))
			self.curObj = newObjList and newObjList[0] or None
			self.curObjIndex = 0
		def asOutputVec(self, objToVec, additionalInfoFunc):
			leftObj = (self.curObjIndex > 0) and self.objList[self.curObjIndex-1] or None
			rightObj = (self.curObjIndex+1 < len(self.objList)) and self.objList[self.curObjIndex+1] or None
			leftObjVec = objToVec(leftObj)
			curObjVec = objToVec(self.curObj)
			rightObjVec = objToVec(rightObj)
			additionalInfo = additionalInfoFunc(self.curObj)
			return leftObjVec + curObjVec + rightObjVec + additionalInfo
	def __init__(self, topLevelList, childsFuncs, additionalInfoFuncs):
		self.topLevelList = topLevelList
		self.childsFuncs = childsFuncs # list(obj -> list(obj))
		self.levelCount = len(childsFuncs) + 1
		additionalInfoFuncs = additionalInfoFuncs or ([None] * self.levelCount)
		assert len(additionalInfoFuncs) == self.levelCount
		additionalInfoFuncs = map(lambda f: (f or (lambda _:())), additionalInfoFuncs)
		self.additionalInfoFuncs = additionalInfoFuncs
		self.levels = map(lambda _: self.LevelRepr(), [None] * self.levelCount)
		NeuralInterface.__init__(self)
		self.resetLevel(0)
		self.update()
	
	def inputVecLenOfLevel(self, level): return self.IdVecLen
	def inputVecLen(self): return sum(map(self.inputVecLenOfLevel, range(self.levelCount)))
	def outputVecLenOfLevel(self, level): return 3 * self.IdVecLen + len(self.additionalInfoFuncs[level](None))
	def outputVecLen(self): return sum(map(self.outputVecLenOfLevel, range(self.levelCount)))
	
	def update(self):
		outVec = ()
		for level in xrange(self.levelCount):
			outVec += self.levels[level].asOutputVec(self.objToVec, self.additionalInfoFuncs[level])
		self.output.activate(outVec)
	def resetLevel(self, level):
		if level == 0: newObjList = self.topLevelList
		else:
			parentObj = self.levels[level-1].curObj
			if parentObj is None: newObjList = []
			else: newObjList = self.childsFuncs[level-1](parentObj)
		self.levels[level].reset(newObjList)
		if level+1 < len(self.levels):
			self.resetLevel(level + 1)
	def selectObjById(self, level, objId):
		if objId == id(self.levels[level].curObj): return True
		if objId in self.levels[level].objDict:
			self.levels[level].curObjIndex = idx = self.levels[level].objDict[objId]
			self.levels[level].curObj = self.levels[level].objList[idx]
			if level+1 < len(self.levels):
				self.resetLevel(level + 1)
			return True
		return False




print "loaded data"
print calcDiff(midistr, mp3str, 10)
