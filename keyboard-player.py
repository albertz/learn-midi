#!/usr/bin/python

import sys

from Foundation import NSAutoreleasePool
pool = NSAutoreleasePool.alloc().init()

from ctypes import *

sdl = cdll.LoadLibrary("/Library/Frameworks/SDL.framework/SDL")

sdl.SDL_Init.argtypes = (c_uint32,)
print "SDL_Init:", sdl.SDL_Init(0xFFFF) # init everything



# foo

sdl.SDL_Quit.restype = None
sdl.SDL_Quit()

del pool
