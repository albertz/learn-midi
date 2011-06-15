#!/usr/bin/python

import better_exchook
better_exchook.install()

import sys, os
from ctypes import *	
sdl = None

def app_main():
	sdl.SDL_Init.argtypes = (c_uint32,)
	sdl.SDL_Init(0xFFFF) # init everything
	
	c_SDLSurface_p = c_void_p # this suffice for us
	sdl.SDL_SetVideoMode.restype = c_SDLSurface_p # screen
	sdl.SDL_SetVideoMode.argtypes = (c_int, c_int, c_int, c_uint32) # width,height,bpp,flags
	
	screenSurface = sdl.SDL_SetVideoMode(640,480,0,0)

	c_SDLKey = c_uint32
	c_SDLMod = c_uint32
	
	class c_SDLkeysym(Structure):
		_fields_ = [
			("scancode", c_uint8),
			("sym", c_SDLKey),
			("mod", c_SDLMod),
			("unicode", c_uint16)
			]
	
	class c_SDLKeyboardEvent(Structure):
		_fields_ = [
			("type", c_uint8),
			("which", c_uint8), # which keyboard device
			("state", c_uint8), # 1 - down, 0 - up
			("keysym", c_SDLkeysym)
			]
	
	class c_SDLDummyEvent(Structure):
		_fields_ = [
			("type", c_uint8),
			("data", c_uint8*20) # just some space filler so that we are big enough
			]
	
	class c_SDLEvent(Union):
		_fields_ = [
			("type", c_uint8),
			("key", c_SDLKeyboardEvent),
			("dummy", c_SDLDummyEvent)
			]
	
	class SDLEventTypes:
		# just the ones we need for now
		SDL_KEYDOWN = 2
		SDL_KEYUP = 3
		SDL_QUIT = 12
		
	ev = c_SDLEvent()
		
	sdl.SDL_WaitEvent.argtypes = (POINTER(c_SDLEvent),)
	while sdl.SDL_WaitEvent(pointer(ev)) == 1:
		if ev.type == SDLEventTypes.SDL_QUIT: break
		elif ev.type in [SDLEventTypes.SDL_KEYDOWN, SDLEventTypes.SDL_KEYUP]:
			down = ev.key.state != 0
			sym = ev.key.keysym.sym
			if sym <= 127: sym = chr(sym)			
			print "SDL keyboard event:", down, repr(sym)
			
			if down and sym == 'q': break
			if down and sym == '\x1b': break # ESC
			
			
	sdl.SDL_Quit.restype = None
	sdl.SDL_Quit()

if sys.platform == "darwin":

	sdl = cdll.LoadLibrary("/Library/Frameworks/SDL.framework/SDL")

	from AppKit import NSApp, NSApplication, NSNotificationCenter, NSApplicationDidFinishLaunchingNotification
	from Foundation import NSAutoreleasePool, NSObject
	pool = NSAutoreleasePool.alloc().init()
	
	class MyApplicationActivator(NSObject):
	
		def activateNow_(self, aNotification):
			NSApp().activateIgnoringOtherApps_(True)
			try:
				app_main()
			except:
				sys.excepthook(*sys.exc_info())
			os._exit(0)
			
	activator = MyApplicationActivator.alloc().init()
	NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
		activator,
		'activateNow:',
		NSApplicationDidFinishLaunchingNotification,
		None,
	)
	
	NSApplication.sharedApplication()
	NSApp().run()
	
	del pool

else:
	app_main()
	
