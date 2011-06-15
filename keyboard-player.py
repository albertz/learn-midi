#!/usr/bin/python

import sys
import Tkinter

app = Tkinter.Frame()
app.master.title("player")

def keypress(ev):
	print "press:", repr(ev.char), ev.keysym, ev.type, ev.time, ev.send_event

def keyrelease(ev):
	print "release:", repr(ev.char), ev.keysym, ev.type, ev.time, ev.send_event

app.master.bind("<KeyPress>", keypress)
app.master.bind("<KeyRelease>", keyrelease)

#app.bind("<KeyPress>", keyin)

app.mainloop()
