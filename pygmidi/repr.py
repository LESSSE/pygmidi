##LESSSE
##04 Abril 2021
##gmidi
##____________
##Representations
##____________

import midi as patlib
import pretty_midi as prelib
import pypianoroll as mullib
import mido as fillib
import midiutil as miulib
from midiarray import midiarray as arrlib
import os

#________PrettyMidi________
def pretty_save(src,path):
    src.write(path)
    return path

def pretty_load(path,res):
    midi_data = prelib.PrettyMIDI(path, resolution=res)
    return midi_data

pretty_repr = {"load": pretty_load, 
				"save": pretty_save, 
				"instance": prelib.PrettyMIDI(),
				"lib": prelib}

#________Midi_Pattern________
def change_res_pat(pattern,n_res):
        o_res = pattern.resolution
        
        if o_res == n_res:
            return pattern
        
        frac = n_res/o_res
        for t in pattern:
            for e in t:
                e.tick = int(round(frac*e.tick))
        pattern.resolution = n_res

        return pattern

def pattern_save(src,path):
    patlib.write_midifile(path, src)
    return path

def pattern_load(path,res):
    midi_data = patlib.read_midifile(path)
    change_res_pat(midi_data,res)
    return midi_data

pattern_repr = {"load": pattern_load,
				"save": pattern_save, 
				"instance": patlib.Pattern(),
				"lib": patlib}

#_____________Pianoroll_Multitrack____________
def multitrack_save(src,path):
    src.write(path)
    return path

def multitrack_load(path,res):
    pretty = prelib.PrettyMIDI(path,resolution=res)
    mul = mullib.from_pretty_midi(pretty, resolution=res)
    
    for t in mul.tracks:
        if t.name == "":
            if t.is_drum:
                t.name = "Drums"
            else:
                t.name = prelib.program_to_instrument_name(t.program)       
    return mul

multitrack_repr = {"load": multitrack_load,
                   "save": multitrack_save, 
                   "instance": mullib.Multitrack(),
                   "lib": mullib}

#______________MidiArray__________________
def midiarray_save(src,path):
    src.save(path)
    return path

def midiarray_load(path,res):
    midi_data = arrlib.MidiArray()
    midi_data.load(path,res)
    return midi_data

midiarray_repr = {"load": midiarray_load, 
					"save": midiarray_save, 
					"instance": arrlib.MidiArray(),
					"lib": arrlib}

#______________MidiFile_______________
def change_res_fil(midifile,n_res):
        o_res = midifile.ticks_per_beat
        
        if o_res == n_res:
            return midifile
        
        frac = n_res/o_res
        for t in midifile.tracks:
            for e in t:
                e.time = int(round(frac*e.time))
        midifile.ticks_per_beat = n_res

        return midifile

def midifile_save(src,path):
    src.save(path)
    return path

def midifile_load(path,res):
    midi_data = fillib.MidiFile(path)
    midi_data = change_res_fil(midi_data,res)
    return midi_data

midifile_repr = {"load": midifile_load, 
					"save": midifile_save, 
					"instance": fillib.MidiFile(),
					"lib": fillib}

#_____________________Reprs__________________

reprs = {mullib.Multitrack : multitrack_repr,
         patlib.Pattern : pattern_repr,
         prelib.PrettyMIDI: pretty_repr,
         arrlib.MidiArray: midiarray_repr,
         fillib.MidiFile: midifile_repr}