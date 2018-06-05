
# coding: utf-8

# # Epic Dataset Creation

# In[1]:


import pymididatatools as pmdt
import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt

# ## Folders

# In[2]:

root_f = "../epic_music_dataset/"
base_f = root_f + "1-Midi_Base"
block_f = root_f + "2-Midi_Blocks"
trans_f = root_f + "3-Midi_Transposed"
unif_f = root_f + "4-Midi_Unif"
piarol_f = root_f + "5-Pianoroll"


# ## Slicing Parameters

# In[3]:

#72 quartes -> least common muiltiple (4*2, 4*3, 4*4, 4* 4.5, 4*6)
def gcd(a, b):
    while b:
        a, b = b, a%b
    return a

def lcm(a):
    lcm1 = a[0]
    for i in a[1:]:
        lcm1 = lcm1*(i//gcd(lcm1, i))
    return lcm1
        
number_bar = 2
quarters_per_bar = [2,3,4,4.5,6]
quarters = []
for i in range(len(quarters_per_bar)):
    quarters += [int(quarters_per_bar[i]*number_bar)]

print(quarters,lcm(quarters))    
ticks_per_slice = lcm(quarters) * 480

def next_function(b,e,midifile):
    for evt in midifile[0]:
        if isinstance(evt, midi.TimeSignatureEvent) and evt.tick < b:
            print(evt.data[0],evt.data[1])
            inc = evt.data[0]*evt.data[1]/4*midifile.resolution
    
    if inc == 0:
        raise Exception("Inc = 0")
    
    return b+inc, b+inc+ticks_per_clip


# ## Orchestration Parameters

# #### 7 Tracks

# In[4]:


#7 Tracks
flutes = 0
oboes = 0
clarinets = 0
saxofones = 0
bassoon = 0
french_horns = 1
trumpets = 1
tubas = 1
percussion = 2
timpani = 3
tubular_bells = 4
chromatic_perc = 4
harp = 6
piano = 7
organ = 7
voices = 5
guitars = 6
basses  = 6
strings = 6


#intruments to atribute to each one of the new tracks
track_to_midi = [[0,71], #woods
               [1,60], #brass
               [9,0],  #percussion
               [2,47], #timpani
               [3,14], #tubular bells 
               [4,52], #voices 
               [5,48], #strings
               [6,1]] #piano


# #### 19 Tracks

# In[5]:


#maps one midi instrument code to the respective destination track
midi_to_track = { 72: flutes, 73: flutes, 74: flutes, 75: flutes, 76: flutes, 77: flutes, 78: flutes, 79: flutes, #flutes and piccolos
                    68: oboes, 69: oboes, #oboe and french horn
                    71: clarinets, #clarinets
                    64: saxofones, 65: saxofones, 66: saxofones, 67: saxofones, #saxofone
                    70: bassoon, # bassoon
                    60: french_horns, 61: french_horns, 62: french_horns, 63: french_horns, #french horns
                    56: trumpets, 59: trumpets, #trumpets
                    57: tubas, 58: tubas, #tuba and trombones
                    112: percussion, 113: percussion, 114: percussion, 115: percussion, 116: percussion, 117: percussion, 118: percussion, 119: percussion, #percussive
                    47: timpani, #timpani
                    14: tubular_bells, #tubular bells
                    8 : chromatic_perc, 9 : chromatic_perc, 10: chromatic_perc, 11: chromatic_perc, 12: chromatic_perc, 13: chromatic_perc, #chromatic perc
                    46: harp, #harp
                    0 : piano, 1 : piano, 2 : piano, 3 : piano, 4 : piano, 5 : piano, 6 : piano, 7 : piano, 15: piano, #piano and keyboard
                    16: organ, 17: organ, 18: organ, 19: organ, 20: organ, 21: organ, 22: organ, 23: organ, #accordion and organ
                    52: voices, 53: voices, 54: voices, #voices
                    24: guitars, 25: guitars, 26: guitars, 27: guitars, 28: guitars, 29: guitars, 30: guitars, 31: guitars, #guitars
                    32: basses, 33: basses, 34: basses, 35: basses, 36: basses, 37: basses, 38: basses, 39: basses, #basses
                    40: strings, 41: strings, 42: strings, 43: strings, 48: strings, 49: strings, 50: strings, 51: strings, 55: strings, 44: strings, 45: strings,#strings
                  }

instruments = pmdt.Instrumentation(midi_to_track, track_to_midi,drums=percussion)


# _______________________________________
# 
# ## Pre-Processing

# In[6]:

print("Slices")
pmdt.slice_songs(base_f,block_f,ticks_per_slice,verbose=True)

# In[1]:
print("Transpose")
pmdt.transpose_songs(block_f,trans_f,-6,6,verbose=True)


# In[2]:
print("Unify")
pmdt.unif_songs(trans_f,unif_f,instruments,verbose=True)


# In[3]:
print("Pianoroll")
pmdt.pianoroll_songs(unif_f,piarol_f,verbose=True)

# ____________________________________
