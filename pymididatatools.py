
# coding: utf-8

# # pymididatatools - A Midi Dataset Tools Collection for Python
# 
# Created by LESSSE - Luis Espirito Santo 
# 
# >**Inter Midi Classes Converters**: classe that encapsules the rom file to different classes
# 
# >**Automatic Midi Slicer**: splits midi in blocks with some fixed number of ticks
# 
# >**Automatic Midi Transposer**: transpose the midi file n semi-tones above
# 
# >**Automatic Midi Track Unifier**: cronverts a midi file in a new midi file with a fixed number of tracks using a intrument mapping
# 
# >**Midi Counters**: fuctions to count ticks and quavers in a midi file
# 
# >**Resolution Changer**: function to change midi resolution (number of ticks in a quarter or crotchet note)
# 
# >**Tracks Midi Signature**: it returns an array of used and unused tracks
# 
# These operators allow to translate a set of heterogeneous midi files into a set of fixed size and similar orquestrated midi files.

# In[1]:


import numpy as np
import math

import gc
import os.path
import shutil
import pretty_midi
import midi
import utils.utils as utils
import glob
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from matplotlib import pyplot as plt

#np.set_printoptions(threshold=np.inf) 


# ## Inter Midi Classes Converters

# In[16]:


class gmidi:
    def iterable(src):
        if gmidi.ismidi(src):
            src = [src]
        elif src is list:
            for i in src:
                if not gmidi.ismidi(i):
                    gmidi.typeError()
        else:
            gmidi.typeError()
        return src
    
    def typeError():
        raise TypeError('Unknown Type:'+str(type(src))+'is no python-midi.Pattern, pretty-midi.PrettyMIDI nor str')
    
    def ismidi(src):
        if isinstance(src,midi.Pattern) or isinstance(src,pretty_midi.PrettyMIDI) or isinstance(src,Multitrack) or type(src):
            return True
        else:
            return False
    
    def save(src,path):
        if isinstance(src,midi.Pattern):
            midi.write_midifile(path, src)
        elif isinstance(src,pretty_midi.PrettyMIDI):
            src.write(path)
        elif isinstance(src,Multitrack):
            src.write(path)
        elif type(src) is str:
            shutil.copy(src, path)
        else:
            gmidi.typeError()
        return path
    
    def get_instruments(src):
        '''Returns a binary array that represent the playing tracks of the midi file'''
        pattern = gmidi.to(src,midi.Pattern)
        pretty = gmidi.to(src,pretty_midi.PrettyMIDI)
        
        if len(pattern) != len(pretty.instruments):
            gmidi.to(pretty,src)
            pattern = gmidi.to(src,midi.Pattern)

        posns = [0 for track in pattern] #position in the list of events of each track
        instruments = [[False,0,False] for track in pattern]#[used,program,drums]

        for i in range(len(pattern)): #For each track
            track = pattern[i]
            pos = posns[i]
            evt = track[pos]
            while not instruments[i][0]:
                if isinstance(evt, midi.NoteOnEvent) and evt.data[1]!=0:
                    instruments[i][0]=True
                elif isinstance(evt, midi.ProgramChangeEvent):
                    instruments[i][1]=evt.data[0]
                    if evt.channel == 9:
                        instruments[i][2]=True
                try:
                    posns[i] += 1 
                    pos = posns[i]
                    evt = track[pos]
                except IndexError:
                    break
        return instruments

    def load_multitrack(path):
        mul = Multitrack(path,name=os.path.basename(path))
        ins = gmidi.get_instruments(path)
        
        if not ins[0][0] and not ins[0][1] and not ins[0][2]:
            ins = ins[1:]
        
        j=0
        tracks = []
        for i in ins:
            if i[0] == 0:
                program = i[1]
                is_drum = i[2]
                name = pretty_midi.program_to_instrument_name(program)
                if is_drum:
                    name = "Drums"
                t = Track(np.zeros(mul.tracks[0].pianoroll.shape,np.int8), program,is_drum,name)
                tracks += [t]
            else:
                program = i[1]
                is_drum = i[2]
                name = pretty_midi.program_to_instrument_name(program)
                if is_drum:
                    name = "Drums"
                mul.tracks[j].name=name
                tracks += [mul.tracks[j]]
                j+=1
        mul = Multitrack(tracks=tracks,tempo=mul.tempo, downbeat=mul.downbeat, beat_resolution=mul.beat_resolution, name=mul.name)
        
        return mul
    
    def load(path,out):
        if out is midi.Pattern:
            midi_data = midi.read_midifile(path)
        elif out is pretty_midi.PrettyMIDI:
            midi_data = pretty_midi.PrettyMIDI(path)
        elif out is Multitrack:
            midi_data = gmidi.load_multitrack(path)
        elif type(out) is str:
            shutil.copy(path, out)
            midi_data = out
        else:
            gmidi.typeError()
        
        return midi_data
        
    def to(src,out):
        temp_file = "temp.mid"
        
        if type(out) is type and isinstance(src,out):
            return src
        elif type(src) != str:
            gmidi.save(src,temp_file)
        else:
            temp_file = src
            
        midi_data=gmidi.load(temp_file,out)
        
        if type(src) != str:
            os.remove(temp_file)
            
        return midi_data
        
    def default_out(src):
        if type(src) is str:
            return src
        elif isinstance(src,pretty_midi.PrettyMIDI):
            return pretty_midi.PrettyMIDI
        elif isinstance(src,midi.Pattern):
            return midi.Pattern
        elif isinstance(src,Multitrack):
            return Multitrack
        else:
            gmidi.typeError()
        


# In[3]:


def count_ticks(src):
    mul = gmidi.to(src,Multitrack)
    
    return mul.tracks[0].pianoroll.shape[0]


# ### Automatic Midi Track Unifier

# In[4]:


class Instrumentation:
    def __init__(self, dic, array, drums=8):
        self.tracks = len(array)
        self.i_to_t = dic
        self.t_to_i = array
        self.drums = drums
        if not all(self.i_to_t[k] < self.tracks for k in self.i_to_t):
            raise ValueError('Destination tracks in dic must be one of represented in array - all(i_to_t[k] <= len(t_to_i) for k in i_to_t)')

    def orchestrate(self,src,out=None):
        if out is None:
            out = gmidi.default_out(src)

        mul = gmidi.to(src,Multitrack)
        new_mul = pypianoroll.copy(mul)
        
        new_mul.tracks=[]
        #print([i.name for i in new_mul.tracks])
        for t in range(len(self.t_to_i)):
            program = self.t_to_i[t][0]
            is_drum = self.t_to_i[t][1]
            name = pretty_midi.program_to_instrument_name(program)
            if is_drum:
                name = "Drums"
        
            new_mul.tracks += [Track(np.zeros(mul.tracks[0].pianoroll.shape,np.int8), program,is_drum,name)]
            #print([i.name for i in new_mul.tracks])
            if is_drum:
                check = lambda x,t: x.is_drum
            else:
                check = lambda x,t: self.i_to_t.get(x.program,None) == t and not x.is_drum
            for i in mul.tracks:
                if check(i,t):
                    new_mul.tracks += [i]
            new_mul.merge_tracks(list(range(t,len(new_mul.tracks))),program=program,is_drum=is_drum,name=name,remove_merged=True) 
        return gmidi.to(new_mul,out)
    
    def original_instruments_in_track(self,n):
        list = []
        for i in self.i_to_t:
            if(self.i_to_t[i] == n):
                list += [pretty_midi.program_to_instrument_name(i)]
        return list
    
    def new_instrument_in_track(self,n):
        return pretty_midi.instrument_name_to_program(self.t_to_i[n])


# ### Automatic MIDI Slicer

# In[5]:


def truncate_midi(src, begin, end, out=None):
    '''Clip a midifile from a 'begin' tick to the 'end' tick.'''
    if out is None:
        out = gmidi.default_out(src)
        
    mul = gmidi.to(src,Multitrack)
    
    mul = pypianoroll.copy(mul)
    
    for i in mul.tracks:
        i.pianoroll = i.pianoroll[begin:end]
   
    return gmidi.to(mul,out)


# In[6]:


def slice_midi(src, ticks_per_clip, mode="chuncking",next_function=None):
    def next_sliding(b,e,midifile):
        return b+inc, b+inc+ticks_per_clip
    
    if next_function==None:
        if mode=="chuncking":
            inc = ticks_per_clip
            next_function=next_sliding
        elif mode=="sliding":
            inc = midifile.resolution
            next_function =next_sliding

    midis=[]
    n = 0
    b, e = 0, ticks_per_clip
    mul = gmidi.to(src,Multitrack) #old_pattern
    ticks = count_ticks(mul)
    while(e < ticks):
        new_mul = truncate_midi(mul,b,e)
        new_mul.name = os.path.splitext(new_mul.name)[0]+"-"+str(n)
        midis+=[new_mul]    
        b,e = next_function(b,e,mul)
        n+=1
    
    return midis


# ### Automatic MIDI Transposer

# In[7]:


#transpose to the sametone vs all tones
def transpose(src, semitones, out=None):
    """"""
    if out is None:
        out = gmidi.default_out(src)
    
    mul = gmidi.to(src,Multitrack)
    mul = pypianoroll.copy(mul)
    for i in mul.tracks:
        pypianoroll.transpose(i,semitones)
                
    return gmidi.to(mul,out)


# In[8]:


def transpose_midi(src, minimum, maximum, out=None):
    """"""
    midis = []
    n = 0
    mul = gmidi.to(src,Multitrack) #old_pattern
    for i in range(maximum-minimum):
        new_mul = transpose(src,i-minimum)
        new_mul.name = os.path.splitext(new_mul.name)[0]+"-"+str(n)
        midis+=[new_mul] 
        n+=1
                
    return midis


# ### Resolution Changer

# In[9]:


def change_resolution(n_res,src, out=None):
    if out is None:
        out = gmidi.default_out(src)
    
    pattern = gmidi.to(src,midi.Pattern)
    
    o_res = pattern.resolution
    frac = n_res/o_res
    for t in pattern:
        for e in t:
            e.tick = int(round(frac*e.tick))
    pattern.resolution = n_res
    
    return gmidi.to(pattern, out)


# ____________________________

# ## Iterative Functions

# _____________________

# In[10]:


def multitrack_to_bool_tensor(multitrack):
    mt = multitrack
    tensor = []
    for t in mt.tracks:
        tensor += [t.pianoroll]
    t = np.array(tensor)
    #(tracks,timesteps,notes)
    t = np.swapaxes(t,0,1)
    #(timesteps,tracks,notes)
    t = np.swapaxes(t,1,2)
    #(timesteps,notes,tracks)
    t = np.reshape(t,(t.shape[0]//96,96,t.shape[1],t.shape[2]))
            
    return t


# _______________________________________________________________________________________

# In[11]:


root_f = "../epic_music_dataset/"
base_f = root_f + "1-Midi_Base"
unif_f = root_f + "2-Midi_Unif"
data_f = root_f + "3-Midi_Processed"


# ## Slicing Parameters

# In[12]:


#72 quartes -> least common multiple (4*2, 4*3, 4*4, 4* 4.5, 4*6)
def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).
    """
    while b:
        a, b = b, a%b
    return a

def lcm(a):
    lcm1 = a[0]
    for i in a[1:]:
        lcm1 = lcm1*(i//gcd(lcm1, i))
    return lcm1
        
number_bar = 8
beats_per_bar = [2,3,4]
divisions_per_beat = [1,2,3,4,6,8]
resolution = lcm(divisions_per_beat)
for i in range(len(beats_per_bar)):
    beats_per_bar += [int(beats_per_bar[i]*number_bar)]

ticks_per_slice = lcm(beats_per_bar) * resolution
print(ticks_per_slice)


# ## Orchestration Parameters

# In[13]:


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
track_to_midi = [[71,False], #woods
               [60,False], #brass
               [0,True],  #percussion
               [47,False], #timpani
               [14,False], #tubular bells 
               [52,False], #voices 
               [48,False], #strings
               [1,False]] #piano


# In[14]:


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

instruments = Instrumentation(midi_to_track, track_to_midi,drums=percussion)


# ______________________________________
# 
# ## Pre-Processing

# In[17]:


#Unification
path_old = base_f  
path_new = unif_f 
o2=instruments

files = glob.glob('{}/*.mid*'.format(path_old))
n_files=0
for f in tqdm(sorted(files)):
    name=os.path.basename(f)
    name=os.path.splitext(name)[0]
    mul = o2.orchestrate(f,'{}/{}.mid'.format(path_new,name))
    utils.eprint("\n",n_files,":",name)
    n_files+=1


# In[18]:


#Process
path_old = unif_f
path_new = data_f
ticks_per_clip = ticks_per_slice

files = glob.glob('{}/*.mid*'.format(path_old))
data = []
n_files=0
for f in tqdm(sorted(files)):
    slices = slice_midi(f,ticks_per_clip)
    for s in slices:
        transposed = transpose_midi(s,-6,6)
        for m in transposed:
            gmidi.to(m,'{}/{}.mid'.format(path_new,m.name))
            data += [multitrack_to_bool_tensor(m)]
    n_files+=1
    
data = np.array(data)


# ___________________
