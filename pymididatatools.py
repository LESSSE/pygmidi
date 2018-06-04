
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
# >**Automatic Midi to and from PianoRoll Converters**: these take into account the orquestration that you want to use and use note velocity
# 
# >**Midi Counters**: fuctions to count ticks and quavers in a midi file
# 
# >**Resolution Changer**: function to change midi resolution (number of ticks in a quarter or crotchet note)
# 
# >**Tracks Midi Signature**: it returns an array of used and unused tracks
# 
# These operators allow to translate a set of heterogeneous midi files into a set of fixed size and similar orquestrated midi files.

# In[72]:


import numpy as np
import math
import gc
import os.path
import shutil
import pretty_midi
import midi
import utils.utils as utils
import glob
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from matplotlib import pyplot as plt

#np.set_printoptions(threshold=np.inf) 


# In[73]:


temp_file = "temp.mid"


# ## Inter Midi Classes Converters

# In[47]:


class gmidi:
    def save(src,path):
        if isinstance(src,midi.Pattern):
            midi.write_midifile(path, src)
        elif isinstance(src,pretty_midi.PrettyMIDI):
            src.write(path)
        elif type(src) is str:
            shutil.copy(src, path)
        else:
             raise TypeError('Unknown Type:'+str(type(src))+'is no python-midi.Pattern, pretty-midi.PrettyMIDI nor str')
        
        return path
        
    def load(path,out):
        if out is midi.Pattern:
            midi_data = midi.read_midifile(path)
        elif out is pretty_midi.PrettyMIDI:
            midi_data = pretty_midi.PrettyMIDI(path)
        elif type(out) is str:
            shutil.copy(path, out)
            midi_data = out
        else:
            raise TypeError('Unknown Type:'+str(type(src))+'is no python-midi.Pattern, pretty-midi.PrettyMIDI nor str')
        
        return midi_data
        
    
    def to(src,out):
        if type(out) is type and isinstance(src,out):
            return src
        else:
            gmidi.save(src,temp_file)
            midi_data=gmidi.load(temp_file,out)
            os.remove(temp_file)
            return midi_data
        
    def default_out(src):
        if type(src) is str:
            return src
        elif isinstance(src,pretty_midi.PrettyMIDI):
            return pretty_midi.PrettyMIDI
        elif isinstance(src,midi.Pattern):
            return midi.Pattern
        else:
            raise TypeError('Unknown Type:'+str(type(src))+'is no python-midi.Pattern, pretty-midi.PrettyMIDI nor str')
        


# ### Automatic MIDI Slicer

# In[48]:


def slice_midi(begin, end, src, out=None):
    '''Clip a midifile from a 'begin' tick to the 'end' tick.'''
    if out is None:
        out = gmidi.default_out(src)
        
    o_pattern = gmidi.to(src,midi.Pattern)
    n_pattern = midi.Pattern()
    n_pattern.resolution = o_pattern.resolution

    for t in range(len(o_pattern)):
        track = midi.Track()
        n_pattern.append(track)

    timeleft = [track[0].tick for track in o_pattern] #array with time for the next event for each track
    posns = [0 for track in o_pattern] #position in the list of events of each track
    playing = [{} for track in o_pattern]
    first_cmd = [True for track in o_pattern]
    lastcmdtime = [begin for t in o_pattern]
    time = 0

    condition = (time <= begin)
    while condition:

        if time > end:
            condition = False
            break

        for i in range(len(timeleft)): #For each track
            o_track = o_pattern[i]
            n_track = n_pattern[i]

            if not condition:
                break
            if time == begin:
                p = playing[i]
                for n in p:
                    evt = midi.NoteOnEvent(tick=0, pitch=n,channel=p[n]['channel'],velocity=p[n]['velocity'])
                    n_track.append(evt)
                    lastcmdtime[i] = time

            while timeleft[i] == 0:                    
                pos = posns[i]
                evt = o_track[pos]

                if isinstance(evt, midi.NoteEvent):
                    if isinstance(evt, midi.NoteOnEvent):
                        playing[i][evt.pitch] = {'velocity': evt.velocity, 'channel': evt.channel}
                    else:
                        playing[i].pop(evt.pitch)

                    if time < begin:
                        pass
                    elif time >= begin and time < end-1:
                        evt = evt.copy()
                        evt.tick = time - lastcmdtime[i]
                        n_track.append(evt)
                        lastcmdtime[i] = time
                    else:
                        condition = False
                        break
                elif isinstance(evt, midi.Event):
                    if time < begin:
                        evt = evt.copy()
                        evt.tick = 0
                        n_track.append(evt)
                    elif time >= begin and time < end-1:
                        evt = evt.copy()
                        evt.tick = time-lastcmdtime[i]
                        n_track.append(evt)
                        lastcmdtime[i] = time
                    else:
                        condition = False
                        break


                try:
                    timeleft[i] = o_track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

        if all(t is None for t in timeleft):
            break
        
        next_evt=min(list(filter(lambda x: x is not None, timeleft)))
        
        for i in range(len(timeleft)): #For each track
            if timeleft[i] is not None:
                timeleft[i] -= next_evt

        time += next_evt

    for u in range(len(n_pattern)):
        n_track = n_pattern[u]
        p = playing[u]
        for n in p:
            time=end-1
            evt = midi.NoteOffEvent(tick=time-lastcmdtime[u], pitch=n,channel=p[n]['channel'],velocity=0)
            n_track.append(evt)
            lastcmdtime[u] = time                             

        eot = midi.EndOfTrackEvent(tick=1)
        n_track.append(eot)

    return gmidi.to(n_pattern,out)


# ### Automatic MIDI Transposer

# In[49]:


#transpose to the sametone vs all tones
def transpose_midi(semitones, src, out=None):
    """"""
    if out is None:
        out = gmidi.default_out(src)
    
    pretty = gmidi.to(src,pretty_midi.PrettyMIDI)
        
    for instrument in pretty.instruments:
        # Don't want to shift drum notes
        if not instrument.is_drum:
            notes = instrument.notes
            n = 0
            while n < len(notes):
                notes[n].pitch += semitones
                if notes[n].pitch not in range(127):
                    notes.pop(n)
                    n-=1
                n+=1
                
    return gmidi.to(pretty,out)
   
    


# ### Automatic Midi Track Unifier

# In[50]:


class Instrumentation:
    
    def __init__(self, dic, array, drums=8):
        self.tracks = len(array)
        self.i_to_t = dic
        self.t_to_i = array
        self.drums = drums
        if not all(self.i_to_t[k] < self.tracks for k in self.i_to_t):
            raise ValueError('Destination tracks in dic must be one of represented in array - all(i_to_t[k] <= len(t_to_i) for k in i_to_t)')

    def orchestrate(self,src,out=None):
        def add_control_cmds(u,n_track):
            n_track.append(midi.ControlChangeEvent(tick=0, channel=self.t_to_i[u][0], data=[121, 0])) #reset channel definitions
            if self.t_to_i[u][1]>=0:
                n_track.append(midi.ProgramChangeEvent(tick=0,channel=self.t_to_i[u][0],data=[self.t_to_i[u][1]])) #instrument
            n_track.append(midi.ControlChangeEvent(tick=0, channel=self.t_to_i[u][0], data=[7, 80]))  #volume
            n_track.append(midi.ControlChangeEvent(tick=0, channel=self.t_to_i[u][0], data=[10, 64])) #span centered
            n_track.append(midi.ControlChangeEvent(tick=0, channel=self.t_to_i[u][0], data=[91, 10])) #reverb 
            n_track.append(midi.ControlChangeEvent(tick=0, channel=self.t_to_i[u][0], data=[93, 10])) #reverb chorus
            n_track.append(midi.PortEvent(tick=0, data=[0]))

        if out is None:
            out = gmidi.default_out(src)

        o_pattern = gmidi.to(src,midi.Pattern)    
        n_pattern = midi.Pattern() #new_pattern
        n_pattern.resolution = o_pattern.resolution 
        for t in range(self.tracks): 
            track = midi.Track()
            n_pattern.append(track)

        timeleft = [track[0].tick for track in o_pattern] #array with time for the next event for each track
        posns = [0 for track in o_pattern] #position in the list of events of each track
        track_destin = [False for track in o_pattern]
        first_cmd = [True for track in n_pattern]
        lastcmdtime = [0 for t in n_pattern]
        time = 0
        
        condition = True
        while condition:
            for i in range(len(o_pattern)): #For each track
                if not condition:
                    break
                while timeleft[i] == 0:
                    o_track = o_pattern[i]
                    pos = posns[i]
                    evt = o_track[pos]
                    u = track_destin[i]
                    
                    if isinstance(evt, midi.NoteEvent): #if it is a note event
                        if u is False: #Destination track defined then it must be equal to the upper one
                            track_destin[i] = track_destin[i-1]
                            u = track_destin[i]
                        if u is False or u is None: #If it continues not defined or it must not go to none of the tracks
                            pass
                        else:
                            if first_cmd[u]:
                                n_track = n_pattern[u]
                                add_control_cmds(u,n_track)
                                first_cmd[u] = False
                                
                            time_interval = (time - lastcmdtime[u])
                            evt.tick=time_interval
                            evt.channel=self.t_to_i[u][0]
                            n_track = n_pattern[u]
                            n_track.append(evt)
                            lastcmdtime[u] = time
                            
                    elif isinstance(evt, midi.ProgramChangeEvent):
                        if evt.channel == 9:
                            track_destin[i]=self.drums
                        else:
                            track_destin[i]=self.i_to_t.get(evt.data[0],None)
                            
                    elif isinstance(evt, midi.TimeSignatureEvent) or isinstance(evt, midi.SetTempoEvent):
                            if u is None:
                                u = 0
                            if first_cmd[u]:
                                n_track = n_pattern[u]
                                add_control_cmds(u,n_track)
                                first_cmd[u] = False    
                            n_track = n_pattern[u]
                            n_track.append(evt)
                            lastcmdtime[u] = time
                    else:
                        pass
                    
                    try:
                        timeleft[i] = o_track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None


            if all(t is None for t in timeleft):
                break

            next_evt=min(list(filter(lambda x: x is not None, timeleft)))

            for i in range(len(timeleft)): #For each track
                if timeleft[i] is not None:
                    timeleft[i] -= next_evt

            time += next_evt

        for u in range(len(n_pattern)):
            n_track = n_pattern[u]
            if first_cmd[u]:
                add_control_cmds(u,n_track)
                first_cmd[u] = False 
            eot = midi.EndOfTrackEvent(tick=1)
            n_track.append(eot)
            
        return gmidi.to(n_pattern,out)
    
    def original_instruments_in_track(self,n):
        list = []
        for i in self.i_to_t:
            if(self.i_to_t[i] == n):
                list += [pretty_midi.program_to_instrument_name(i)]
        return list
    
    def new_instrument_in_track(self,n):
        return pretty_midi.instrument_name_to_program(self.t_to_i[n])


# ### Automatic Midi to and from PianoRoll Converters

# In[51]:


def get_signature(src):
    '''Returns a binary array that represent the playing tracks of the midi file'''
    pattern = gmidi.to(src,midi.Pattern)

    posns = [0 for track in pattern] #position in the list of events of each track
    used = np.array([0 for track in pattern])
    
    for i in range(len(pattern)): #For each track
        track = pattern[i]
        pos = posns[i]
        evt = track[pos]
        while not used[i]:
            if isinstance(evt, midi.NoteOnEvent):
                used[i]=1
            try:
                posns[i] += 1 
                pos = posns[i]
                evt = track[pos]
            except IndexError:
                break
                
    return used


# In[52]:


def midi_to_pianoroll(src):
    midifile = "midifile.mid"
    midifile = gmidi.to(src,midifile)

    signature = get_signature(midifile)
    pianoroll = Multitrack(midifile)
    
    os.remove(midifile)
    
    return pianoroll, signature


# In[53]:


def pianoroll_to_midi(pianoroll,signature,out):  
    midifile = "midifile.mid"
    pianoroll.write(midifile)
        
    return gmidi.to(midifile, out)


# ### Counters

# In[54]:


def count_ticks(src):
    pattern = gmidi.to(src,midi.Pattern)
    
    sum2 = 0
    for i in pattern:
        sum1 = 0
        for n in i:
            sum1 += n.tick
        if sum1 > sum2:
            sum2 = sum1
    return sum2

# In[55]:


def count_quarters(src):
    pattern = gmidi.to(src,midi.Pattern)
    
    t = count_ticks(pattern)
    q = t/pattern.resolution
    
    return q

def count_tracks(src):
    pattern = gmidi.to(src,midi.Pattern)
    return len(pattern)

# ### Resolution Changer

# In[56]:


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

# In[63]:


def slice_songs(path_old, path_new, ticks_per_clip, mode="chuncking",next_function=None, bi=0,ei=None, verbose=False):
    def next_sliding(b,e,midifile):
        return b+inc, b+inc+ticks_per_clip
    
    if next_function==None:
        if mode=="chuncking":
            inc = ticks_per_clip
            next_function=next_sliding
        elif mode=="sliding":
            inc = midifile.resolution
            next_function =next_sliding
    
    n_files=0
    files = glob.glob('{}/*.mid*'.format(path_old))
    for f in tqdm(sorted(files)[bi:ei]):
        b, e = 0, ticks_per_clip
        slices=0
        name=os.path.basename(f)
        name=os.path.splitext(str(n_files).zfill(5)+name)[0]
        name=path_new+"/"+name
        old_pattern = midi.read_midifile(f) #old_pattern
        ticks = count_ticks(old_pattern)
        while(e < ticks):
            new_pattern = slice_midi(b,e,old_pattern)
            midi.write_midifile('{}-{}.mid'.format(name,str(slices)), new_pattern)
            if verbose:
                utils.eprint("\n",n_files,":",'{}-{}.mid'.format(name,str(slices))," - Ticks:",count_ticks(new_pattern)," - Tracks:", count_tracks(new_pattern))
            slices += 1
            n_files += 1
            b,e = next_function(b,e,old_pattern)


# In[71]:


def transpose_songs(path_old, path_new, minimum, maximum, bi=0,ei=None, verbose=False):
    files = glob.glob('{}/*.mid*'.format(path_old))
    n_files=0
    for f in tqdm(sorted(files)[bi:ei]):
        for i in range(maximum-minimum):
            name=os.path.basename(f)
            name=os.path.splitext(str(n_files).zfill(5)+name)[0]
            name=name+"-"+str(i)
            name=path_new+"/"+name+".mid"
            if verbose:
                utils.eprint("\n",n_files,":",name," - Ticks:",count_ticks(new_pattern)," - Tracks:", count_tracks(new_pattern))
            transpose_midi(i-minimum,f,name)
            n_files+=1


# In[75]:


def unif_songs(path_old, path_new, o2,bi=0,ei=None, verbose=False):
    files = glob.glob('{}/*.mid*'.format(path_old))
    n_files=0
    for f in tqdm(sorted(files)[bi:ei]):
        name=os.path.basename(f)
        name=os.path.splitext(str(n_files).zfill(5)+name)[0]
        old_pattern = midi.read_midifile(f) #old_pattern
        new_pattern = o2.orchestrate(old_pattern)
        if verbose:
            utils.eprint("\n",n_files,":",name," - Ticks:",count_ticks(new_pattern)," - Tracks:", count_tracks(new_pattern))
        midi.write_midifile('{}/{}.mid'.format(path_new,name), new_pattern)
        n_files+=1


# In[60]:

def pianoroll_songs(path_old, path_new,bi=0,ei=None, verbose=False):
    files = glob.glob('{}/*.mid*'.format(path_old))
    n_files=0
    for f in tqdm(sorted(files)[bi:ei]):
        name=os.path.basename(f)
        name=os.path.splitext(name)[0]
        pianoroll, a = midi_to_pianoroll(f)
        if verbose:
            utils.eprint("\n",n_files,":",'{}/{}{}'.format(path_new,name,"_pianoroll"),"|",'{}/{}{}'.format(path_new,name,"_sig")," - Ticks:",pianoroll.get_active_length()," - Tracks:",len(a))
        pianoroll.save('{}/{}{}'.format(path_new,name,"_pianoroll"))
        np.save('{}/{}{}'.format(path_new,name,"_sig"),[a])
        n_files+=1


# _____________________

# In[16]:


def Multitrack_to_tensor(multitrack,signature):
    mt = multitrack
    s = signature
    tensor = []
    j=0
    for i in s:
        if i == 0:
            tensor += [np.zeros((3456,128))]
        else:
            tensor += [mt.tracks[j].pianoroll]
            j+=1
    return tensor


# ___________________
