
# coding: utf-8

# In[13]:

import sys
import shutil
import os.path
import midi as patlib
import pretty_midi as prelib
import pypianoroll as mullib
import numpy as np
import matplotlib as mpl
import gmidi.plot as plot
import gmidi.pyormidict
from matplotlib import pyplot as plt


# In[2]:


#_____Pretty______
def pretty_save(src,path):
    src.write(path)
    return path

def pretty_load(path,res):
    midi_data = prelib.PrettyMIDI(path,resolution=res)
    return midi_data

pretty_repr = {"load": pretty_load, "save": pretty_save, "instance": prelib.PrettyMIDI()}


# In[3]:


#_____Pattern______

def change_res(pattern,n_res):
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
    pretty = pretty_load(path,480)
    if len(midi_data) < len(pretty.instruments):
        pretty_save(pretty,path)
    midi_data = patlib.read_midifile(path)
    change_res(midi_data,res)
    
    used = False
    program = 0
    is_drum = 0
    name = None    

    track = midi_data[0]
    pos = 0
    
    while (not used and not program and not is_drum):
        evt = track[pos]
        
        if isinstance(evt, patlib.NoteOnEvent) and evt.data[1]!=0:
            used=True
        elif isinstance(evt, patlib.ProgramChangeEvent):
            program=evt.data[0]
            if evt.channel == 9:
                is_drum=True
        try:
            pos += 1 
            evt = track[pos]
        except IndexError:
            midi_data = midi_data[1:]
            track = midi_data[0]
            pos = 0
    
    return midi_data

pattern_repr = {"load": pattern_load, "save": pattern_save, "instance": patlib.Pattern()}


# In[4]:


#_____Multitrack________
def get_instruments(path):
    '''Returns an array that represents track information of one .mid file'''
    src=path
    pattern = pattern_load(src,480)
    pretty = pretty_load(src,480)

    if len(pattern) < len(pretty.instruments):
        #utils.eprint("Diff:",len(pattern),len(pretty.instruments))
        pretty_save(pretty,path)
        pattern = pattern_load(src,480)

    posns = [0 for track in pattern] #position in the list of events of each track
    instruments = [{"used":False,"program":0,"is_drum":False,"name":None} for track in pattern]

    for i in range(len(pattern)): #For each track
        track = pattern[i]
        pos = posns[i]
        evt = track[pos]
        while not instruments[i]["used"]:
            if isinstance(evt, patlib.TrackNameEvent):
                instruments[i]["name"]=evt.text
            elif isinstance(evt, patlib.NoteOnEvent) and evt.data[1]!=0:
                instruments[i]["used"]=True
            elif isinstance(evt, patlib.ProgramChangeEvent):
                instruments[i]["program"]=evt.data[0]
                if evt.channel == 9:
                    instruments[i]["is_drum"]=True
            try:
                posns[i] += 1 
                pos = posns[i]
                evt = track[pos]
            except IndexError:
                break

    while not instruments[0]["used"] and not instruments[0]["program"] and not instruments[0]["is_drum"]:
        instruments = instruments[1:]
    #print(instruments)
    return instruments

def multitrack_save(src,path):
    src.write(path)
    return path

def multitrack_load(path,res):
    mul = mullib.Multitrack(beat_resolution=res,name=os.path.basename(path))
    ins = get_instruments(path)
    pretty = pretty_load(path,res)
    mul.parse_pretty_midi(pretty,skip_empty_tracks=False)       
    if len(mul.tracks)!=len(pretty.instruments):
        raise MidiError

    j=0
    tracks = []
    for i in ins:
        program = i["program"]
        is_drum = i["is_drum"]
        name = i["name"]
        if name is None:
            if is_drum:
               name = "Drums"
            else:
               name = prelib.program_to_instrument_name(program)
        if i["used"] == 0:
            t = mullib.Track(np.zeros(mul.tracks[0].pianoroll.shape,np.int8), program,is_drum,name)
            tracks += [t]
        else:
            mul.tracks[j].name=name
            tracks += [mul.tracks[j]]
            j+=1
            
    mul = mullib.Multitrack(tracks=tracks,tempo=mul.tempo, downbeat=mul.downbeat, beat_resolution=mul.beat_resolution, name=mul.name)

    return mul

multitrack_repr = {"load": multitrack_load, "save": multitrack_save, "instance": mullib.Multitrack()}

# In[5]:


#____MidiArray_________
class MidiArray:
    
    def __init__(self,mt=None,res=24,array=np.zeros((1,128,1)),tracks_map=[{"program":0,"is_drum":False}]):
        """Loads a midi file to an array shaped (timesteps,pitch,tracks))"""
        if mt is not None:
            tensor = []
            for t in mt.tracks:
                tensor += [t.pianoroll]
            t = np.array(tensor) #(tracks,timesteps,pitch)
            t = np.swapaxes(t,0,1) #(timesteps,tracks,pitch)
            t = np.swapaxes(t,1,2) #(timesteps,pitch,tracks)
            self.array = t
            self.tracks_map = get_instruments(mt)
            self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
        elif len(array.shape) == 3:
            self.array = array
            self.tracks_map = tracks_map
            self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
        else:
            self.array = None
            self.tracks_map = None
            self.dims = {'timesteps': 0,
                       'pitches': 0,
                        'tracks': 0}
   
    def __getitem__(self, item):
        return self.array[item]

    def __getslice__(self, i, j):
        # The deprecated __getslice__ is still called when subclassing built-in types
        # for calls of the form List[i:j]
        return self.array[slice(i,j)]
            
    def load(self,path,res=24):
        mt = multitrack_load(path,res)
        tensor = []
        for t in mt.tracks:
            tensor += [t.pianoroll]
        t = np.array(tensor) #(tracks,timesteps,pitch)
        t = np.swapaxes(t,0,1) #(timesteps,tracks,pitch)
        t = np.swapaxes(t,1,2) #(timesteps,pitch,tracks)
        self.array = t
        self.tracks_map = get_instruments(path)
        self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
    
    def save(self,path):
        """Saves mididata in an array shaped (timesteps,pitch,tracks)) into a .mid file"""
        src = self.array
        src = np.swapaxes(src,1,2) #(timesteps,pitch,tracks)
        src = np.swapaxes(src,0,1) #(timesteps,tracks,pitch)
        mul = mullib.Multitrack()
       
        if len(self.tracks_map) < len(self.array):
            l = len(self.tracks_map)
        else:
            l = len(self.array)
 
        for i in range(l):
            program = self.tracks_map[i]["program"]
            is_drum = self.tracks_map[i]["is_drum"]
            pianoroll = src[i]
            name = self.tracks_map[i].get("name")
            if name is None:
              if is_drum:
                name = "Drums"
              else:
                name = prelib.program_to_instrument_name(program)
            
            t = mullib.Track(pianoroll, program,is_drum,name)
            mul.tracks += [t]
        mul.write(path) #adds one track 
    
    def __setattr__(self, name, value):
        object.__setattr__(self,name, value)
        if name == "array" and value is not None:
            self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
        
    
    
def midiarray_save(src,path):
    src.save(path)
    return path

def midiarray_load(path,res):
    midi_data = MidiArray()
    midi_data.load(path,res)
    return midi_data

midiarray_repr = {"load": midiarray_load, "save": midiarray_save, "instance": MidiArray()}


def vmax(a,v):
        v = v-1
        n = (v*(a>v) - a)
        return a + n*(a>v)

def vmin(a,v):
     return a*(a>v)

# In[6]:
class Gmidi(object):
    """Facade for different classes of midi representations
    _repr is a dict which keys are midi representation classes and values are dict with 
        - ["load"](path) function: from path to representation
        - ["save"](data,path) function: from representation to path
    """
    
    #str representation of midifile is the path of a file with midi information
    def _str_save(data,path): 
        try:
            shutil.copy(data, path)
        except:
            pass
        return path

    def _str_load(path,res):
        return path
 
    #_______________________________________________________________________
    def __init__(self,src,res=24,reprs={mullib.Multitrack : multitrack_repr,
                                       patlib.Pattern : pattern_repr,
                                       prelib.PrettyMIDI: pretty_repr,
                                       MidiArray: midiarray_repr}):
        _str_repr = {"load": Gmidi._str_load, "save": Gmidi._str_save}
        self._log=False
        self._reprs=reprs
        self._sreprs={}
        self._sreprs.update(self._reprs)
        self._sreprs.update({str : _str_repr})
        if self.in_sreprs(src): 
            self._state=src
        else:
            typeError(type(src))
        self._res=res
    
    def typeError(type):
        raise TypeError('Unknown Representation:'+type+'is not in'+_rerp)
    
    def in_sreprs(self,data):
        return isinstance(data,tuple(x for x in self._sreprs))
    
    def in_reprs(self,data):
        return isinstance(data,tuple(x for x in self._reprs))
    
    def save(self,path):
        for i in self._reprs:
            if isinstance(self._state,i):
                self._state=self._reprs[i]["save"](self._state,path)
    
    def to(self,out,clean=True):
        if isinstance(out,str):
            temp_file = out
            clean = False
        else:
            temp_file = "gmidi_tmp.mid"
                            
        if type(out) is type and isinstance(self._state,out):
            return self._state
        
        if self._log:
            print("To",out,file=sys.stderr)
        
        for i in self._sreprs:
            if isinstance(self._state,i):
                self._state=self._sreprs[i]["save"](self._state,temp_file)
                break
        if isinstance(out,str):
            out = str
        self._state=self._sreprs[out]["load"](temp_file,self._res)
        
        if clean:
            os.remove(temp_file)
        
        return self._state
    
    def truncate(self,begin, end):
        '''Clip a midifile from a 'begin' tick to the 'end' tick.'''
        self.to(mullib.Multitrack)
        if begin>len(self._state.tempo) or end < 0 or end <= begin:
            raise TypeError("Empty Slice: midi cannot be empty")
            
        for i in self._state.tracks:
            i.pianoroll = i.pianoroll[begin:end]
        self._state.tempo = self._state.tempo[begin:end]
        #self._state.downbeat = np.zeros(self._state.tempo.shape, dtype=bool)
        self._state.downbeat = self._state.downbeat[begin:end]
        self._state.downbeat[0]=True

        s1=self.dims["timesteps"]
        s2=end-begin
        if s1 < s2:
            #print("Padding:",s1,"+",s2-s1)
            self._state.array=np.pad(self._state.array, ((0,s2-s1),(0,0),(0,0)), 'constant', constant_values=0)
        elif s1 > s2:
            #print("Size:",s1,"Supposed:", s2)
            self._state.array=self._state.array[:s2]
        else:
            pass
        
        return self._state 

    def process(path,
                  i_to_t={('all',False): 0, ('default',False): 0},
                  t_to_i=[{'name':"tutti",'program':0,'is_drum':False}],
                  ticks=4*24*4,
                  min_pitch=0,
                  max_pitch=128,
                  min_vel=0,
                  max_vel=128,
                  transpose=(-6,6)):
       """
       Given a path it returns a np.array(12,-1,ticks,pitch,len(t_to_i)) with pre-processed and augmented data
       - unifies orchestration
       - choped in blocks
       - augmented by transposing from transpose[0] semitones to transpose[1] semitones (default from one dim fifth bellow to one perfect fourth above
       - slicing the wanted pitches
       - normalizing the velocities between 0 and 1
       """
       data = []
       for i in range(*transpose):
          data.append(Gmidi(path))
          data[-1].orchestrate(gmidi.pyormidict.translate(i_to_t),t_to_i)
          data[-1].transpose(i)
          tracks_map = data[-1].tracks_map
          data[-1]=data[i+6].chop(ticks)
          for i in range(len(data[-1])):
            data[-1][i]=data[-1][i].array
       array = np.array(data)
       array = vmax(vmin(array,min_vel),max_vel) #ignore all notes with vel under vel_min and ensure the max_vel is the highest value
       array = array[:,:,:,min_pitch:max_pitch] #clip the pitches we don't want
       array = (array)/float(max_vel)
       return array
 
    def to_gif(self,path):
        self.to(mullib.Multitrack)
        p = self._state.get_merged_pianoroll("max")
        plot.save_animation(path,p,1152,fps=2,hop=24) 

    def to_jpg(self,path):
        self.to(mullib.Multitrack)
        plot.plot_multitrack(self._state,path)  

    def transpose(self, semitones):
        self.to(mullib.Multitrack)
        for i in self._state.tracks:
            if not i.is_drum:
                i.transpose(semitones)
        return self._state   
 
    def orchestrate(self,i_to_t={(0,False):0},t_to_i=[{'program':1,"is_drum":False}]):
        """Reorchestrate using a mapping from a tuple i1=(program,is_drum) to a track t1 (i_to_t[i1]=t1) and
        associating a tuple i2=(program,is_drum) to each one of the final tracks t2 (t_to_i[t2]=i2). 
        Not mapped combinations will not be included in the final result."""
        self.dims        
        mul = self.to(mullib.Multitrack)
        new_mul = mullib.copy(mul)
        
        new_mul.tracks=[]
        new_mul.tracks += [mullib.Track(pianoroll=np.ones(mul.tracks[0].pianoroll.shape))]
        #print([i.name for i in new_mul.tracks])
        for t in range(len(t_to_i)):
            program = t_to_i[t]["program"]
            is_drum = t_to_i[t]["is_drum"]
            name = prelib.program_to_instrument_name(program)
            
            if is_drum:
                name = "Drums"
            
            new_mul.tracks += [mullib.Track(pianoroll=np.zeros(mul.tracks[0].pianoroll.shape),
                                            program=program,
                                            is_drum=is_drum,
                                            name=name)]
            for i in mul.tracks:
                if i_to_t.get((i.program,i.is_drum),i_to_t.get((128,i.is_drum),None)) == t:
                    new_mul.tracks += [i]
            new_mul.merge_tracks(list(range(t+1,len(new_mul.tracks))),
                                 program=program,
                                 is_drum=is_drum,
                                 name=name,
                                 remove_merged=True)
        
        new_mul.tracks=new_mul.tracks[1:]
        self._state = new_mul

        return self._state
    
    def chop(self,ticks_per_clip, mode="chuncking", next_function=None):
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
        self.to(mullib.Multitrack)
        mul = self._state #old_pattern
        new_mul = Gmidi(mullib.copy(mul))
        ticks = mul.tracks[0].pianoroll.shape[0]
        while(e <= ticks):
            new_mul.truncate(b,e)
            midis+=[new_mul]    
            b,e = next_function(b,e,mul)
            new_mul = Gmidi(mullib.copy(mul))
            n+=1
        return midis
    
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self,name)
        except AttributeError as e:
            if hasattr(self._state,name):
                return getattr(self._state, name)
            else:
                for r in self._reprs:
                    if hasattr(self._reprs[r]["instance"],name):
                        self.to(r)
                        return getattr(self._state, name)
                raise e
                
    def __setattr__(self, name, value):
        attr = ["_state","_reprs","_sreprs","_res",
                "chop","orchestrate","transpose","truncate",
                "to","save","in_sreprs","in_reprs",
                "_str_load","_str_save","_log"]
        if name in self.__dict__ or name in attr:
            object.__setattr__(self,name, value)
        elif hasattr(self._state,name):
            setattr(self._state,name, value)
        else:
            for r in self._reprs:
                if hasattr(self._reprs[r]["instance"],name):
                    self.to(r)
                    object.__setattr__(self._state, name,value)
                    return
            object.__setattr__(self,name, value)
                
    def __repr__(self):
        for i in self._sreprs:
            if isinstance(self._state,i):
                return "Gmidi("+str(i)+","+str(self._state)+")"
        raise MidiError
        
    def __getitem__(self, item):
        return self._state[item]

    def __getslice__(self, i, j):
        # The deprecated __getslice__ is still called when subclassing built-in types
        # for calls of the form List[i:j]
        return self._state[slice(i,j)]
   
