##LESSSE
##10 November 2018
##gmidi
##____________
##Main class 
##____________


import sys
import shutil
import os.path
import numpy as np
import matplotlib as mpl
from . import pretty_midi as prelib
from . import pypianoroll as mullib
from .utils import plot
from .utils import pyormidict
from matplotlib import pyplot as plt
from .repr import reprs
from scipy.io import wavfile
import random

def vmax(a,v):
        v = v-1
        v = v/float(128)
        n = (v*(a>v) - a)
        return a + n*(a>v)

def vmin(a,v):
    v = v/float(128)
    return a*(a>v)

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
    def __init__(self,src,res=24,reprs = reprs):
        _str_repr = {"load": Gmidi._str_load, "save": Gmidi._str_save}
        self._log=False
        self._reprs=reprs
        self._sreprs={}
        self._sreprs.update(self._reprs)
        self._sreprs.update({str : _str_repr})
        if self.in_sreprs(src):
            if isinstance(src,str) and not os.path.isfile(src):
                raise ValueError("If src is a path it should point to a file")
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
            s = "gmidi_tmp_lessseesssel_pmt_idimg"
            temp_file = ''.join(random.sample(s,len(s)))
                            
        if type(out) is type and isinstance(self._state,out):
            return self._state
        
        if self._log:
            print("To",out,file=sys.stderr)
        
        try:
            for i in self._sreprs:
                if isinstance(self._state,i):
                    self._state=self._sreprs[i]["save"](self._state,temp_file+".mid")
                    break
            if isinstance(out,str):
                out = str
            self._state=self._sreprs[out]["load"](temp_file+".mid",self._res)
        finally:
            if clean:
                os.remove(temp_file+".mid")
        
        return self._state
    
    def vel_limit(self,v_min=0,v_max=128):
        self.to(arrlib.MidiArray)
        self.array = vmax(vmin(self.array,min_vel),max_vel)

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
          data.append(Gmidi(path,))
          data[-1].orchestrate(pyormidict.translate(i_to_t),t_to_i)
          data[-1].transpose(i)
          tracks_map = data[-1].tracks_map
          data[-1]=data[i-transpose[0]].chop(ticks)
          for i in range(len(data[-1])):
            data[-1][i]=data[-1][i].array
       array = np.array(data)
       array = vmax(vmin(array,min_vel),max_vel)
       array = array[:,:,:,min_pitch:max_pitch] #clip the pitches we don't want
       return array
 
    def to_gif(self,path):
        self.to(mullib.Multitrack)
        p = self._state.get_merged_pianoroll("max")
        plot.save_animation(path,p,1152,fps=2,hop=24) 

    def to_jpg(self,path):
        self.to(mullib.Multitrack)
        plot.plot_multitrack(self._state,path)  

    def to_wav(self,path,fs=16000):
        #exporting to wav
        try:
            a = self.fluidsynth(fs=fs)
        except:
            a = self.synthesize(fs=fs)
        wavfile.write(path, fs, a)

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
   
