##LESSSE
##10 November 2018
##gmidi
##____________
##Minimalistic array representation for symbolic music machine learning
##____________

import os.path
from .. import pretty_midi as prelib
from .. import pypianoroll as mullib
from .utilities import get_instruments
import numpy as np

def vmax(a,v):
        n = (v*(a>v) - a)
        return a + n*(a>v)

def vmin(a,v):
    return a*(a>v)

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
            self.array = t/np.float16(127)
            self.res = mt.beat_resolution
            self.tracks_map = get_instruments(mt)
            self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
        elif len(array.shape) == 3:
            if len(tracks_map) != array.shape[-1]:
                raise ValueError("track_map do not have the same number of tracks of the array")
            self.array = vmax(vmin(array,0),1)
            self.res = res
            self.tracks_map = tracks_map
            self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
        else:
            self.array = None
            self.res = res
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
        mul = mullib.Multitrack(beat_resolution=res,name=os.path.basename(path))
        pretty = midi_data = prelib.PrettyMIDI(path,resolution=res)
        mul.parse_pretty_midi(pretty,skip_empty_tracks=False)
        mt = mul
        tensor = []
        for t in mt.tracks:
            tensor += [t.pianoroll]
        t = np.array(tensor) #(tracks,timesteps,pitch)
        t = np.swapaxes(t,0,1) #(timesteps,tracks,pitch)
        t = np.swapaxes(t,1,2) #(timesteps,pitch,tracks)
        self.array = t/np.float16(127)
        self.res = res
        self.tracks_map = get_instruments(path)
        self.dims = {'timesteps': self.array.shape[0],
                       'pitches': self.array.shape[1],
                        'tracks': self.array.shape[2]}
    
    def save(self,path):
        """Saves mididata in an array shaped (timesteps,pitch,tracks)) into a .mid file"""
        src = self.array*np.float16(127) #(timesteps,pitch,tracks)
        src = np.swapaxes(src.astype(np.uint8),1,2) #(timesteps,tracks,pitch)
        src = np.swapaxes(src,0,1) #(tracks,timesteps,pitch)
        mul = mullib.Multitrack(beat_resolution=self.res)
        mul.tracks = mul.tracks[1:]

        if len(self.tracks_map) != len(src):
            raise ValueError("track_map do not have the same number of tracks of the array")
        
        l = len(self.tracks_map)

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
        
    

