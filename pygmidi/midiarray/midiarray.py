##LESSSE
##10 November 2018
##gmidi
##____________
##Minimalistic array representation for symbolic music machine learning
##____________

import os.path
import pretty_midi as prelib
import pypianoroll as mullib
from .utilities import get_instruments
import numpy as np


def vmax(a, v):
    n = (v * (a > v) - a)
    return a + n * (a > v)


def vmin(a, v):
    return a * (a > v)


class MidiArray:

    def __init__(self, mt=None, res=24, array=np.zeros((1, 128, 1)), tracks_map=[{"program": 0, "is_drum": False}]):
        """Loads a midi file to an array shaped (timesteps,pitch,tracks))"""
        if mt is not None:
            tensor = []
            for t in mt.tracks:
                tensor += [t.pianoroll]
            t = np.array(tensor)  # (tracks,timesteps,pitch)
            t = np.swapaxes(t, 0, 1)  # (timesteps,tracks,pitch)
            t = np.swapaxes(t, 1, 2)  # (timesteps,pitch,tracks)
            self.array = t / np.float16(127)
            self.res = mt.beat_resolution
            self.tracks_map = get_instruments(mt)
            self.dims = {'timesteps': self.array.shape[0],
                         'pitches': self.array.shape[1],
                         'tracks': self.array.shape[2]}
        else:
            if len(array.shape) != 3:
                raise ValueError("array must be of dimension 3 and not " + str(len(array.shape)))
            if len(tracks_map) != array.shape[-1]:
                raise ValueError("track_map do not have the same number of tracks of the array")
            self.array = vmax(vmin(array, 0), 1)
            self.res = res
            self.tracks_map = tracks_map
            self.dims = {'timesteps': self.array.shape[0],
                         'pitches': self.array.shape[1],
                         'tracks': self.array.shape[2]}

    def __getitem__(self, item):
        return self.array[item]

    def __getslice__(self, i, j):
        # The deprecated __getslice__ is still called when subclassing built-in types
        # for calls of the form List[i:j]
        return self.array[slice(i, j)]

    def load(self, path, res=24):
        pretty = prelib.PrettyMIDI(path, resolution=res)
        mul = mullib.from_pretty_midi(pretty, resolution=res)
        mt = mul
        tensor = []
        for t in mt.tracks:
            tensor += [t.pianoroll]
        t = np.array(tensor)  # (tracks,timesteps,pitch)
        print(mt.tracks)
        t = np.swapaxes(t, 0, 1)  # (timesteps,tracks,pitch)
        t = np.swapaxes(t, 1, 2)  # (timesteps,pitch,tracks)
        self.array = t / np.float16(127)
        self.res = res
        self.tracks_map = get_instruments(path)
        self.dims = {'timesteps': self.array.shape[0],
                     'pitches': self.array.shape[1],
                     'tracks': self.array.shape[2]}

    def save(self, path):
        """Saves mididata in an array shaped (timesteps,pitch,tracks)) into a .mid file"""
        src = self.array * np.float16(127)  # (timesteps,pitch,tracks)
        src = np.swapaxes(src.astype(np.uint8), 1, 2)  # (timesteps,tracks,pitch)
        src = np.swapaxes(src, 0, 1)  # (tracks,timesteps,pitch)
        mul = mullib.Multitrack(resolution=self.res)
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

            t = mullib.StandardTrack(name=name, program=program, is_drum=is_drum, pianoroll=pianoroll)
            mul.tracks += [t]
        mul.write(path)  # adds one track

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "array" and value is not None:
            self.dims = {'timesteps': self.array.shape[0],
                         'pitches': self.array.shape[1],
                         'tracks': self.array.shape[2]}
