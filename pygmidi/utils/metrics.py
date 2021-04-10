##LESSSE
##10 November 2018
##gmidi
##____________
##List of music related metrics
##____________

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from . import evaltools as evaltool
from itertools import product

##_____AUX_______
def to_chroma(pianoroll):
    """Return the chroma features (timesteps, 12, tracks). This chroma is not normalized, means that for different pianorolls 0 may not correpond to the same note)."""
    padded = np.pad(pianoroll, ((0,0), (0, 0), (0, 12 - pianoroll.shape[-2] % 12),(0,0)),
                    'constant')
    return np.sum(np.reshape(padded, (pianoroll.shape[0],pianoroll.shape[1], 12, -1,pianoroll.shape[-1])), 3)


def get_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    """Compute and return a tonal matrix for computing the tonal distance [1].
    Default argument values are set as suggested by the paper.

    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting harmonic
    change in musical audio. In Proc. ACM MM Workshop on Audio and Music
    Computing Multimedia, 2006.
    """
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
    return tonal_matrix

tonal_matrix = get_tonal_matrix()

def tonal_dist(chroma1, chroma2, tonal_matrix=tonal_matrix):
    """Return the tonal distance between two chroma features."""
    chroma1 = chroma1 / np.where(np.sum(chroma1)>0,np.sum(chroma1),1)
    result1 = np.matmul(tonal_matrix, chroma1)
    chroma2 = chroma2 / np.where(np.sum(chroma2)>0,np.sum(chroma2),1)
    result2 = np.matmul(tonal_matrix, chroma2)
    return np.linalg.norm(result1 - result2)

##____________Metrics_________________
"""Parameters:  
	-pianoroll: a 4D tensor (blocks, timesteps, pitch, tracks) 
   Return:
	- a 1D tensor (blocks)
"""

def empty(pianoroll):
    """Return 1 for""" 
    return ~np.any(pianoroll,(-1,-2,-3))

def empty_timesteps(pianoroll):
    """Return rate of empty timesteps"""
    return np.sum(np.sum(pianoroll,(-1,-2))==0,-1)/pianoroll.shape[-3]

def num_pitch_used(pianoroll):
    """Return the number of unique pitches used in a piano-roll. The same pitch in two different instruments is considered the same pitch"""
    return np.sum(np.sum(pianoroll,(-1,-3)) >0,-1)

def num_pitch_classes_used(pianoroll):
    """Return the number pitches classes used in a piano-roll. 12 is the higher value"""
    return num_pitch_used(to_chroma(pianoroll))

def num_instrument_used(pianoroll):
    """Return the number of intruments used in a piano-roll"""
    return np.sum(np.sum(pianoroll,(-3,-2)) >0,-1)

def dominant_instrument(pianoroll):
    """Return the number of the track with most volume"""
    p = np.sum(pianoroll, (-2,-3))
    return np.argmax(p,-1)

def average_volume(pianoroll):
    """Return the average volume level when there is at least someone playing"""
    p = np.sum(pianoroll, (-1,-2))
    n = np.sum(p!=0,-1)
    p = np.sum(p,-1)
    return np.where(n>0, p/n, np.nan)
    
def pitch_min(pianoroll):
    """Return the lowest pitch used"""
    def first_nonzero(arr, axis, invalid_val=np.nan):
         mask = arr!=0
         return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return first_nonzero(np.sum(pianoroll, (-1,-3)),1)

def pitch_max(pianoroll):
    """Return the highest pitch used"""
    def last_nonzero(arr, axis, invalid_val=np.nan):
         mask = arr!=0
         val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
         return np.where(mask.any(axis=axis), val, invalid_val)
    return last_nonzero(np.sum(pianoroll, (-1,-3)),1)

def pitch_extension(pianoroll):
    """Return the pitch extension""" 
    return pitch_max(pianoroll) - pitch_min(pianoroll)

def polyphonic_threshold_factory(threshold=2):
    """Return the number of timesteps where more than threshold notes are being played to the number of not empty timesteps"""
    def polyphonic_threshold(pianoroll):
      return np.sum(np.sum(np.sum(pianoroll,-1)!=0,-1)>threshold,-1)/(pianoroll.shape[-3]*(1-empty_timesteps(pianoroll)))
    return polyphonic_threshold

def scale_ratio_factory(scale="diato"):
    scales = {}
    scales["diato"] = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1] 
    scales["harmo"] = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    scales["melod"] = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    scales["penta"] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    scales["whole"] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    scales["chrom"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    def scale_ratio(pianoroll):
       s = np.array(scales[scale])
       chroma = to_chroma(pianoroll)
       l = [s]
       d = np.sum(chroma,(-1,-2,-3))
       n = np.sum(np.multiply(np.sum(chroma, axis=(-3,-1)), s, dtype=float),-1)
       result = [np.where(d>0,n/d,np.nan)] 
       while not np.array_equal(np.roll(s,1),l[0]):
         s = np.roll(s,1)
         l += [s]
         d = np.sum(chroma,(-1,-2,-3))
         n = np.sum(np.multiply(np.sum(chroma, axis=(-3,-1)), s, dtype=float),-1)
         result = [np.where(d>0,n/d,np.nan)]
       result = np.array(result)
       return result[np.argmax(np.sum(result,-1))]
    return scale_ratio   

def pitch_threshold_factory(threshold=45):
    """Return the ration of the number of the notes higher than threshold to the total number of notes in a piano-roll."""
    def pitch_threshold(pianoroll):
      v = np.sum(pianoroll, (-1,-3))
      return np.where(v.any(axis=-1), np.count_nonzero(v[...,threshold:],-1)/np.count_nonzero(v,-1), np.nan)
    return pitch_threshold

def notes_on_count(pianoroll):
    """Return the number of notes on captured in pianoroll"""
    padded = np.pad((pianoroll>0).astype(int), ((0,0),(1, 1), (0, 0),(0,0)), 'constant')
    diff = np.diff(padded, axis=-3)
    result = []
    for i in diff:
       flattened = i.T.reshape(-1,)
       onsets = (flattened > 0).nonzero()[0]
       result+=[len(onsets)]
    return np.array(result)

def reused_notes_ratio(pianoroll):
    """Returns the inverse of number of different pitchs to the number of notes played"""
    return notes_on_count(pianoroll)/num_pitch_used(pianoroll)

def qualified_rhythm_factory(threshold=2):
  """Return the ratio of the number of the qualified notes (notes longer than `threshold` (in time step)) to the total number of notes in a piano-roll."""
  def rhythm_threshold(pianoroll):
    padded = np.pad((pianoroll>0).astype(int), ((0,0),(1, 1), (0, 0),(0,0)), 'constant')
    diff = np.diff(padded, axis=-3)
    result = []
    for i in diff:
       flattened = i.T.reshape(-1,)
       onsets = (flattened > 0).nonzero()[0]
       offsets = (flattened < 0).nonzero()[0]
       num_qualified_note = (offsets - onsets >= threshold).sum()
       if len(onsets) > 0:
           result+=[num_qualified_note / len(onsets)]
       else:
           result+=[-1]
    return np.array(result)
  return rhythm_threshold

def harmonicity_factory(resolution=24, tonal_matrix=tonal_matrix):
  """Return the harmonicity metric value"""
  def harmonicity(pianoroll): 
    score_list = []
    chroma = to_chroma(pianoroll)
    pairs = list(filter(lambda x: x[0] != x[1],product(range(chroma.shape[-1]), range(chroma.shape[-1])))) 
    for b in range(chroma.shape[0]):
     score_list_pairs = []
     for p in pairs:
      for r in range(chroma.shape[-3]//resolution):
        start = r * resolution
        end = (r + 1) * resolution
        beat_chroma1 = np.sum(chroma[b,...,start:end,:,p[0]:p[0]+1], -3)
        beat_chroma2 = np.sum(chroma[b,...,start:end,:,p[1]:p[1]+1], -3)
        if np.any(beat_chroma1) and np.any(beat_chroma2):
          score_list_pairs.append(tonal_dist(beat_chroma1, beat_chroma2, tonal_matrix))
     score_list.append(np.mean(np.array(score_list_pairs)))
    return np.array(score_list)
  return harmonicity

class metric:
    def __init__(self,fun,tracks_map,name):
        self.c = evaltool.characterizors(fun,name+" Char")
        self.r = evaltool.regression(fun,name+" Regr")
        self.u = evaltool.uniformization(fun,name+" Unif")
        self.fun = fun
        self.t_map = tracks_map
        self.name = name
        
    def add(self,pianoroll):
        m = np.nonzero(self.t_map)[0]
        if not isinstance(pianoroll,list):
            pianoroll = [pianoroll]
        for i in range(len(pianoroll)):
            pianoroll[i] = pianoroll[i][:,:,:,m]
        self.c.add(pianoroll)
        self.r.add(pianoroll)
        self.u.add(pianoroll)
        
    def present(self,path=None):
        if path is None:
            path = path = self.name.lower().replace(" ","_")
        if not os.path.exists(path):
            os.makedirs(path)
        self.c.present(os.path.join(path,"characterizors.png"))
        self.r.present(os.path.join(path,"regression.png"))
        self.u.present(os.path.join(path,"uniformization.png"))
    
    def normality(self,table=True):
        cdf = self.c.normality(True)
        rdf = self.r.normality(True)
        udf = self.u.normality(True)
        a = pd.concat([cdf,rdf,udf],1)
        if table:
            return a
        return a.values
    
    def similarity(self,pianoroll, table=False):
        cdf = self.c.similarity(pianoroll,True)
        rdf = self.r.similarity(pianoroll,True)
        udf = self.u.similarity(pianoroll,True)
        a = pd.concat([cdf,rdf,udf],1)
        if table:
            return a
        return a.values
    
    def to_dir(self,path=None):
        if path is None:
            path = path = self.name.lower().replace(" ","_")
        if not os.path.exists(path):
            os.makedirs(path)
        self.c.to_csv(os.path.join(path,"characterizors.csv"))
        self.r.to_csv(os.path.join(path,"regression.csv"))
        self.u.to_csv(os.path.join(path,"uniformization.csv"))
        self.present(path)
    
    def from_dir(self,path):
        self.c.from_csv(os.path.join(path,"characterizors.csv"))
        self.r.from_csv(os.path.join(path,"regression.csv"))
        self.u.from_csv(os.path.join(path,"uniformization.csv"))
