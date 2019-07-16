##LESSSE
##10 November 2018
##gmidi
##____________
##Pianoroll Multi-track module slightly modified from Hao-Wen Dong available in https://github.com/salu133445/pypianoroll
##____________

"""
Pypianoroll
===========
A python package for handling multi-track pianorolls.

Features
--------
- handle pianorolls of multiple tracks with metadata
- utilities for manipulating pianorolls
- save to and load from .npz files using efficient sparse matrix format
- parse from and write to MIDI files
"""
from __future__ import absolute_import, division, print_function
from .version import __version__
from .track import Track
from .multitrack import Multitrack
from .plot import plot_pianoroll, save_animation
from .utilities import (
    check_pianoroll, assign_constant, binarize, clip, copy, load, pad,
    pad_to_multiple, pad_to_same, parse, plot, save, transpose,
    trim_trailing_silence, write)
from pygmidi.pypianoroll import metrics
