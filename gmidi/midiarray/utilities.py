##LESSSE
##10 November 2018
##gmidi
##____________
##Utils for midiarray
##____________

from .. import pypianoroll as mullib
from .. import pretty_midi as prelib
import os

def get_instruments(mt):
    if isinstance(mt,str):
        path = mt
        res = 480
        mul = mullib.Multitrack(beat_resolution=res,name=os.path.basename(path))
        pretty = prelib.PrettyMIDI(path,resolution=res)
        mul.parse_pretty_midi(pretty,skip_empty_tracks=False)
        mt = mul   

    return [{"name" : i.name, "is_drum": i.is_drum, "program" : i.program, "used" : i.pianoroll.any()} for i in mt.tracks]
