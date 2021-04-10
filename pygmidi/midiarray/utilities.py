##LESSSE
##10 November 2018
##gmidi
##____________
##Utils for midiarray
##____________

import pypianoroll as mullib
import pretty_midi as prelib
import os

def get_instruments(mt):
    if isinstance(mt,str):
        path = mt
        res = 480
        pretty = prelib.PrettyMIDI(path,resolution=res)
        mul = mullib.from_pretty_midi(pretty, resolution=res)
        mt = mul   

    return [{"name" : i.name, "is_drum": i.is_drum, "program" : i.program, "used" : i.pianoroll.any()} for i in mt.tracks]
