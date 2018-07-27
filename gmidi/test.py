import gmidi
from pyormidict import translate

print("Path:")
path = input()

#______________1 - Path to Multitrack_____________
"""a = gmidi.Gmidi(path)
a.tracks
print("1 Path-Multitrack DONE")
"""
#______________2 - Path to MidiArray_____________
"""a = gmidi.Gmidi(path)
a.array
print("2 Path-MidiArray DONE")
"""
#______________3 - Multitrack to MidiArray
"""a = gmidi.Gmidi(path)
a.tracks
a.array
print("3 Multi-Array DONE")
"""
#______________4 - MidiArray to Multitrack
a = gmidi.Gmidi(path)
a.array
print(a.tracks)
print("4 Midi-Multi DONE")

#______________5 - Orchestrate
a = gmidi.Gmidi(path)
i_to_t = {('woods',False): 0,
                                   ('brass',False): 1,
                                   ('percussion',True): 2,
                                   ('timpani',False): 3,
                                   ('chromatic_percussion',False): 4,
                                   ('voices', False): 5,
                                   ('guitars',False): 6,
                                   ('basses',False): 6,
                                   ('strings',False): 6,
                                   ('keyboards',False): 7,
           }
t_to_i = [{'program':71,'is_drum':False}, #woods
               {'program':60,'is_drum':False}, #brass
               {'program':0,'is_drum':True},  #percussion
               {'program':47,'is_drum':False}, #timpani
               {'program':14,'is_drum':False}, #tubular bells
               {'program':52,'is_drum':False}, #voices
               {'program':48,'is_drum':False}, #strings
               {'program':1,'is_drum':False}
 ] 
a.orchestrate(translate(i_to_t),t_to_i)
a.save("made.mid")
print("5 Orchestrate-Midi DONE")

#______________6 - Chop
a = gmidi.Gmidi(path)
ticks = 4*24*4
l = a.chop(ticks)
print("6 Chop DONE")

#______________7 - Orchestrate and Chop
a = gmidi.Gmidi(path)
a.orchestrate(translate(i_to_t),t_to_i)
l = a.chop(ticks)
print("7 Orchestrate-Chop DONE")

#_____________8 - Semi Orchestrate and Chop

