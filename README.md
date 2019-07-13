
# pygmidi
### A general representation for midi for symbolic music data-driven approaches

Python General Midi or pygmidi is a facade for the most used representations of midi in python, so you can use it transparently. It includes:
* [midi Pattern](https://github.com/vishnubob/python-midi)
* [PrettyMidi](https://github.com/craffel/pretty-midi)
* [pianoroll Multitrack](https://salu133445.github.io/pypianoroll/)
* [MidiFile](https://mido.readthedocs.io/en/latest/midi_files.html)
* MidiArray: a new minimalistic representation that uses only an array and a list of instruments, allowing to easily have a tensor representation of the midi file aimming machine learning purposes
 
**New Features:**
* Methods to automatically orchestrate and slice midi files to help pre-processing midi data
* `sparray` is a sparse array representation for efficient disk saving purposes with which you may save and load tensor representated midi files for both time and space efficiency 
* Set of evaluation tools for variable sized sequences of pianoroll blocks and metrics for music evaluation 
* A lot of different ways to export to jpg, gif and wav formats (**comming exporting to PDF**)

## Instalation
____________________________________________________________

```
$ git clone https://github.com/LESSSE/pygmidi
$ cd gmidi
$ pip install . --user
```

## Typical Use
____________________________________________________________

```
    #!/usr/bin/env python

    from pygmidi import pygmidi
    g = pygmidi("path_to_midifile.mid")
    
    g.transpose(-1)
    g.to_npz("new_npz_path")
    g.to_gif("new_gif_path")
```


## For processing midi files in a directory
____________________________________________________________

```
    #!/usr/bin/env python

    from pygmidi import pygmidi,utils,midiarray
    from glob import glob
    from os import path
    
    programs_map = {('woods',False): 0,
                ('brass',False): 1,
                ('percussion',True): 2,
                ('timpani',False): 3,
                ('chromatic_percussion',False): 4,
                ('voices', False): 5,
                ('guitars',False): 6,
                ('basses',False): 6,
                ('strings',False): 6,
                ('keyboards',False): 7}

    #Tracks_map is the configuration for the new orchestrated tracks
    tracks_map = [{'program':71,'is_drum':False,"name":"woods"}, #woods
                {'program':60,'is_drum':False,"name":"brass"}, #brass
                {'program':0,'is_drum':True,"name":"percussion"},  #percussion
                {'program':47,'is_drum':False,"name":"timpani"}, #timpani
                {'program':14,'is_drum':False,"name":"tubular bells"}, #tubular bells
                {'program':52,'is_drum':False,"name":"voices"}, #voices
                {'program':48,'is_drum':False,"name":"strings"}, #strings
                {'program':1,'is_drum':False,"name":"piano"}] #piano
    
    for i in glob('{}/*.mid'.format("midi")):
         a = pygmidi.process(songs[0],
                  i_to_t=programs_map,
                  t_to_i=tracks_map,
                  ticks=4*24*4,
                  transpose=(0,1))
         utils.sparray.save("{}/{}.npz".format("npz",path.splitext(path.basename(i))[0]),a)
```


