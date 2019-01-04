
# Gmidi
### A general representation for midi for symbolic music data-driven approaches

General Midi or gmidi is a facade for the most used representations of midiin python, so you can use it transparently. It includes:
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
$ git clone https://github.com/LESSSE/gmidi
$ cd gmidi
$ pip install . --user
```

## Typical Use
____________________________________________________________

```
    #!/usr/bin/env python

    from gmidi import gmidi
    g = gmidi("path_to_midifile.mid")
    
    g.transpose(-1)
    g.to_npz("new_npz_path")
    g.to_gif("new_gif_path")
```
