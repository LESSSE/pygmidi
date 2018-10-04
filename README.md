
# Gmidi - a general representation for midi for data-driven approaches

General Midi or gmidi is a facade for the most used representations of midiin python, so you can use it transparently. It includes:
* [midi Pattern](https://github.com/vishnubob/python-midi)
* [PrettyMidi](https://github.com/craffel/pretty-midi)
* [pianoroll Multitrack](https://salu133445.github.io/pypianoroll/)
* MidiArray 
* ~~[MIDIFile](https://github.com/MarkCWirt/MIDIUtil)~~ (Coming)
 
**New Features:**
* `MidiArray` is a new representation that uses an array and a list of instruments, allowing to easily have a tensor representation of a midi file
* Methods to automatically orchestrate and slice midi files to help pre processing  of midi data
* `sparray` is a sparse array representation for efficient disk saving purposes with which you may save and load tensor representated midi files for both time and space efficiency 
* Set of evaluation tools for variable sized sequences of pianoroll blocks and metrics for music evaluation  

## Instalation
____________________________________________________________

```python
$ git clone https://github.com/LESSSE/gmidi
$ cd gmidi
$ pip install . --user
```

## Starting
____________________________________________________________



