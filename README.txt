===========
   Gmidi
===========

A general representation for midi for data-driven approaches. 

Facade for the most known representations of midi, so you can use it transparently. It includes:

* midi Pattern
* PrettyMidi
* Multitrack
* **NEW** - MidiArray representation: array and a list of instruments
* **NEW** - Methods to automatically orchestrate, transpose and slice midi files
* **NEW** - Methods to export midi to a sparse matrix representation of pianoroll and to jpg and gif vizualization

Typical usage:

    #!/usr/bin/env python

    from gmidi import gmidi
    g = gmidi("path_to_midifile.mid")
    
    g.transpose(-1)
    g.to_npz("new_npz_path")
    g.to_gif("new_gif_path")

Contributors: LESSSE
`git <'http://github.com/LESSSE/gmidi>`_.

