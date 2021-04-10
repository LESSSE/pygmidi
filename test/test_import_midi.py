import sys, os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "pygmidi"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# _________________________________
def test_pypianoroll():
    import repr
    import pypianoroll

    m = pypianoroll.read(path="Viva_La_Vida.midi")

    m.resolution
    m.tracks[0].pianoroll

    m.trim(0, 12 * m.resolution)
    m.binarize()
    m.plot()


# ______________________________
def test_import_multitrack():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("multitrack")


def test_export_multitrack():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("multitrack")

    new_midi_path = "new_midi_file.midi"
    g.to(new_midi_path)
    assert os.path.exists(new_midi_path)
    os.remove(new_midi_path)


def test_manipulate_multitrack():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("multitrack")

    g.trim(0, 12 * g.resolution)
    g.binarize()
    g.plot()
    g.blend()
    g.count_downbeat()

    g.resolution
    g.tempo
    g.downbeat
    g.tracks


# ________________________________
def test_import_midiarray():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("midiarray")


def test_export_midiarray():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("midiarray")

    new_midi_path = "new_midi_file.midi"
    g.to(new_midi_path)
    assert os.path.exists(new_midi_path)
    os.remove(new_midi_path)


# ________________________________
def test_import_pattern():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("pattern")


def test_export_pattern():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("pattern")

    new_midi_path = "new_midi_file.midi"
    g.to(new_midi_path)
    assert os.path.exists(new_midi_path)
    os.remove(new_midi_path)


# ___________________________________
def test_import_prettymidi():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("prettymidi")


def test_export_prettymidi():
    import pygmidi
    g = pygmidi.Pygmidi("Viva_La_Vida.midi")

    g.to("prettymidi")

    new_midi_path = "new_midi_file.midi"
    g.to(new_midi_path)
    assert os.path.exists(new_midi_path)
    os.remove(new_midi_path)


# __________________________________
def test_1():
    from pygmidi import Pygmidi
    g = Pygmidi("Viva_La_Vida.midi")

    g.transpose(-1)
    g.trim(0, 4 * g.resolution)

    new_midi_path = "new_midi_file.midi"
    g.to(new_midi_path)
    assert os.path.exists(new_midi_path)

    new_wav_path = "new_wav_file.wav"
    g.to_wav(new_wav_path)
    assert os.path.exists(new_wav_path)
    os.remove(new_midi_path)
    os.remove(new_wav_path)


def test_2():
    import pygmidi
    from pygmidi import utils
    from glob import glob
    from os import path

    programs_map = {('woods', False): 0,
                    ('brass', False): 1,
                    ('percussion', True): 2,
                    ('timpani', False): 3,
                    ('chromatic_percussion', False): 4,
                    ('voices', False): 5,
                    ('guitars', False): 6,
                    ('basses', False): 6,
                    ('strings', False): 6,
                    ('keyboards', False): 7}

    # Tracks_map is the configuration for the new orchestrated tracks
    tracks_map = [{'program': 71, 'is_drum': False, "name": "woods"},  # woods
                  {'program': 60, 'is_drum': False, "name": "brass"},  # brass
                  {'program': 0, 'is_drum': True, "name": "percussion"},  # percussion
                  {'program': 47, 'is_drum': False, "name": "timpani"},  # timpani
                  {'program': 14, 'is_drum': False, "name": "tubular bells"},  # tubular bells
                  {'program': 52, 'is_drum': False, "name": "voices"},  # voices
                  {'program': 48, 'is_drum': False, "name": "strings"},  # strings
                  {'program': 1, 'is_drum': False, "name": "piano"}]  # piano

    for f in glob('*.midi'):
        a = pygmidi.Pygmidi.process(f,
                                    i_to_t=programs_map,
                                    t_to_i=tracks_map,
                                    ticks=4 * 24 * 4,
                                    transpose=(0, 1))
        new_npz_path = "{}.npz".format(path.splitext(path.basename(f))[0])
        utils.sparray.save(new_npz_path, a)
        assert os.path.exists(new_npz_path)
        os.remove(new_npz_path)
