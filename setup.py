from setuptools import setup
import os

def readme():
    with open('README.txt') as f:
        return f.read()

VERSION = {}
with open(os.path.join('pygmidi', 'versionfile.py')) as f:
    exec(f.read(), VERSION)

setup(name='pygmidi',
      version=VERSION['__version__'],
      description='Python General MIDI representation for symbolic musci data analysis',
      long_description=open('README.md').read(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: Free for non-commercial use',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Education',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Sound/Audio :: MIDI'
      ],
      keywords=['general','midi','dataset','music','symbolic','pygmidi'],
      url='http://github.com/LESSSE/pygmidi',
      author='Luís Espírito Santo (LESSSE)',
      author_email='luis.a.santo@tecnico.ulisboa.pt',
      license='',
      package_dir = {
            'pygmidi': 'pygmidi',
            'pygmidi.midiarray': 'pygmidi/midiarray',
            'pygmidi.pypianoroll': 'pygmidi/pypianoroll',
            'pygmidi.pretty_midi': 'pygmidi/pretty_midi',
            'pygmidi.utils': 'pygmidi/utils'
            },
      packages=['pygmidi','pygmidi.midiarray','pygmidi.pypianoroll','pygmidi.pretty_midi','pygmidi.utils'],
      install_requires=[
          'midi @ ssh+git@github.com:mgedmin/python-midi.git@python3#egg=midi-0.2.3',
          'six>=1.0.0,<2.0',
          'numpy>=1.10.0,<2.0',
          'scipy>=1.0.0,<2.0',
          'mido',
          'pandas',
          'pyfluidsynth'
      ],
      extras_require={
        'plot':  ['matplotlib>=1.5'],
        'gif': ['moviepy>=0.2.3.2'],
      },
      include_package_data=True,
      zip_safe=False)
