from setuptools import setup
import os

def readme():
    with open('README.txt') as f:
        return f.read()

VERSION = {}
with open(os.path.join('gmidi', 'version.py')) as f:
    exec(f.read(), VERSION)

setup(name='gmidi',
      version=VERSION['__version__'],
      description='Genereal midi representation for data analysis',
      long_description=open('README.md').read(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: Free for non-commercial use',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Education',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Sound/Audio :: MIDI'
      ],
      keywords=['general','midi','dataset','music','symbolic'],
      url='http://github.com/LESSSE/gmidi',
      author='Luís Espírito Santo (LESSSE)',
      author_email='luis.a.santo@tecnico.ulisboa.pt',
      license='',
      packages=['gmidi','gmidi.midiarray','gmidi.pypianoroll','gmidi.pretty_midi','gmidi.utils'],
      install_requires=[
          'six>=1.0.0,<2.0',
          'numpy>=1.10.0,<2.0',
          'scipy>=1.0.0,<2.0',
          'mido',
          'midi',
          'pandas',
          'pyfluidsynth'
      ],
      extras_require={
        'plot':  ['matplotlib>=1.5'],
        'animation': ['moviepy>=0.2.3.2'],
      },
      include_package_data=True,
      zip_safe=False)
