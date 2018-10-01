from setuptools import setup

def readme():
    with open('README.txt') as f:
        return f.read()

setup(name='gmidi',
      version='1.0',
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
      keywords='general midi dataset music symbolic',
      url='http://github.com/LESSSE/gmidi',
      author='LESSSE',
      author_email='luis.a.santo@tecnico.ulisboa.pt',
      license='',
      packages=['gmidi'],
      install_requires=[
          'mido',
          'pretty-midi',
          'matplotlib',
          'numpy',
          'pypianoroll',
          'midi',
          'pandas',
          'moviepy'
      ],
      include_package_data=True,
      zip_safe=False)
