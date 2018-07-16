from setuptools import setup

setup(name='gmidi',
      version='0.1',
      description='Genereal midi representation for data analysis',
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
          'midi'
      ],
      zip_safe=False)
