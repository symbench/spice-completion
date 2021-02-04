import os
from setuptools import setup

with open(os.path.join('.', 'requirements.txt')) as f:
    install_deps = [ line.strip() for line in f.readlines() ]

setup(name='spice_completion',
      version='0.0.1',
      description='Experimental code for netlist synthesis',
      author='Brian Broll',
      author_email='brian.broll@vanderbilt.edu',
      install_requires=install_deps,
      license='MIT',
      packages=['spice_completion'],
      zip_safe=False)
