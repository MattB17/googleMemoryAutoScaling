from setuptools import setup, find_packages
from os import path
from io import open

# Get current directory and long description from README
current_dir = path.abspath(path.dirname(__file__))

# the setup
setup(
    name="MemoryAutoScaling",
    version="0.1.0",
    description="Prediction Methods for Memory Auto Scaling",
    author="Matt Buckley",
    packages=find_packages(exclude=['contrib', 'docs', 'tests'])
)
