from setuptools import setup
from os import path


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name='phase_to_rate',
    version='0.0.1',
    description='Repository to reproduce our findings about phase to rate recording in the EC-DG system.',
    long_description=readme,
    author='Daniel Müller-Komorowska',
    author_email='danielmuellermsc@gmail.com',
    url='https://github.com/barisckuru/phase-to-rate.git',
    license=license,
    packages=['phase_to_rate'],
    install_requires=requirements)
