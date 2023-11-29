from setuptools import setup, find_packages

setup(
    name='torchmultitask',
    version='0.0.1',
    url='https://github.com/guillaumeBellec/multitask',
    author='Guillaume Bellec',
    author_email='guillaume.bellec@epfl.ch',
    description='This is a minimal solution for multitask deep learning in PyTorch.',
    packages=find_packages(include=["torchmultitask", "torchmultitask.*"]),
    install_requires=['numpy >= 1.24.2', 'torch >= 1.13.1', 'scipy >= 1.10.0'],
)