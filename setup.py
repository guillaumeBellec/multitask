from setuptools import setup, find_packages

setup(
    name='Multitask Splitter in Pytorch',
    version='0.0.0',
    url='https://github.com/guillaumeBellec/multitask',
    author='Guillaume Bellec',
    author_email='guillaume@bellec.eu',
    description='This is a minimal solution for multitask deep learning in PyTorch.',
    packages=find_packages(),
    install_requires=['numpy >= 1.24.2', 'torch >= 1.13.1', 'scipy >= 1.10.0'],
)