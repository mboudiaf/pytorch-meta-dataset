"""
refs:
    - setup tools: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#using-find-or-find-packages
    - https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install
"""
from setuptools import setup
from setuptools import find_packages
import os

# import pathlib

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytorch-meta-dataset',  # project name
    version='0.0.1',
    description="Brando and Patrick's pytorch-meta-datset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/pytorch-meta-dataset',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='MIT',

    # currently
    package_dir={'': 'pytorch_meta_dataset'},
    packages=find_packages('pytorch_meta_dataset'),  # imports all modules/folders with  __init__.py & python files

    # for pytorch see doc string at the top of file
    install_requires=[
        'absl-py==0.11.0',
        'tfrecord==1.11',
        # 'torchvision==0.8.2+cu110',  # original pytorch-meta-dataset
        # 'torchvision',
        'torchvision==0.10.1+cu111',
        'tqdm==4.54.1'
    ]
)

