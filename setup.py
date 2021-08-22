import logging
from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8'
]

try:
    import mxnet

    mxnet_requires = []
except ModuleNotFoundError:
    mxnet_requires = ["mxnet"]
except Exception as e:
    mxnet_requires = []
    logging.error(e)

try:
    import torch

    ml_pytorch_deps = []
except ModuleNotFoundError:
    import sys

    if 5 <= sys.version_info[1]:
        ml_pytorch_deps = ["torch"]
    else:
        ml_pytorch_deps = []
        logging.warning("Current python version %s is not supported by pytorch", str(sys.version_info[:2]))
except Exception as e:
    ml_pytorch_deps = []
    logging.error(e)

setup(
    name='PyBaize',
    version='0.0.5',
    extras_require={
        'test': test_deps + mxnet_requires + ml_pytorch_deps,
        'mxnet': mxnet_requires,
        'torch': ml_pytorch_deps,
        'full': mxnet_requires + ml_pytorch_deps
    },
    packages=find_packages(),
    install_requires=[
        'longling[ml]>=1.3.32',
        'nni>=1.8'
    ],  # And any other dependencies foo needs
    entry_points={
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
