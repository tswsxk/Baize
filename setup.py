from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8'
]

setup(
    name='PyBaize',
    version='0.0.1',
    extras_require={
        'test': test_deps,
    },
    packages=find_packages(),
    install_requires=[
        'longling>=1.3.28',
    ],  # And any other dependencies foo needs
    entry_points={
    },
)
