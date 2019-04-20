from distutils.core import setup

REQUIRED_PACKAGES = [
    'tensorflow==1.12.0',
    'matplotlib==3.0.2',
    'numpy==1.16.1',
]

setup(
    name='pizza_agent',
    version='0.1',
    description='Policy gradient for learning to cut Pizza.',
)