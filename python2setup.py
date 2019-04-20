from setuptools import find_packages
from setuptools import setup


import subprocess
from distutils.command.build import build as _build

import setuptools

class build(_build):
    """A build command class that will be invoked during package install.
    The package built using the current setup.py will be staged and later
    installed in the worker using `pip install package'. This class will be
    instantiated during install for this specific scenario and will trigger
    running the custom commands specified.
    """
    sub_commands = _build.sub_commands + [('CustomCommands', None)]


_LIBS = 'python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig'.split()

CUSTOM_COMMANDS = [
    ['apt-get', 'update', '-y'],
    ['apt-get', 'install', '-y'] + _LIBS,
]


class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print 'Running command: %s' % command_list
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        print 'Command output: %s' % stdout_data
        if p.returncode != 0:
            raise RuntimeError('Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)

#####

REQUIRED_PACKAGES = [
    'tensorflow==1.12.0',
    'matplotlib==3.0.2',
    'numpy==1.16.1',

]

setup(
    name='pizza_agent',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Policy gradient for learning to cut Pizza.',
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    }
)