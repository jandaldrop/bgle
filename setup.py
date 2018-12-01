from distutils.core import setup

import numpy as np


setup(
    name="bgle",
    version="0.1",
    description="Simulation code for the generalized Langevin equation",
    author="Jan Daldrop",
    include_dirs=[np.get_include()],
    install_requires=['numpy>=1.15'],
    packages=["bgle"])
