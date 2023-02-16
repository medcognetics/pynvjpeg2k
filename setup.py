#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from skbuild import setup


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
if __name__ == "__main__":
    setup(
        name="pynvjpeg",
        version="0.1.0",
        packages=["pynvjpeg"],
    )
