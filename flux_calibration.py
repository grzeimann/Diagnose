#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:33:23 2021

@author: gregz
"""

import glob
import numpy as np
import os.path as op

from astropy.io import fits

path = '/work/03730/gregz/maverick/classifications'
filenames = sorted(glob.glob(op.join(path, 'classifications*.fits')))
print(filenames)

speclist = []
for filename in filenames:
    f = fits.open(filename)
    is_wd = np.where((f[1].data == 1) * (f[5].data['starnames'] == 'W'))[0]
    if len(is_wd) > 0:
        speclist.append(f[0].data[is_wd])
speclist = np.vstack(speclist)
fits.PrimaryHDU(speclist).writeto('out.fits', overwrite=True)