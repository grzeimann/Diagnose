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
from astropy.table import Table

path = '/work/03730/gregz/maverick/classifications'
filenames = sorted(glob.glob(op.join(path, 'classification*.fits')))

stellar_types = ['B', 'A', 'W']

speclist = []
R, D, SN, GM, MJD, S = ([], [], [], [], [], [])
for filename in filenames:
    f = fits.open(filename)
    logic = np.zeros((len(f[1].data),), dtype=bool)
    for st_type in stellar_types:
        logic *= (f[5].data['starnames'] == st_type)
    is_st = np.where((f[1].data == 1) * logic)[0]
    if len(is_st) > 0:
        speclist.append(f[0].data[is_st])
        R.append(f[-1].data['RA'][is_st])
        D.append(f[-1].data['Dec'][is_st])
        SN.append(f[-1].data['sn'][is_st])
        GM.append(f[-1].data['gmag'][is_st])
        MJD.append(f[-1].data['mjd'][is_st])
        S.append(f[-1].data['shotid'][is_st])
speclist = np.vstack(speclist)
R, D, SN, GM, MJD, S = [np.vstack(i) for i in [R, D, SN, GM, MJD, S]]
T = Table([R, D, SN, GM, MJD, S], names=['RA', 'Dec', 'sn', 'gmag', 'mjd', 
                                         'shotid'])
fits.HDUList([fits.PrimaryHDU(speclist), fits.BinTableHDU(T)]).writeto(
             'out.fits', overwrite=True)
