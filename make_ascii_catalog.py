#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:36:32 2021

@author: gregz
"""

import glob
import numpy as np
import os.path as op
import sys

from astropy.io import fits
from astropy.table import Table

filename = sys.argv[1]
t = 1.
folder = sys.argv[2]
N = fits.open(filename)[2].shape[0]
filenames = sorted(glob.glob(op.join(folder, 'classification*.fits')))
T = fits.open(filenames[0])['info'].data
objID, RA, Dec, shotid, gmag, rmag = [[T[name][0]]*N for name in 
                              ['objID', 'RA', 'Dec', 'shotid', 'gmag', 'rmag']] 
imag, zmag, ymag, sn, barycor, mjd, exptime = [[T[name][0]]*N for name in 
                   ['imag', 'zmag', 'ymag', 'sn', 'barycor', 'mjd', 'exptime']]
chi2_star, chi2_galaxy, chi2_qso = ([1.]*N, [1.]*N, [1.]*N)
z_star, z_galaxy, z_qso, z_best = ([0.]*N, [0.]*N, [0.]*N, [0.]*N)
classification, stellartype = (['GALAXY']*N, ['W']*N)
colnames = ['objID', 'RA', 'Dec', 'shotid', 'gmag', 'rmag', 'imag', 'zmag',
            'ymag', 'sn', 'barycor', 'mjd', 'exptime', 'chi2_star',
            'chi2_galaxy', 'chi2_qso', 'z_star', 'z_galaxy', 'z_qso', 'z_best',
            'classification', 'stellartype']

i = 0
for fn in filenames:
    print('Working on %s' % fn)
    f = fits.open(fn)
    l = f[1].shape[0]
    objID[i:i+l] = f['info'].data['objID']
    RA[i:i+l] = f['info'].data['RA']
    Dec[i:i+l] = f['info'].data['Dec']
    shotid[i:i+l] = f['info'].data['shotid']
    gmag[i:i+l] = f['info'].data['gmag']
    rmag[i:i+l] = f['info'].data['rmag']
    imag[i:i+l] = f['info'].data['imag']
    zmag[i:i+l] = f['info'].data['zmag']
    ymag[i:i+l] = f['info'].data['ymag']
    sn[i:i+l] = f['info'].data['sn']
    barycor[i:i+l] = f['info'].data['barycor']
    mjd[i:i+l] = f['info'].data['mjd']
    exptime[i:i+l] = f['info'].data['exptime']
    chi2_star[i:i+l] = f['chi2'].data[:, 0]
    chi2_galaxy[i:i+l] = f['chi2'].data[:, 1]
    chi2_qso[i:i+l] = f['chi2'].data[:, 2]
    f['zs'].data[:, 0] = f['zs'].data[:, 0] / 2.99798e8
    z_star[i:i+l] = f['zs'].data[:, 0]
    z_galaxy[i:i+l] = f['zs'].data[:, 1]
    z_qso[i:i+l] = f['zs'].data[:, 2]
    best_ind = np.array(f['class'].data, dtype=int) - 1
    zbest = np.ones((l,))*-999.
    for j in np.arange(3):
        sel = np.where(best_ind == j)[0]
        zbest[sel] = f['zs'].data[sel, j]
    z_best[i:i+l] = zbest
    class_label = np.array(['GALAXY'] * l)
    for j, label in zip(np.arange(4), ['STAR', 'GALAXY', 'QSO', 'UNKNOWN']):
        sel = np.where(best_ind == j)[0]
        class_label[sel] = label 
    classification[i:i+l] = class_label
    stellartype[i:i+l] = f['stellartype'].data
    i += l

T = Table([objID, RA, Dec, shotid, gmag, rmag, imag, zmag,
            ymag, sn, barycor, mjd, exptime, chi2_star,
            chi2_galaxy, chi2_qso, z_star, z_galaxy, z_qso, z_best,
            classification, stellartype], names=colnames)
T.write('HETVIPS_classifications.txt', format='ascii.fixed_width_two_line', 
        overwrite=True)