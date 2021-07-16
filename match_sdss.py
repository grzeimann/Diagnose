#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:58:06 2021

@author: gregz
"""

import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

t = fits.open("/work/07918/mqd5709/stampede2/sdss_spectra/specObj-dr10.fits")
ra_sdss = t[1].data['PLUG_RA']
dec_sdss = t[1].data["PLUG_DEC"]

f = fits.open("/work/03730/gregz/maverick/parallel/HETVIPS_continuum_v1.0.fits")
info = f[1].data
ra_virus = info['RA']
dec_virus = info['Dec']
coords_sdss = SkyCoord(ra=ra_sdss*u.deg,dec=dec_sdss*u.deg)
coords_virus = SkyCoord(ra=ra_virus*u.deg,dec=dec_virus*u.deg)

print('Start matching')
idx_sdss,d2d,d3d = coords_virus.match_to_catalog_sky(coords_sdss)
print('Done matching')
matches = np.where(d2d.arcsec<3.)[0]
    
np.savetxt('sdss_matches.txt', np.array(matches, dtype=int), fmt='%d')
np.savetxt('sdss_matches_indices.txt', np.array(idx_sdss[matches], dtype=int), fmt='%d')