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
coords_sdss = SkyCoord(ra=ra_sdss*u.arcsecond,dec=dec_sdss*u.arcsecond)
coords_virus = SkyCoord(ra=ra_virus*u.arcsecond,dec=dec_virus*u.arcsecond)
print('Matching the catalogs')
idx_sdss,d2d,d3d = coords_virus.match_to_catalog_sky(coords_sdss)
print('Done matching')
delta_ra, delta_dec, matches = ([], [], [])
for i in np.arange(len(idx_sdss)):
    if d2d.arcsec > 3.:
        continue
    matches.append(idx_sdss[i])
    delta_ra.append(3600. * np.cos(np.deg2rad(dec_virus[i])) * (ra_virus[i] - ra_sdss[idx_sdss[i]]))
    delta_dec.append(3600. * (dec_virus[i] - dec_sdss[idx_sdss[i]]))
    
matches = np.hastack(matches)
np.savetxt('sdss_matches.txt', matches)