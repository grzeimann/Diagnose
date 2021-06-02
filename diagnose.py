#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:44:43 2020

@author: gregz
"""

import config
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import sys
import time
import warnings

from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Polynomial1D
from pca_library import get_pca_stars, get_pca_galaxy, get_pca_qso
from input_utils import parse_args, setup_logging

def get_reduced_chi2(H, spectrum, spectrum_error, m, sel, zbins, 
                     n_components=10):
    chi2 = np.zeros((len(zbins),))
    for j in np.arange(len(zbins)):
        solgal = np.linalg.lstsq(H[j].T[sel], spectrum[sel] - m)[0]
        gal_model = np.dot(H[j].T, solgal) + M[i]
        chi2[j] = (1. / (sel.sum() + 1. - n_components) *
                   np.sum((spectrum[sel] - gal_model[sel])**2 / 
                           spectrum_error[sel]**2))
    ind = np.argmin(chi2)
    sol = np.linalg.lstsq(H[ind].T[sel], spectrum[sel] - m)[0]
    model = np.dot(H[ind].T, sol) + m
    model[np.isnan(spectrum)] = np.nan
    dof = (sel.sum() + 1. - n_components)
    return chi2, model, zbins[ind], dof


args = parse_args()
log = setup_logging('diagnose')
process = psutil.Process(os.getpid())
log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))

# This is a hack to ignore unwanted (and hopefully not important) warnings
warnings.simplefilter("ignore")


# Load the continuum catalog from virus observations
f = fits.open(args.catalog)

# =============================================================================
# The wavelength is never defined in the catalog, but is the default VIRUS
# wavelength to which all spectra are rectified
# =============================================================================
wave = np.linspace(3470, 5540, 1036)

# =============================================================================
# The info about the catalog objects is in extension 1 and includes RA, Dec,
# and signal to noise (sn)
# =============================================================================
info = f[1].data

# =============================================================================
# The spectra are in extension 2
# =============================================================================
spectra = f[2].data

# =============================================================================
# The errors are in extension 3
# =============================================================================
error = f[3].data

# =============================================================================
# The weights are in extension 4.  The weights are the fiber covering fraction
# for a given source.  A weight of 1.0 means that 100% of the light was covered
# For a parallel observation, typically the weights range from 0.2-0.4 because
# the fill factor for the fibers is 1/3.
# =============================================================================
weight = f[4].data

# Initialize astropy fitter
fitter = LevMarLSQFitter()

if args.index_filename is not None:
    try:
        shortsel = np.loadtxt(args.index_filename)
    except:
        log.error('Could not read file: %s' % args.index_filename)
        log.error('Try saving indices with numpy.savetxt()')
        sys.exit()
else:
    if args.low_index is None:
        log.error('Please set "--low_index" and "--high_index" if no'
                      '"--index_filename" is provided')
        sys.exit()
    if args.high_index is None:
        log.error('Please set "--low_index" and "--high_index" if no'
                      '"--index_filename" is provided')
        sys.exit()
    shortsel = np.arange(args.low_index, args.high_index)

# Select down to the desired indices
spec = spectra[shortsel]
weigh = weight[shortsel]
err = error[shortsel]

# Build an empty mask array (boolean)
mask = np.zeros(spec.shape, dtype=bool)

# Loop through the number of spectra
log.info('Total number of spectra: %i' % spec.shape[0])
for ind in np.arange(spec.shape[0]):
    if (ind % 1000) == 999:
        # print so I know how long this is taking
        log.info('[Masking] at index %i' % (ind+1))
    # Initialize this polynomial model for 3rd order
    P = Polynomial1D(3)
    # Select values that are not extreme outliers (watch for nans so use nanmedian)
    wmask = weigh[ind] > 0.3*np.nanmedian(weigh[ind])
    # Fit Polynomial
    fit = fitter(P, wave[wmask], weigh[ind][wmask])
    # Mask values below 80% of the fit
    mask[ind] = weigh[ind] < 0.8 * fit(wave)

# Mask neighbor pixels as well because they tend to have issues but don't meet
# the threshold.  This is a conservative step, but it is done for robustness
# rather than completeness
mask[:, 1:] += mask[:, :-1]
mask[:, :-1] += mask[:, 1:]

# Mask additionally spectra whose average weight is less than 5%
badspectra = np.nanmedian(weigh, axis=1) < 0.05
mask[badspectra] = True
spec[mask] = np.nan

# Get PCA models
H_stars = get_pca_stars(config.stellar_template_file, vbins=config.vbins)
H_galaxy = get_pca_galaxy(zbins=config.zbins_gal)
H_qso = get_pca_qso(zbins=config.zbins_qso)

# Get the mean for each spectrum
M = np.nanmean(spec, axis=1)

# Build empty arrays to save the data
chis = np.zeros((len(shortsel), 3))
thresh = np.zeros((len(shortsel),))
classification = np.zeros((len(shortsel),))
zs = np.zeros((len(shortsel), 3))
models = np.zeros((len(shortsel), 3, 1036))

# Checking memory useage
process = psutil.Process(os.getpid())
log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))

# Begin the fitting
for i in np.arange(len(spec)):
    if (i % 1000) == 999:
        log.info('Fitting the %i spectrum' % (i+1))
    sel = np.isfinite(spec[i]) * (wave>3600)
    chi2_stars, model_stars, vbest, dof_stars = get_reduced_chi2(H_stars, 
                                                                 spec[i], 
                                                                 err[i], M[i],
                                                                 sel, 
                                                                 config.vbins)
    chi2_galaxy, model_galaxy, zbest_galaxy, dof_galaxy = get_reduced_chi2(H_galaxy, 
                                                                 spec[i], 
                                                                 err[i], M[i],
                                                                 sel, 
                                                                 config.zbins_gal)
    chi2_qso, model_qso, zbest_qso, dof_qso = get_reduced_chi2(H_qso, spec[i], 
                                                                 err[i], M[i],
                                                                 sel, 
                                                                 config.zbins_qso)
    thresh[i] = np.sqrt(2. * dof_stars) / dof_stars
    chis[i] = np.array([chi2_stars, chi2_galaxy, chi2_qso])
    zs[i] = np.array([vbest, zbest_galaxy, zbest_qso])
    models[i, 0] = model_stars
    models[i, 1] = model_galaxy
    models[i, 2] = model_qso
    norm = np.min(chis[i])
    s = np.sort(chis[i] / norm)
    if s[0] < (s[1] - thresh[i]):
        classification[i] = np.argmin(chis[i]) + 1
    else:
        classification[i] = 4
    
fitslist = [fits.PrimaryHDU(models), fits.ImageHDU(classification), 
            fits.ImageHDU(chis), fits.ImageHDU(thresh), fits.ImagHDU(zs)]
for f, n in zip(fitslist, ['models', 'class', 'chi2', 'thresh',
                           'zs']):
    f.header['EXTNAME'] = n
fits.HDUList(fitslist).writeto('classification_%03d.fits' % args.suffix)