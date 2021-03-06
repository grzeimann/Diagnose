#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:44:43 2020

@author: gregz
"""

import config
import numpy as np
import os.path as op
import sys
import warnings

from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Polynomial1D
from astropy.table import Table
from pca_library import get_pca_stars, get_pca_galaxy, get_pca_qso
from input_utils import parse_args, setup_logging

def get_script_path():
    '''
    Get script path, aka, where does Diagnose live?
    '''
    return op.dirname(op.realpath(__file__))

def get_min(x, y, ind):
    ind0 = ind - 1
    ind1 = ind 
    ind2 = ind + 1
    dx = x[1] - x[0]
    m = (x[ind1] - dx * (y[ind2] - y[ind0]) /
         (2. * (y[ind2] - 2. * y[ind1] + y[ind0])))
    p = np.polyfit(x[ind-1:ind+2], y[ind-1:ind+2], 2)
    my = np.polyval(p, m)
    return m, my

def get_reduced_chi2(H, spectrum, spectrum_error, m, sel1, zbins, 
                     n_components=10, outlier_thresh=5.):
    chi2 = np.zeros((len(zbins),))
    sel = sel1 * True
    for j in np.arange(len(zbins)):
        solgal = np.linalg.lstsq(H[j].T[sel], spectrum[sel] - m)[0]
        gal_model = np.dot(H[j].T, solgal) + m
        V = (spectrum[sel] - gal_model[sel]) / spectrum_error[sel]
        chi2[j] = (1. / (sel.sum() + 1. - n_components) * np.sum(V**2))
    ind = np.argmin(chi2)
    if (ind == 0) or (ind == (len(chi2)-1)):
        z = zbins[ind]
        mchi = chi2[ind]
    else:
        z, mchi = get_min(zbins, chi2, ind)
    sol = np.linalg.lstsq(H[ind].T[sel], spectrum[sel] - m)[0]
    model = np.dot(H[ind].T, sol) + m
    model[np.isnan(spectrum)] = np.nan
    dof = (sel.sum() + 1. - n_components)
    return mchi, model, z, dof, sol


args = parse_args()
log = setup_logging('diagnose')

# This is a hack to ignore unwanted (and hopefully not important) warnings
warnings.simplefilter("ignore")


# Get flux calibration correction
dirname = get_script_path()
T = Table.read(op.join(dirname, 'calibration', 'normalization.txt'), 
               format='ascii.fixed_width_two_line')
flux_normalization = np.array(T['normalization'])

# Load the continuum catalog from virus observations
f = fits.open(args.catalog)

# =============================================================================
# The wavelength is never defined in the catalog, but is the default VIRUS
# wavelength to which all spectra are rectified
# =============================================================================
wave = np.linspace(3470, 5540, 1036)

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

if spectra.ndim == 1:
    spectra = spectra[np.newaxis, :]
    error = error[np.newaxis, :]
    weight = weight[np.newaxis, :]

# =============================================================================
# The info about the catalog objects is in extension 1 and includes RA, Dec,
# and signal to noise (sn)
# =============================================================================
try:
    info = f[1].data
except:
    log.warning('Table is empty')
    info = fits.BinTableHDU(Table(np.zeros(len(spectra)), names=['dummy'])).data


# Initialize astropy fitter
fitter = LevMarLSQFitter()

if args.index_filename is not None:
    try:
        shortsel = np.loadtxt(args.index_filename)
        shortsel = np.array(shortsel, dtype=int)
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
    shortsel = np.arange(args.low_index, args.high_index, dtype=int)

# Select down to the desired indices
spec = spectra[shortsel] 
weigh = weight[shortsel]
err = error[shortsel] 
if args.normalize:
    spec = spec * flux_normalization[np.newaxis, :]
    err = err * flux_normalization[np.newaxis, :]

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
    wmask = weigh[ind] > 0.3 * np.nanmedian(weigh[ind])
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
if args.quick:
    g = fits.open(op.join(dirname, 'saved', 'models.fits'))
    H_stars = g[1].data
    H_galaxy = g[2].data
    H_qso = g[3].data
else:
    H_stars = get_pca_stars(config.stellar_template_file, vbins=config.vbins)
    H_galaxy = get_pca_galaxy(zbins=config.zbins_gal)
    H_qso = get_pca_qso(zbins=config.zbins_qso)

# Get the mean for each spectrum
M = np.nanmean(spec, axis=1)

name_list = ['A', 'B', 'CV', 'F', 'G', 'K', 'M', 'WD']
# Build empty arrays to save the data
chis = np.zeros((len(shortsel), 3))
thresh = np.zeros((len(shortsel),))
starnames = np.chararray((len(shortsel),))
classification = np.zeros((len(shortsel),))
zs = np.zeros((len(shortsel), 3))
models = np.zeros((len(shortsel), 6, 1036))
pca_array = np.zeros((len(shortsel), len(name_list)+2, 10))

# Checking memory useage

H_stars_list = [H_stars[:, :5], H_stars[:, 5:10], H_stars[:, 10:13], 
                H_stars[:, 13:18], H_stars[:, 18:23], H_stars[:, 23:28],
                H_stars[:, 28:33], H_stars[:, 33:]]
name_list = ['A', 'B', 'CV', 'F', 'G', 'K', 'M', 'WD']
# Begin the fitting
for i in np.arange(len(spec)):
    if (i % 1000) == 999:
        log.info('Fitting the %i spectrum' % (i+1))
    sel = np.isfinite(spec[i]) * (wave>3650) * (wave<5450)
    if sel.sum() < 100:
        continue
    chi_list, m_list, v_list = ([], [], [])
    for k, h_stars in enumerate(H_stars_list):
        chi2_stars, model_stars, vbest, dof_stars, pca_comp_stars = get_reduced_chi2(h_stars, 
                                                                     spec[i], 
                                                                     err[i], M[i],
                                                                     sel, 
                                                                     config.vbins)
        chi_list.append(chi2_stars)
        m_list.append(model_stars)
        v_list.append(vbest)
        pca_array[i, k, :len(pca_comp_stars)] = pca_comp_stars
    ind = np.argmin(chi_list)
    chi2_stars = chi_list[ind]
    model_stars = m_list[ind]
    vbest = v_list[ind]
    starnames[i] = name_list[ind]
    chi2_galaxy, model_galaxy, zbest_galaxy, dof_galaxy, pca_comp_gal = get_reduced_chi2(H_galaxy, 
                                                                 spec[i], 
                                                                 err[i], M[i],
                                                                 sel, 
                                                                 config.zbins_gal)
    pca_array[i, k+1, :len(pca_comp_gal)] = pca_comp_gal
    chi2_qso, model_qso, zbest_qso, dof_qso, pca_comp_qso = get_reduced_chi2(H_qso, spec[i], 
                                                                 err[i], M[i],
                                                                 sel, 
                                                                 config.zbins_qso)
    pca_array[i, k+2, :len(pca_comp_qso)] = pca_comp_qso
    thresh[i] = np.sqrt(2. * dof_stars) / dof_stars
    chis[i] = np.array([chi2_stars, chi2_galaxy, 
                        chi2_qso])
    zs[i] = np.array([vbest, zbest_galaxy, zbest_qso])
    models[i, 0] = model_stars
    models[i, 1] = model_galaxy
    models[i, 2] = model_qso
    models[i, 3] = spec[i]
    models[i, 4] = err[i]
    models[i, 5] = weigh[i]
    norm = 1. # np.min(chis[i])
    s = np.sort(chis[i] / norm)
    if s[0] < (s[1] - thresh[i]):
        classification[i] = np.argmin(chis[i]) + 1
    else:
        classification[i] = 4

fitslist = [fits.PrimaryHDU(models), fits.ImageHDU(classification), 
            fits.ImageHDU(chis), fits.ImageHDU(thresh), fits.ImageHDU(zs),
            fits.BinTableHDU(Table([starnames], names=['starnames'])), 
            fits.BinTableHDU(Table(info)[shortsel]),
            fits.ImageHDU(pca_array)]
for f, n in zip(fitslist, ['models', 'class', 'chi2', 'thresh',
                           'zs', 'stellartype', 'info', 'pca']):
    f.header['EXTNAME'] = n
fits.HDUList(fitslist).writeto('classification_%03d.fits' % args.suffix, 
            overwrite=True)