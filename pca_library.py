#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:44:36 2021

@author: gregz
"""

import numpy as np
import os.path as op
import sys

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

def get_script_path():
    '''
    Get script path, aka, where does Diagnose live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

def get_pca_miles(n_components=50, 
                  lp='/Users/gregz/Downloads/MILES_library_v9.1_FITS'):
# =============================================================================
#     For Local Use Only to construct templates
# =============================================================================
    basedir = lp
    file1 = open(op.join(basedir, 'paramsMILES_v9.1.txt'), 'r')
    Lines = file1.readlines()
    info = []
    for line in Lines:
        l = line.strip()
        n = l[23:27]
        t, g, m = l[51:72].split()
        if m == '--':
            m = 0.0
        if t == '--':
            continue
       
        t = float(t)
        g = float(g)
        m = float(m)
        info.append([n, t, g, m])
    
    # define wavelength
    G = Gaussian1DKernel(2.3)
    wave = np.linspace(3470, 5540, 1036)
    models = np.zeros((len(info), len(wave)))
    i_array = np.zeros((len(info), 3))
    for i in np.arange(len(info)):
        g = fits.open(op.join(basedir, 's%s.fits' % info[i][0]))
        waveo = (g[0].header['CRVAL1'] + np.linspace(0, len(g[0].data[0])-1,
                                                 len(g[0].data[0]))
                 * g[0].header['CDELT1'])
        I = interp1d(waveo, convolve(g[0].data[0], G, preserve_nan=True), kind='quadratic', bounds_error=False,
                     fill_value=np.nan)
        models[i] = I(wave)
        i_array[i] = info[i][1:]
    
    
    k = []
    inds = np.argsort(i_array[:, 0])
    sel = wave > 3550.
    a12 = np.zeros((len(models), len(models)))
    b12 = np.zeros((len(models), len(models)))
    for i in np.arange(len(models)):
        a12[i] = np.sum(models[i][np.newaxis, sel] * models[:, sel], axis=1) / np.sum(models[:, sel]**2, axis=1)
        b12[i] = np.sqrt(np.sum((models[i][np.newaxis, sel] - a12[i][:, np.newaxis]*models[:, sel])**2, axis=1)
                         / np.sum(models[:, sel]**2, axis=1))
    for i in inds:
        if len(k) == 0:
            k.append(i)
            continue
        if np.any(b12[i][k] < 0.05):
            continue
        k.append(i)
    print(len(k))
    new_models = models * 1.    
    new_models[np.isnan(new_models)] = 0.0
    M = new_models[:, sel].mean(axis=1)
    S = new_models[:, sel].std(axis=1)
    z = (new_models - M[:, np.newaxis]) / S[:, np.newaxis]
    pca = PCA(n_components=n_components).fit(z[k][:, sel])
    H = pca.components_
    Hp = np.zeros((H.shape[0], len(wave)))
    Hp[:, sel] = H
    H = Hp * 1.
    T = np.zeros(new_models.shape)
    SOLS = np.zeros((len(new_models), n_components))
    for i in np.arange(T.shape[0]):
        sol = np.linalg.lstsq(H.T[sel], z[i][sel])[0]
        SOLS[i] = sol
        T[i] = np.dot(H.T, sol) * S[i] + M[i]
    T[:, ~sel] = 0.
    return H, models, i_array, T, SOLS, pca

def get_pca_stars(template_name, vbins=np.linspace(-500, 500, 50),
                  wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    lp = op.join(dirname, 'stellar-templates', template_name)
    f = fits.open(lp)
    models = f[0].data
    orig_wave = 10**(f[0].header['CRVAL1']+np.arange(models.shape[1])*
                     f[0].header['CDELT1'])
    M = np.zeros((len(vbins), models.shape[0], len(wave)))
    for j in np.arange(models.shape[0]):
        I = interp1d(orig_wave, models[j], kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        for i, v in enumerate(vbins):
            M[i, j] = I(wave / (1. + (v / 2.99792e5)))
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M
    

def get_pca_qso(zbins=np.linspace(0.01, 0.5, 50), 
                wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    lp = op.join(dirname, 'redrock-templates', 'rrtemplate-qso.fits')
    f = fits.open(lp)
    models = f[0].data
    orig_wave = 10**(f[0].header['CRVAL1']+np.arange(models.shape[1])*
                     f[0].header['CDELT1'])
    M = np.zeros((len(zbins), models.shape[0], len(wave)))
    for j in np.arange(models.shape[0]):
        I = interp1d(orig_wave, models[j], kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        for i, z in enumerate(zbins):
            M[i, j] = I(wave/(1.+z))
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M

def get_pca_galaxy(zbins=np.linspace(0.01, 0.5, 50), 
                   wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    lp = op.join(dirname, 'redrock-templates', 'rrtemplate-galaxy.fits')
    f = fits.open(lp)
    models = f[0].data
    orig_wave = (f[0].header['CRVAL1']+np.arange(models.shape[1])*
                 f[0].header['CDELT1'])
    M = np.zeros((len(zbins), models.shape[0], len(wave)))
    for j in np.arange(models.shape[0]):
        I = interp1d(orig_wave, models[j], kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        for i, z in enumerate(zbins):
            M[i, j] = I(wave/(1.+z))
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M