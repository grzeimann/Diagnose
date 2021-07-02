#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:44:36 2021

@author: gregz
"""

import glob
import numpy as np
import os.path as op

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

def get_script_path():
    '''
    Get script path, aka, where does Diagnose live?
    '''
    return op.dirname(op.realpath(__file__))

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
    wdspec = get_wd_spec()
    models = np.zeros((len(info), len(wave)))
    i_array = np.ones((len(info), 3))*45000.
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
    new_models = models* 1. 
    new_models[np.isnan(new_models)] = 0.0
    M = new_models[:, sel].mean(axis=1)
    S = new_models[:, sel].std(axis=1)
    z = (new_models - M[:, np.newaxis]) / S[:, np.newaxis]
    pca = PCA(n_components=n_components).fit(z[k][:, sel])
    H = pca.components_
    Hp = np.zeros((H.shape[0], len(wave)))
    Hp[:, sel] = H
    H = Hp * 1.
    HT = np.zeros((H.shape[0]+len(wdspec), H.shape[1]))
    HT[:len(H)] = H
    HT[len(H):] = wdspec
    T = np.zeros(new_models.shape)
    SOLS = np.zeros((len(new_models), n_components))
    for i in np.arange(T.shape[0]):
        sol = np.linalg.lstsq(H.T[sel], z[i][sel])[0]
        SOLS[i] = sol
        T[i] = np.dot(H.T, sol) * S[i] + M[i]
    T[:, ~sel] = 0.
    return HT, models, i_array, T, SOLS, pca, k

def get_pca_stars(template_name, vbins=np.linspace(-500, 500, 50),
                  wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    lp = op.join(dirname, 'stellar-templates', template_name)
    f = fits.open(lp)
    models = f[0].data
    orig_wave = (f[0].header['CRVAL1'] + 
                 np.arange(models.shape[1])*f[0].header['CDELT1'])
    M = np.zeros((len(vbins), models.shape[0], len(wave)))
    for j in np.arange(models.shape[0]):
        I = interp1d(orig_wave, models[j], kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        for i, v in enumerate(vbins):
            M[i, j] = I(wave / (1. + (v / 2.99792e5)))
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M
   
def get_wd_spec(wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    lp = op.join(dirname, 'redrock-templates', 'rrtemplate-star-WD.fits')
    f = fits.open(lp)
    models  = f[0].data
    nwave = (f[0].header['CRVAL1']+np.arange(models.shape[1])*
                     f[0].header['CDELT1'])
    G = Gaussian1DKernel(1.5)
    M = np.zeros((models.shape[0], len(wave)))
    nbins = int(len(nwave) / (2. / (nwave[1] - nwave[0])))
    nwave = np.array([np.mean(chunk) for chunk in np.array_split(nwave, nbins)])
    for j in np.arange(models.shape[0]):
        nmod = np.array([np.mean(chunk) for chunk in np.array_split(models[j], nbins)])
        nmod = convolve(nmod, G, boundary='extend')
        I = interp1d(nwave, nmod, kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        M[j] = I(wave)
    return M    

def get_redrock_stellar_spec(wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    fnames = glob.glob(op.join(dirname, 'redrock-templates', 
                               'rrtemplate-star-*.fits'))
    M = []
    for fname in fnames:
        f = fits.open(fname)
        models  = f[0].data
        print(op.basename(fname), models.shape)
        nwave = (f[0].header['CRVAL1']+np.arange(models.shape[1])*
                     f[0].header['CDELT1'])
        G = Gaussian1DKernel(1.5)
        nbins = int(len(nwave) / (2. / (nwave[1] - nwave[0])))
        nwave = np.array([np.mean(chunk) for chunk in np.array_split(nwave, nbins)])
        for j in np.arange(models.shape[0]):
            nmod = np.array([np.mean(chunk) for chunk in np.array_split(models[j], nbins)])
            nmod = convolve(nmod, G, boundary='extend')
            I = interp1d(nwave, nmod, kind='quadratic', bounds_error=False, 
                         fill_value=0.0)
            M.append(I(wave))
    return np.array(M) 


def get_pca_qso(zbins=np.linspace(0.01, 0.5, 50), 
                wave=np.linspace(3470, 5540, 1036)):
    dirname = get_script_path()
    lp = op.join(dirname, 'redrock-templates', 'rrtemplate-qso.fits')
    f = fits.open(lp)
    models = f[0].data
    orig_wave = 10**(f[0].header['CRVAL1']+np.arange(models.shape[1])*
                     f[0].header['CDELT1'])
    M = np.zeros((len(zbins), models.shape[0], len(wave)))
    G = Gaussian1DKernel(0.5)
    for j in np.arange(models.shape[0]):
        for i, z in enumerate(zbins):
            nwave = orig_wave * (1.+z)
            nbins = int(len(nwave) / (2. / (nwave[1] - nwave[0])))
            nwave = np.array([np.mean(chunk) for chunk in np.array_split(nwave, nbins)])
            nmod = np.array([np.mean(chunk) for chunk in np.array_split(models[j], nbins)])
            nmod = convolve(nmod, G, boundary='extend')
            I = interp1d(nwave, nmod, kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
            M[i, j] = I(wave)
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
    G = Gaussian1DKernel(1.5)
    for j in np.arange(models.shape[0]):
        for i, z in enumerate(zbins):
            nwave = orig_wave * (1.+z)
            nbins = int(len(nwave) / (2. / (nwave[1] - nwave[0])))
            nwave = np.array([np.mean(chunk) for chunk in np.array_split(nwave, nbins)])
            nmod = np.array([np.mean(chunk) for chunk in np.array_split(models[j], nbins)])
            nmod = convolve(nmod, G, boundary='extend')
            I = interp1d(nwave, nmod, kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
            M[i, j] = I(wave)
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M