#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:06:16 2021

@author: gregz
"""
import numpy as np

stellar_template_file = 'miles_pca_ncomp_10.fits'
vbins = np.arange(-500, 550, 50)
zbins_gal = np.arange(0.005, 0.471, .001)
zbins_qso = np.arange(0.1, 4.01, 0.01)