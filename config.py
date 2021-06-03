#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:06:16 2021

@author: gregz
"""
import numpy as np

stellar_template_file = 'miles_pca_ncomp_10.fits'
vbins = np.linspace(-500, 500, 50)
zbins_gal = np.linspace(0.00, 0.47, 47*10+1)
zbins_qso = np.linspace(0.5, 3.5, int((3.5-0.4)/0.01+1))