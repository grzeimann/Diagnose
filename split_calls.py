#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:53:22 2021

@author: gregz
"""

import numpy as np
import argparse as ap
import os.path as op

from astropy.io import fits

def get_script_path():
    '''
    Get script path, aka, where does Diagnose live?
    '''
    return op.dirname(op.realpath(__file__))

description = " Classifying Spectra in a VIRUS Catalog "
                     
parser = ap.ArgumentParser(description=description,
                        formatter_class=ap.RawTextHelpFormatter)
                        
parser.add_argument("catalog", nargs='?', type=str,
                    help='''Filename for the catalog''')

parser.add_argument("-N", "--n_tasks", type=int,
                    help='''Total number of tasks''', 
                    default=80)

parser.add_argument("-Nf", "--n_files", type=int,
                    help='''Total number of files''', 
                    default=2)

parser.add_argument("-n", "--normalize",
                help='''Flux Normalization''',
                action="count", default=0)

parser.add_argument("-q", "--quick",
                help='''Use existing models''',
                action="count", default=0)
                    
args = parser.parse_args(args=None)

dirname = get_script_path()
diagnose_path = op.join(dirname, 'diagnose.py')
call = ('python3 %s %s '
        '-li %i -hi %i -s %i')

if args.quick:
    call = call + ' -q'
if args.normalize:
    call = call + ' -n'

f = fits.open(args.catalog)
L = f[2].data.shape[0]
indices = np.arange(L, dtype=int)
chunks = np.array_split(indices, args.n_tasks)

lows = [chunk[0] for chunk in chunks]
highs = [chunk[-1]+1 for chunk in chunks]

cnt = 1
low_chunks = np.array_split(lows, args.n_files)
high_chunks = np.array_split(highs, args.n_files)
for k in np.arange(args.n_files):
    with open('diagnose_calls_%i' % (k+1), 'w') as out_file:              
        for li, hi in zip(low_chunks[k], high_chunks[k]):
            call_str = call % (diagnose_path, args.catalog, li, hi, cnt)
            out_file.write(call_str + '\n')
            cnt += 1




