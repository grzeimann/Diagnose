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
                    default=None)

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
with open('diagnose_calls', 'w') as out_file:              
    for li, hi in zip(lows, highs):
        call_str = call % (diagnose_path, args.catalog, li, hi, cnt)
        out_file.write(call_str + '\n')
        cnt += 1




