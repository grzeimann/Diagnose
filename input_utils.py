#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 12:54:43 2021

@author: gregz
"""
import argparse as ap
import logging

def setup_logging(logname='input_utils'):
    '''Set up a logger for shuffle with a name ``input_utils``.

    Use a StreamHandler to write to stdout and set the level to DEBUG if
    verbose is set from the command line
    '''
    log = logging.getLogger('input_utils')
    if not len(log.handlers):
        fmt = '[%(levelname)s - %(asctime)s] %(message)s'
        fmt = logging.Formatter(fmt)

        level = logging.INFO

        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        log = logging.getLogger('input_utils')
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
    return log

def parse_args(argv=None):
    """Parse the command line arguments

    Parameters
    ----------
    argv : list of string
        arguments to parse; if ``None``, ``sys.argv`` is used

    Returns
    -------
    Namespace
        parsed arguments
    """
    
    description = " Classifying Spectra in a VIRUS Catalog "
                     
    parser = ap.ArgumentParser(description=description,
                            formatter_class=ap.RawTextHelpFormatter)
                            
    parser.add_argument("catalog", nargs='?', type=str,
                        help='''Filename for the catalog''')
    
    parser.add_argument("-s", "--suffix", type=int,
                        help='''Filename for the catalog''',
                        default=1)

    parser.add_argument("-li", "--low_index", type=int,
                        help='''Lower Index''', default=None)
    
    parser.add_argument("-hi", "--high_index", type=int,
                        help='''Upper Index''', default=None)
    
    parser.add_argument("-if", "--index_filename", type=str,
                        help='''Indice Filename''', default=None)

    parser.add_argument("-hr","--highres", help='''High Res Mode?''',
                        action="count", default=0)
                        
    args = parser.parse_args(args=argv)

    return args