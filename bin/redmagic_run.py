#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate the redMaGiC red galaxy selection')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-C', '--clobber', action='store_true', default=False, help='Clobber existing run')
    parser.add_argument('-n', '--nrandoms', action='store', type=int, required=False,
                        help='Number of randoms to make')

    args = parser.parse_args()

    redmapper.redmagic.run_redmagic_task(args.configfile, clobber=args.clobber, n_randoms=args.nrandoms)
