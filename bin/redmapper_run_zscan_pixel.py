#!/usr/bin/env python

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run redmapper zscan for a single pixel')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-p', '--pixel', action='store', type=int, required=True,
                        help='Healpix pixel to run')
    parser.add_argument('-n', '--nside', action='store', type=int, required=True,
                        help='Healpix nside for pixel')
    parser.add_argument('-d', '--path', action='store', type=str, required=False, default='./',
                        help='Path to set for outputs')

    args = parser.parse_args()

    redmapper.pipeline.run_zscan_pixel_task(args.configfile, args.pixel, args.nside, path=args.path)
