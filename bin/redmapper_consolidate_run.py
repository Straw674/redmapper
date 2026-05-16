#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate a redMaPPer parallel run')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-l', '--lambda_cuts', action='store', type=float,
                        nargs='+', required=False, help='Minimum lambdas')
    parser.add_argument('-v', '--vlim_lstars', action='store', type=float,
                        nargs='+', required=False, help='Volume-limit L* cuts')

    args = parser.parse_args()

    if args.vlim_lstars is None:
        vlim_lstars = []
    else:
        vlim_lstars = args.vlim_lstars

    redmapper.pipeline.run_redmapper_consolidate_task(args.configfile, lambda_cuts=args.lambda_cuts, vlim_lstars=vlim_lstars)
