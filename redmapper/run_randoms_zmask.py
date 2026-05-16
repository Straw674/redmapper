"""Function to run the redmapper randoms using zmask randoms.
"""
import fitsio
import numpy as np
import esutil
import os

from .cluster import ClusterCatalog
from .catalog import Catalog
from .background import read_background
from .mask import get_mask, calc_maskcorr
from .galaxy import GalaxyCatalog
from .randoms import RandomCatalog
from .cluster import Cluster
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .configuration import Configuration

def _randoms_more_setup(state, **kwargs):
    config = state['config']
    incat = RandomCatalog.from_randfile(config.randfile,
                                        nside=config.nside,
                                        hpix=config.hpix,
                                        border=config.border)

    dtype = [('MEM_MATCH_ID', 'i4'),
             ('RA', 'f8'),
             ('DEC', 'f8'),
             ('Z', 'f4'),
             ('LAMBDA', 'f4'),
             ('LAMBDA_E', 'f4'),
             ('Z_LAMBDA', 'f4'),
             ('Z_LAMBDA_E', 'f4'),
             ('R_LAMBDA', 'f4'),
             ('R_MASK', 'f4'),
             ('SCALEVAL', 'f4'),
             ('MASKFRAC', 'f4'),
             ('EBV_MEAN', 'f4'),
             ('ID_INPUT', 'i4'),
             ('LAMBDA_IN', 'f4'),
             ('Z_IN', 'f4')]

    cat = ClusterCatalog.zeros(incat.size,
                                    zredstr=state['zredstr'],
                                    config=config,
                                    bkg=state['bkg'],
                                    cosmo=state['cosmo'],
                                    r0=state['r0'],
                                    beta=state['beta'],
                                    dtype=dtype)

    cat.ra = incat.ra
    cat.dec = incat.dec
    cat.mem_match_id = incat.id
    cat.z = incat.z
    cat.Lambda = incat.Lambda
    cat.id_input = incat.id_input
    cat.lambda_in = incat.Lambda
    cat.z_in = incat.z

    state['cat'] = cat
    return state, True

def _randoms_process_cluster(cluster, state):
    cluster.Lambda = cluster.lambda_in
    cluster.r_lambda = state['r0']*(cluster.Lambda/100.)**state['beta']
    cluster.r_mask = cluster.r_lambda

    maxmag = cluster.mstar - 2.5*np.log10(state['config'].lval_reference)
    cpars = calc_maskcorr(state['mask']['maskgals'], cluster.mstar, maxmag, state['zredstr']['limmag'], state['mask']['rng'])
    cval = np.sum(cpars*cluster.r_lambda**(np.arange(cpars.size)[::-1]))
    cluster.scaleval = 1./(1. - cval)

    return False

def run_randoms_zmask(conf):
    """
    Run randoms on a zmask.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf

    read_gals = config.depthfile is None

    cat, members = run_cluster_pipeline(
        config,
        runmode='percolation',
        filetype='randoms_zmask',
        more_setup_fn=_randoms_more_setup,
        process_cluster_fn=_randoms_process_cluster,
        read_gals=read_gals,
        read_zreds=False,
        zreds_required=False,
        cutgals_bkgrange=False,
        cutgals_chisqmax=False,
        do_lam_plusminus=False,
        match_centers_to_galaxies=False,
        do_percolation_masking=False,
        record_members=False
    )
        
    return cat, members
