"""Function to compute richness/redshifts for a set of ra/dec/z positions.
"""
import fitsio
import numpy as np
import esutil

from .cluster import ClusterCatalog
from .background import read_background
from .mask import get_mask
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .configuration import Configuration
from .logger import logger

def _runcat_more_setup(state, do_percolation_masking=False, maxiter=5, tol=0.005, converge_zlambda=False, **kwargs):
    config = state['config']
    logger.info("Reading in catalog file...")
    cat = ClusterCatalog.from_catfile(config.catfile,
                                           zredstr=state['zredstr'],
                                           config=config,
                                           bkg=state['bkg'],
                                           cosmo=state['cosmo'],
                                           r0=state['r0'],
                                           beta=state['beta'])

    generate_mem_match_ids(cat)

    if config.percolation_memlum is not None:
        if (config.percolation_memlum > 0.0 and
            config.percolation_memlum < config.lval_reference):
            if config.percolation_memlum < state['limlum']:
                state['limlum'] = config.percolation_memlum

    if config.percolation_lmask is not None:
        if config.percolation_lmask > 0.0:
            if config.percolation_lmask < state['limlum']:
                state['limlum'] = config.percolation_lmask

    state['cat'] = cat
    state['maxiter'] = maxiter
    state['tol'] = tol
    state['converge_zlambda'] = converge_zlambda
    
    return state, True

def _runcat_process_cluster(cluster, state):
    bad = False
    iteration = 0
    done = False
    config = state['config']
    maxmag = cluster.mstar - 2.5*np.log10(state['limlum'])

    while iteration < state['maxiter'] and not done:
        if bad:
            done = True
            continue

        if (cluster.maskfrac > 0.7):
            bad = True
            done = True
            continue

        lam = cluster.calc_richness(state['mask'])

        if (lam < 3.0):
            bad = True
            done = True
            reset_bad_values(cluster)
            continue

        z_lambda, z_lambda_e, pzbins, pz, niter = compute_zlambda(cluster, state['mask'], cluster.redshift,
                                                           calc_err=True, calcpz=True)
        cluster.z_lambda = z_lambda
        cluster.z_lambda_err = z_lambda_e
        cluster.pzbins[:] = pzbins
        cluster.pz[:] = pz

        if z_lambda < 0.0:
            bad = True
            done = True
            reset_bad_values(cluster)
            continue

        cluster.z_lambda = z_lambda
        cluster.z_lambda_e = z_lambda_e
        cluster.z_lambda_niter = niter
        cluster.pzbins = pzbins
        cluster.pz = pz

        if state['converge_zlambda']:
            if (np.abs(cluster.redshift - cluster.z_lambda) < state['tol']):
                done = True

            cluster.redshift = cluster.z_lambda
        else:
            done = True

        iteration += 1

    return bad

def run_catalog(conf, do_percolation_masking=False, maxiter=5, converge_zlambda=False):
    """
    Run catalog on a set of positions.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf

    cat, members = run_cluster_pipeline(
        config,
        runmode='percolation',
        filetype='lambda_chisq',
        more_setup_fn=_runcat_more_setup,
        process_cluster_fn=_runcat_process_cluster,
        read_gals=True,
        read_zreds=False,
        zreds_required=False,
        cutgals_bkgrange=True,
        cutgals_chisqmax=True,
        do_percolation_masking=do_percolation_masking,
        maxiter=maxiter,
        converge_zlambda=converge_zlambda,
        do_lam_plusminus=True,
        match_centers_to_galaxies=True,
        record_members=True,
        do_correct_zlambda=True,
        do_pz=True
    )
        
    return cat, members


