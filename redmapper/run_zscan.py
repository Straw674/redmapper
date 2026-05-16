"""Function to compute richness for a set of ra/dec positions by scanning redshift.
"""
from functools import reduce
import sys

import fitsio
import numpy as np
import esutil

from .cluster import ClusterCatalog
from .background import read_background
from .mask import compute_maskgals_mark
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from . import depthmap
from .depth_fitting import apply_depthlim
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .centering import CENTERING_FUNCS
from .configuration import Configuration
from .logger import logger

def _zscan_more_setup(state, **kwargs):
    config = state['config']
    logger.info("Reading in catalog file...")
    cat = ClusterCatalog.from_catfile(config.catfile,
                                           zredstr=state['zredstr'],
                                           config=config,
                                           bkg=state['bkg'],
                                           zredbkg=state['zredbkg'],
                                           cosmo=state['cosmo'],
                                           r0=state['r0'],
                                           beta=state['beta'])

    nzstep = int(np.ceil((config.zrange[1] - config.zrange[0])/config.zscan_zstep))
    z_array = np.arange(nzstep, dtype=np.float64)*config.zscan_zstep + config.zrange[0]
    state['z_array'] = z_array
    state['nzstep'] = nzstep

    zscan_dtype = [('z_steps', 'f4', nzstep),
                   ('lambda_steps', 'f4', nzstep),
                   ('likelihood_steps', 'f4', nzstep),
                   ('lmax', 'f4'),
                   ('max_ind', 'i4'),
                   ('zmax', 'f4'),
                   ('ra_opt', 'f8'),
                   ('dec_opt', 'f8'),
                   ('lambda_opt', 'f4'),
                   ('lambda_opt_e', 'f4'),
                   ('z_lambda_opt', 'f4'),
                   ('z_lambda_opt_e', 'f4')]

    cat.add_fields(zscan_dtype)
    generate_mem_match_ids(cat)

    cat.ra_orig = cat.ra
    cat.dec_orig = cat.dec

    cat.z_init = z_array[0]
    cat.z = z_array[0]

    state['cat'] = cat
    
    state['refine_r0'] = config.percolation_r0
    state['refine_beta'] = config.percolation_beta

    if state['refine_beta'] == 0.0:
        state['refine_maxrad'] = 1.2*state['refine_r0']
    else:
        state['refine_maxrad'] = state['refine_r0']*(300./100.)**state['refine_beta']

    return state, True

def _zscan_process_cluster(cluster, state):
    bad = False
    done = False
    config = state['config']

    cluster.z_steps[:] = state['z_array']
    cluster.lambda_steps[:] = -1.0
    cluster.likelihood_steps[:] = -1.0
    cluster.lmax = -1.0
    cluster.max_ind = -1
    cluster.zmax = -1.0

    if state['depthstr'] is None:
        apply_depthlim(state['mask']['maskgals'],
                       cluster.neighbors.refmag,
                       cluster.neighbors.refmag_err,
                       state['depthlim_pars'])

    for zb, zuse in enumerate(state['z_array']):
        cluster.redshift = zuse
        cluster.update_neighbors_dist()

        maxmag = cluster.mstar - 2.5*np.log10(state['limlum'])

        lc, = np.where((cluster.neighbors.refmag < maxmag) &
                       (cluster.neighbors.r < state['maxrad']))

        if lc.size < 2:
            continue

        state['mask']['maskgals'].mark = compute_maskgals_mark(state['mask']['mask_data'], cluster, state['mask']['maskgals'], rng=state['mask']['rng'], config=config)

        if state['depthstr'] is not None:
            depthmap.compute_maskdepth(state['depthstr'], state['mask']['maskgals'],
                                       cluster.ra, cluster.dec, cluster.mpc_scale)

        lam = cluster.calc_richness(state['mask'], index=lc, calc_err=False)

        if lam < config.zscan_minlambda:
            continue

        incut, = np.where((cluster.neighbors.pmem > 0.0) &
                          (cluster.neighbors.r > 1e-6) &
                          (cluster.neighbors.pmem < 1.0))
        if incut.size > 0:
            like = -lam/cluster.scaleval - np.sum(np.log(1.0 - cluster.neighbors.pmem[incut]))
        else:
            like = -1.0

        cluster.lambda_steps[zb] = lam
        cluster.likelihood_steps[zb] = like

    cluster.max_ind = np.argmax(cluster.likelihood_steps)
    cluster.lmax = cluster.likelihood_steps[cluster.max_ind]
    cluster.zmax = cluster.z_steps[cluster.max_ind]

    if cluster.lambda_steps[cluster.max_ind] < config.zscan_minlambda:
        bad = True
        return bad

    zuse = cluster.zmax
    cluster.r0 = state['refine_r0']
    cluster.beta = state['refine_beta']

    cluster.redshift = zuse
    maxmag = cluster.mstar - 2.5*np.log10(state['limlum'])

    cluster.find_neighbors(2.0*state['refine_maxrad'], state['gals'], megaparsec=True, maxmag=maxmag)

    bad = False
    for i in range(2):
        if bad:
            continue

        cluster.redshift = zuse

        state['mask']['maskgals'].mark = compute_maskgals_mark(state['mask']['mask_data'], cluster, state['mask']['maskgals'], rng=state['mask']['rng'], config=config)
        if state['depthstr'] is not None:
            depthmap.compute_maskdepth(state['depthstr'], state['mask']['maskgals'],
                                       cluster.ra, cluster.dec, cluster.mpc_scale)

        lam = cluster.calc_richness(state['mask'])

        if (lam/cluster.scaleval < config.zscan_minlambda):
            bad = True
            continue

        if i == 0:
            z_lambda, z_lambda_e, pzbins, pz, _ = compute_zlambda(cluster, state['mask'], cluster.redshift,
                                                               calc_err=True, calcpz=True)
            cluster.z_lambda = z_lambda
            cluster.z_lambda_err = z_lambda_e
            cluster.pzbins[:] = pzbins
            cluster.pz[:] = pz

            if z_lambda > 0.0:
                zuse = z_lambda

        if cluster.z_lambda < 0.0:
            bad = True
            continue

    if bad:
        reset_bad_values(cluster)
        return bad

    cluster.redshift = cluster.z_lambda

    cent = CENTERING_FUNCS[config.centerclass](cluster, config, zlambda_corr=state['zlambda_corr'])

    if not cent['success'] or cent['ngood'] == 0:
        logger.info("Could not find optical center on a cluster.")
        return False

    cluster.ra_opt = cent['ra'][0]
    cluster.dec_opt = cent['dec'][0]

    if cent['index'][0] >= 0:
        cluster.mag[:] = cluster.neighbors.mag[cent['index'][0], :]
        cluster.mag_err[:] = cluster.neighbors.mag_err[cent['index'][0], :]
        cluster.refmag = cluster.neighbors.refmag[cent['index'][0]]
        cluster.refmag_err = cluster.neighbors.refmag_err[cent['index'][0]]
        cluster.ebv_mean = cluster.neighbors.ebv[cent['index'][0]]
        if state['did_read_zreds']:
            cluster.zred = cluster.neighbors.zred[cent['index'][0]]
            cluster.zred_e = cluster.neighbors.zred_e[cent['index'][0]]
            cluster.zred_chisq = cluster.neighbors.zred_chisq[cent['index'][0]]

        cluster.id_cent[:] = cluster.neighbors.id[cent['index']]

    cluster.ncent_good = cent['ngood']
    cluster.ra_cent[:] = cent['ra']
    cluster.dec_cent[:] = cent['dec']
    cluster.p_cen[:] = cent['p_cen']
    cluster.q_cen[:] = cent['q_cen']
    cluster.p_fg[:] = cent['p_fg']
    cluster.q_miss = cent['q_miss']
    cluster.p_sat[:] = cent['p_sat']
    cluster.p_c[:] = cent['p_c']

    opt_cluster = cluster.copy()
    opt_cluster.ra = cluster.ra_opt
    opt_cluster.dec = cluster.dec_opt
    opt_cluster.update_neighbors_dist()

    lam = opt_cluster.calc_richness(state['mask'])

    cluster.lambda_opt = opt_cluster.Lambda
    cluster.lambda_opt_e = opt_cluster.Lambda_e

    z_lambda_opt, z_lambda_opt_e, _, _, _ = compute_zlambda(opt_cluster, state['mask'], opt_cluster.redshift,
                                                         calc_err=True, calcpz=False)

    cluster.z_lambda_opt = z_lambda_opt
    cluster.z_lambda_opt_e = z_lambda_opt_e
    return False

def run_zscan(conf):
    """
    Run zscan on a catalog.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf
        
    cat, members = run_cluster_pipeline(
        config,
        runmode='zscan',
        filetype='zscan',
        more_setup_fn=_zscan_more_setup,
        process_cluster_fn=_zscan_process_cluster,
        read_gals=True,
        read_zreds=True,
        zreds_required=True,
        zredbkg_required=True,
        cutgals_bkgrange=True,
        cutgals_chisqmax=False,
        use_rmask_settings=False,
        do_percolation_masking=False,
        do_lam_plusminus=True,
        match_centers_to_galaxies=False,
        record_members=True,
        do_correct_zlambda=True,
        do_pz=True,
        use_maxmag_in_matching=False,
        min_lambda=config.zscan_minlambda
    )
        
    return cat, members


