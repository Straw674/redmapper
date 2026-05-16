"""Function to run the first pass through a catalog for cluster finding.
"""
import fitsio
import numpy as np
import esutil
import os

from .cluster import ClusterCatalog
from .catalog import Catalog
from .background import read_background
from .mask import get_mask_values
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from .redsequence import redsequence_mstar
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .configuration import Configuration
from .logger import logger

def _firstpass_more_setup(state, keepz=False, cleaninput=False, specmode=False, **kwargs):
    """Setup input catalogs for firstpass."""
    incat = None
    config = state['config']

    if config.seedfile is not None and os.path.isfile(config.seedfile):
        logger.info("%s: Firstpass using seedfile: %s" % (state['hpix_logstr'], config.seedfile))
        incat = Catalog.from_fits_file(config.seedfile)
    else:
        if specmode:
            raise RuntimeError("Must have config.seedfile for run_firstpass in specmode.")
        elif not state['did_read_zreds']:
            raise RuntimeError("Must have config.seedfile for run_firstpass with no zreds.")
        logger.info("%s: Firstpass using zreds as input" % (state['hpix_logstr']))

    if incat is not None:
        cat = ClusterCatalog.zeros(incat.size, zredstr=state['zredstr'], config=config,
                                   bkg=state['bkg'], cosmo=state['cosmo'],
                                   r0=state['r0'], beta=state['beta'])
        cat.ra = incat.ra
        cat.dec = incat.dec
        cat.mag = incat.mag
        cat.mag_err = incat.mag_err
        cat.refmag = incat.refmag
        cat.refmag_err = incat.refmag_err
        cat.zred = incat.zred
        cat.zred_e = incat.zred_e
        cat.zred_chisq = incat.zred_chisq
        cat.chisq = incat.zred_chisq
        cat.ebv_mean = incat.ebv
        cat.z_spec_init = incat.zspec

        if specmode:
            cat.z_init = incat.zspec
            cat.z = incat.zspec
        else:
            cat.z_init = incat.zred
            cat.z = incat.zred

        cuse = ((cat.zred >= config.zrange[0]) & (cat.zred <= config.zrange[1]))
        cuse &= (cat.chisq < config.chisq_max)

        use, = np.where(cuse)
        if use.size == 0:
            logger.info("No good zred inputs for firstpass on pixel %s" % (state['hpix_logstr']))
            state['cat'] = None
            return state, False

        cat = cat[use]

        if cleaninput:
            catmask = get_mask_values(state['mask']['mask_data'], cat.ra, cat.dec, rng=state['mask']['rng'], config=state['mask']['config'])
            cat = cat[catmask]
            if cat.size == 0:
                logger.info("No zred positions are in the mask on pixel %s" % (state['hpix_logstr']))
                state['cat'] = None
                return state, False
        state['cat'] = cat
    else:
        gals = state['gals']
        cuse = ((gals.zred >= config.zrange[0]) & (gals.zred <= config.zrange[1]))
        cuse &= (gals.chisq < config.chisq_max)
        use, = np.where(cuse)

        if use.size == 0:
            logger.info("No usable zred inputs for firstpass on pixel %s" % (state['hpix_logstr']))
            state['cat'] = None
            return state, False

        mstar = redsequence_mstar(state['zredstr'], gals.zred[use])
        mlim = mstar - 2.5 * np.log10(config.lval_reference)
        good, = np.where(gals.refmag[use] < mlim)

        if good.size == 0:
            logger.info("No good zred inputs for firstpass on pixel %s" % (state['hpix_logstr']))
            state['cat'] = None
            return state, False

        cat = ClusterCatalog.zeros(good.size, zredstr=state['zredstr'], config=config,
                                   bkg=state['bkg'], cosmo=state['cosmo'],
                                   r0=state['r0'], beta=state['beta'])
        cat.ra = gals.ra[use[good]]
        cat.dec = gals.dec[use[good]]
        cat.mag = gals.mag[use[good]]
        cat.mag_err = gals.mag_err[use[good]]
        cat.refmag = gals.refmag[use[good]]
        cat.refmag_err = gals.refmag_err[use[good]]
        cat.zred = gals.zred[use[good]]
        cat.zred_e = gals.zred_e[use[good]]
        cat.zred_chisq = gals.chisq[use[good]]
        cat.chisq = gals.chisq[use[good]]

        cat.z_init = cat.zred
        cat.z = cat.zred

        state['cat'] = cat

    generate_mem_match_ids(state['cat'])

    state['maxiter'] = 1 if specmode else config.firstpass_niter
    state['keepz'] = keepz
    state['specmode'] = specmode

    return state, True

def _firstpass_process_cluster(cluster, state):
    bad = False
    done = False
    config = state['config']

    for i in range(state['maxiter']):
        if bad:
            done = True
            continue

        lam = cluster.calc_richness(state['mask'], calc_err=False)

        if (lam < np.abs(config.firstpass_minlambda / cluster.scaleval)):
            bad = True
            done = True
            reset_bad_values(cluster)
            continue

        if i < state['maxiter']:
            z_lambda, z_lambda_e, _, _, niter = compute_zlambda(cluster, state['mask'], cluster.redshift,
                                                         calc_err=True, calcpz=False)
            cluster.z_lambda = z_lambda
            cluster.z_lambda_err = z_lambda_e

            if z_lambda < config.zrange[0] or z_lambda > config.zrange[1]:
                bad = True
                done = True
                reset_bad_values(cluster)
                continue

            if not state['keepz']:
                cluster.redshift = z_lambda

    if bad:
        cluster.z_lambda = -1.0
        cluster.z_lambda_e = -1.0
        cluster.z_lambda_niter = 0
    else:
        cluster.z_lambda = z_lambda
        cluster.z_lambda_e = z_lambda_e
        cluster.z_lambda_niter = niter

    cind = np.argmin(cluster.neighbors.r)
    cluster.chisq = cluster.neighbors.chisq[cind]

    if state['specmode']:
        cluster.z = cluster.z_spec_init
    else:
        cluster.z = cluster.z_lambda

    return bad

def run_firstpass(conf, keepz=False, cleaninput=False, specmode=False):
    """
    Run firstpass on a catalog.

    Parameters
    ----------
    conf: `redmapper.Configuration` or `str`
       Configuration object or filename of configuration object
    keepz: `bool`, optional
       Keep input redshifts?  (Otherwise use z_lambda).
       Default is False.
    cleaninput: `bool`, optional
       Clean seed clusters that are out of the footprint?  Default is False.
    specmode: `bool`, optional
       Run in spectroscopic-seed mode.  Default is False.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf

    filetype = 'firstpass_spec' if specmode else 'firstpass'

    cat, members = run_cluster_pipeline(
        config,
        runmode='firstpass',
        filetype=filetype,
        more_setup_fn=_firstpass_more_setup,
        process_cluster_fn=_firstpass_process_cluster,
        read_gals=True,
        read_zreds=True,
        zreds_required=False,
        cutgals_bkgrange=True,
        cutgals_chisqmax=True,
        keepz=keepz,
        cleaninput=cleaninput,
        specmode=specmode,
        min_lambda=config.firstpass_minlambda
    )


    return cat, members


