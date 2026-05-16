"""Functional pipeline orchestrator and base class to run a cluster catalog through one stage of processing.

The ClusterRunner class is the base class for looping over all the clusters in
a catalog and performing computations on each cluster.  This includes first
passes, percolation, richness computation at existing positions/redshifts, etc.
"""
import fitsio
import numpy as np
import esutil
import os
import gc
from esutil.cosmology import Cosmo

from .configuration import Configuration
from .cluster import ClusterCatalog
from .background import read_background, read_zred_background
from .color_background import read_color_background
from .mask import get_mask, select_maskgals_sample, compute_maskgals_mark
from .galaxy import GalaxyCatalog
from .catalog import Catalog
from .cluster import Cluster
from . import depthmap
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from .redsequence import read_redsequence
from .depth_fitting import compute_depthlim_pars, apply_depthlim
from .utilities import getMemoryString
from .logger import logger

def setup_cluster_pipeline(config, runmode, read_gals=True, read_zreds=False, zreds_required=False,
                           zredbkg_required=False, use_colorbkg=False, use_parfile=True,
                           use_rmask_settings=True, cutgals_chisqmax=False, cutgals_bkgrange=False):
    """
    Setup common state for the cluster pipeline.
    """
    state = {}
    state['config'] = config
    state['hpix_logstr'] = "All" if len(config.hpix) == 0 else ", ".join(str(x) for x in config.hpix)
    state['r0'] = getattr(config, runmode + '_r0')
    state['beta'] = getattr(config, runmode + '_beta')

    if use_rmask_settings:
        state['rmask_0'] = config.percolation_rmask_0
        state['rmask_beta'] = config.percolation_rmask_beta
        state['rmask_gamma'] = config.percolation_rmask_gamma
        state['rmask_zpivot'] = config.percolation_rmask_zpivot
    else:
        state['rmask_0'] = state['r0']
        state['rmask_beta'] = state['beta']
        state['rmask_gamma'] = config.percolation_rmask_gamma
        state['rmask_zpivot'] = config.percolation_rmask_zpivot

    state['percolation_lmask'] = config.lval_reference if config.percolation_lmask < 0.0 else config.percolation_lmask

    if state['beta'] == 0.0:
        state['maxrad'] = 1.2 * state['r0']
        state['maxrad2'] = 1.2 * state['rmask_0']
    else:
        state['maxrad'] = state['r0'] * (300./100.)**state['beta']
        state['maxrad2'] = state['rmask_0'] * (300./100.)**state['rmask_beta']

    if state['maxrad2'] > state['maxrad']:
        state['maxrad'] = state['maxrad2']

    if config.bkg_local_compute or config.bkg_local_use:
        if config.bkg_local_annuli[1] > state['maxrad']:
            state['maxrad'] = config.bkg_local_annuli[1]

    if use_colorbkg:
        state['cbkg'] = read_color_background(config.bkgfile_color, usehdrarea=True)
        state['bkg'] = None
    else:
        state['bkg'] = read_background(config.bkgfile)
        state['cbkg'] = None

    state['zredbkg'] = read_zred_background(config.bkgfile) if zredbkg_required else None

    if use_parfile:
        state['zredstr'] = read_redsequence(config.parfile, fine=True)
    else:
        state['zredstr'] = read_redsequence(None, config=config)

    try:
        state['zlambda_corr'] = read_zlambda_correction(parfile=config.zlambdafile, zlambda_pivot=config.zlambda_pivot)
    except:
        state['zlambda_corr'] = None

    state['mask'] = get_mask(config)

    try:
        state['depthstr'] = depthmap.read_depth_map(config)
    except:
        state['depthstr'] = None

    if state['depthstr'] is None and not read_gals:
        raise RuntimeError("Must have a valid depthstr if read_gals is False")

    state['cosmo'] = config.cosmo
    zredfile = config.zredfile if read_zreds else None
    if zreds_required and zredfile is None:
        raise RuntimeError("zreds are required, but zredfile is None")

    state['gals'] = None
    state['did_read_zreds'] = False

    if read_gals:
        if cutgals_bkgrange:
            refmag_low = state['bkg']['refmagbins'][0]
            refmag_high = state['bkg']['refmagbins'][-1] + (state['bkg']['refmagbins'][1] - state['bkg']['refmagbins'][0])
        else:
            refmag_low = -1000.0
            refmag_high = 1000.0

        if config.use_tempfiles_to_conserve_memory:
            chisq_max = config.chisq_max if cutgals_chisqmax else 1e30
            state['gals'] = GalaxyCatalog.from_galfile(config.galfile,
                                                       nside=config.nside,
                                                       hpix=config.hpix,
                                                       border=config.border,
                                                       zredfile=zredfile,
                                                       use_tempfile=True,
                                                       refmag_range=[refmag_low, refmag_high],
                                                       chisq_max=chisq_max,
                                                       zspec=config.centering_use_zspec)
            if zredfile is not None:
                state['did_read_zreds'] = True
        else:
            state['gals'] = GalaxyCatalog.from_galfile(config.galfile,
                                                       nside=config.nside,
                                                       hpix=config.hpix,
                                                       border=config.border,
                                                       zredfile=zredfile,
                                                       zspec=config.centering_use_zspec)
            if zredfile is not None:
                state['did_read_zreds'] = True
            
            guse = ((state['gals'].refmag > refmag_low) & (state['gals'].refmag < refmag_high))
            if state['did_read_zreds'] and cutgals_chisqmax:
                guse &= (state['gals'].chisq < config.chisq_max)
            logger.info("Cutting %d of %d galaxies." % (guse.sum(), state['gals'].size))
            state['gals'] = state['gals'][guse]

        if len(state['gals']) == 0:
            logger.info(f"No good galaxies for {runmode} in pixel {state['hpix_logstr']}")
            return state, False

    if state['depthstr'] is None:
        try:
            state['depthlim_pars'] = compute_depthlim_pars(state['gals'].refmag, state['gals'].refmag_err)
        except RuntimeError:
            logger.info(f"Failed to obtain depth info in {runmode} for pixel {state['hpix_logstr']} with {len(state['gals'])} galaxies.  Skipping pixel.")
            return state, False
    else:
        state['depthlim_pars'] = None

    state['limlum'] = np.clip(config.lval_reference - 0.1, 0.01, None)
    return state, True

def generate_mem_match_ids(cat):
    min_id = cat.mem_match_id.min()
    max_id = cat.mem_match_id.max()
    if min_id == max_id:
        cat.mem_match_id = np.arange(cat.size) + 1
    else:
        if np.unique(cat.mem_match_id).size != cat.size:
            raise RuntimeError("Input values for mem_match_id are not unique (and not all unset)")
    return cat

def reset_bad_values(cluster):
    cluster.Lambda = -1.0
    cluster.Lambda_e = -1.0
    cluster.scaleval = -1.0
    cluster.z_lambda = -1.0
    cluster.z_lambda_e = -1.0

def run_cluster_pipeline(config, runmode, filetype, more_setup_fn, process_cluster_fn, postprocess_fn=None,
                         read_gals=True, read_zreds=False, zreds_required=False, zredbkg_required=False,
                         use_colorbkg=False, use_parfile=True, use_maxmag_in_matching=True, use_rmask_settings=True,
                         cutgals_chisqmax=False, cutgals_bkgrange=False, do_percolation_masking=False,
                         do_lam_plusminus=False, use_memradius=False, use_memlum=False, match_centers_to_galaxies=False,
                         min_lambda=-1.0, record_members=False, doublerun=False, do_correct_zlambda=False, do_pz=False,
                         **kwargs):
    """
    Run a functional pipeline over the clusters.
    """
    config.start_file_logging()

    state, success = setup_cluster_pipeline(
        config, runmode, read_gals=read_gals, read_zreds=read_zreds, zreds_required=zreds_required,
        zredbkg_required=zredbkg_required, use_colorbkg=use_colorbkg, use_parfile=use_parfile,
        use_rmask_settings=use_rmask_settings, cutgals_chisqmax=cutgals_chisqmax, cutgals_bkgrange=cutgals_bkgrange
    )

    if not success:
        return None, None

    # Note: more_setup_fn MUST set state['cat']
    state, success = more_setup_fn(state, **kwargs)
    if not success or state.get('cat') is None:
        return None, None

    cat = state['cat']
    gals = state['gals']

    if match_centers_to_galaxies:
        i0, i1, dist = gals.match_many(cat.ra, cat.dec, 1./3600.)
        cat.refmag[i0] = gals.refmag[i1]
        cat.refmag_err[i0] = gals.refmag_err[i1]
        cat.mag[i0, :] = gals.mag[i1, :]
        cat.mag_err[i0, :] = gals.mag_err[i1, :]
        if state['did_read_zreds']:
            cat.zred = gals.zred[i1]
            cat.zred_e = gals.zred_e[i1]
            cat.zred_chisq = gals.zred_chisq[i1]

    if do_percolation_masking or doublerun:
        state['pgal'] = np.zeros(gals.size, dtype=np.float32)
    pass

    members_list = []
    nruniter = 2 if doublerun else 1
    
    # Store settings in state for process_cluster_fn if needed
    state['use_maxmag_in_matching'] = use_maxmag_in_matching
    state['do_percolation_masking'] = do_percolation_masking
    state['do_lam_plusminus'] = do_lam_plusminus
    state['do_correct_zlambda'] = do_correct_zlambda
    state['do_pz'] = do_pz
    state['min_lambda'] = min_lambda

    for it in range(nruniter):
        if doublerun:
            if it == 0:
                logger.info("First iteration...")
                state['do_percolation_masking'] = False
            else:
                logger.info("Second iteration with percolation...")
                if 'doublerun_sort_fn' in kwargs:
                    cat = kwargs['doublerun_sort_fn'](cat)
                    state['cat'] = cat
                state['do_percolation_masking'] = True
                record_members = True

        cctr = 0
        for cluster in cat:
            state['mask']['maskgals'], cluster.maskgal_index = select_maskgals_sample(config, state['mask']['maskgals_all'], state['mask']['rng'])

            if (cctr % 1000) == 0:
                logger.info("%s: Working on cluster %d of %d" % (state['hpix_logstr'], cctr, cat.size))
            pass
            cctr += 1

            if use_maxmag_in_matching:
                maxmag = cluster.mstar - 2.5*np.log10(state['limlum'])
            else:
                maxmag = None

            if read_gals:
                cluster.find_neighbors(state['maxrad'], gals, megaparsec=True, maxmag=maxmag)
                if cluster.neighbors.size == 0:
                    reset_bad_values(cluster)
                    continue

                if state['do_percolation_masking']:
                    cluster.neighbors.pfree[:] = 1.0 - state['pgal'][cluster.neighbors.index]
                else:
                    cluster.neighbors.pfree[:] = 1.0

            if state['depthstr'] is None:
                apply_depthlim(state['mask']['maskgals'], cluster.neighbors.refmag, cluster.neighbors.refmag_err, state['depthlim_pars'])
            else:
                depthmap.compute_maskdepth(state['depthstr'], state['mask']['maskgals'], cluster.ra, cluster.dec, cluster.mpc_scale)

            cluster.lim_exptime = np.median(state['mask']['maskgals'].exptime)
            cluster.lim_limmag = np.median(state['mask']['maskgals'].limmag)
            cluster.lim_limmag_hard = config.limmag_catalog

            state['mask']['maskgals'].mark = compute_maskgals_mark(state['mask']['mask_data'], cluster, state['mask']['maskgals'], rng=state['mask']['rng'], config=config)

            inside, = np.where(state['mask']['maskgals'].r < 1.0)
            bad, = np.where(state['mask']['maskgals'].mark[inside] == 0)
            cluster.maskfrac = float(bad.size) / float(inside.size)

            if cluster.maskfrac == 1.0 or cluster.lim_limmag <= 1.0:
                bad_cluster = True
            else:                bad_cluster = process_cluster_fn(cluster, state)


            if bad_cluster:
                reset_bad_values(cluster)
                continue

            if read_gals:
                if config.bkg_local_compute and not config.bkg_local_use:
                    depth = state['depthlim_pars'] if state['depthstr'] is None else state['depthstr']
                    cluster.bkg_local = cluster.compute_bkg_local(state['mask'], depth)

            if state['do_correct_zlambda'] and state['zlambda_corr'] is not None and read_gals:
                if state['do_pz']:
                    zlam, zlam_e, pzbins, pzvals = apply_zlambda_correction(
                        state['zlambda_corr'], cluster.Lambda, cluster.z_lambda, cluster.z_lambda_e,
                        pzbins=cluster.pzbins, pzvals=cluster.pz
                    )
                    cluster.pzbins = pzbins
                    cluster.pzvals = pzvals
                else:
                    zlam, zlam_e = apply_zlambda_correction(
                        state['zlambda_corr'], cluster.Lambda, cluster.z_lambda, cluster.z_lambda_e
                    )
                cluster.z_lambda = zlam
                cluster.z_lambda_e = zlam_e

            inside, = np.where(state['mask']['maskgals'].r < cluster.r_lambda)
            bad, = np.where(state['mask']['maskgals'].mark[inside] == 0)
            if inside.size == 0:
                cluster.maskfrac = 1.0
            else:
                cluster.maskfrac = float(bad.size) / float(inside.size)

            if state['do_lam_plusminus'] and read_gals:
                cluster_temp = cluster.copy()
                cluster_temp.redshift = cluster.z_lambda - config.zlambda_epsilon
                lam_zmeps = cluster_temp.calc_richness(state['mask'])
                elambda_zmeps = cluster_temp.lambda_e
                cluster_temp.redshift = cluster.z_lambda + config.zlambda_epsilon
                lam_zpeps = cluster_temp.calc_richness(state['mask'])
                elambda_zpeps = cluster_temp.lambda_e

                if lam_zmeps > 0 and lam_zpeps > 0:
                    cluster.dlambda_dz = (np.log(lam_zpeps) - np.log(lam_zmeps)) / (2. * config.zlambda_epsilon)
                    cluster.dlambda_dz2 = (np.log(lam_zpeps) + np.log(lam_zmeps) - 2.*np.log(cluster.Lambda)) / (config.zlambda_epsilon**2.)
                    cluster.dlambdavar_dz = (elambda_zpeps**2. - elambda_zmeps**2.) / (2.*config.zlambda_epsilon)
                    cluster.dlambdavar_dz2 = (elambda_zpeps**2. + elambda_zmeps**2. - 2.*cluster.Lambda_e**2.) / (config.zlambda_epsilon**2.)

            if state['do_percolation_masking'] and read_gals:
                r_mask = (state['rmask_0'] * (cluster.Lambda/100.)**state['rmask_beta'] *
                          ((1. + cluster.redshift)/(1. + state['rmask_zpivot']))**state['rmask_gamma'])
                if r_mask < cluster.r_lambda:
                    r_mask = cluster.r_lambda
                cluster.r_mask = r_mask

                lim = cluster.mstar - 2.5*np.log10(state['percolation_lmask'])
                u, = np.where((cluster.neighbors.refmag < lim) &
                              (cluster.neighbors.r < r_mask) &
                              (cluster.neighbors.p > 0.0))
                if u.size > 0:
                    state['pgal'][cluster.neighbors.index[u]] += cluster.neighbors.p[u]

            if read_gals:
                pfree_temp = cluster.neighbors.pfree[:]

            if (use_memradius or use_memlum) and read_gals:
                ok = (cluster.neighbors.p > 0.01)
                if use_memradius:
                    ok &= (cluster.neighbors.r < config.percolation_memradius * cluster.r_lambda)
                if use_memlum:
                    ok &= (cluster.neighbors.refmag < (cluster.mstar - 2.5*np.log10(config.percolation_memlum)))
                pfree_temp[~ok] = 0.0
            elif read_gals:
                ok = (cluster.neighbors.pmem > 0.01)
                pfree_temp[~ok] = 0.0

            if record_members and read_gals:
                pfree_temp = cluster.neighbors.pfree[:]
                if use_memradius or use_memlum:
                    ok = (cluster.neighbors.p > 0.01)
                    if use_memradius:
                        ok &= (cluster.neighbors.r < config.percolation_memradius * cluster.r_lambda)
                    if use_memlum:
                        ok &= (cluster.neighbors.refmag < (cluster.mstar - 2.5*np.log10(config.percolation_memlum)))
                    pfree_temp[~ok] = 0.0
                else:
                    ok = (cluster.neighbors.pmem > 0.01)
                    pfree_temp[~ok] = 0.0

                memuse, = np.where((pfree_temp > 0.01) | (cluster.neighbors.centering_cand == 1))
                mem_temp = Catalog.zeros(memuse.size, dtype=config.member_dtype)

                mem_temp.mem_match_id[:] = cluster.mem_match_id
                mem_temp.id[:] = cluster.neighbors.id[memuse]
                mem_temp.z[:] = cluster.redshift
                mem_temp.ra[:] = cluster.neighbors.ra[memuse]
                mem_temp.dec[:] = cluster.neighbors.dec[memuse]
                mem_temp.r[:] = cluster.neighbors.r[memuse]
                mem_temp.p[:] = cluster.neighbors.p[memuse]
                mem_temp.pfree[:] = pfree_temp[memuse]
                mem_temp.pcol[:] = cluster.neighbors.pcol[memuse]
                mem_temp.theta_i[:] = cluster.neighbors.theta_i[memuse]
                mem_temp.theta_r[:] = cluster.neighbors.theta_r[memuse]
                mem_temp.refmag[:] = cluster.neighbors.refmag[memuse]
                mem_temp.refmag_err[:] = cluster.neighbors.refmag_err[memuse]
                if state['did_read_zreds']:
                    mem_temp.zred[:] = cluster.neighbors.zred[memuse]
                    mem_temp.zred_e[:] = cluster.neighbors.zred_e[memuse]
                mem_temp.chisq[:] = cluster.neighbors.chisq[memuse]
                mem_temp.ebv[:] = cluster.neighbors.ebv[memuse]
                mem_temp.mag[:, :] = cluster.neighbors.mag[memuse, :]
                mem_temp.mag_err[:, :] = cluster.neighbors.mag_err[memuse, :]

                members_list.append(mem_temp)

    if len(members_list) > 0:
        members = Catalog(np.concatenate([m._ndarray for m in members_list]))
    else:
        members = None

    if postprocess_fn is not None:
        cat, members = postprocess_fn(cat, members, state)
    else:
        use, = np.where(cat.Lambda >= min_lambda)
        cat = ClusterCatalog(cat._ndarray[use])
        if members is not None:
            a, b = esutil.numpy_util.match(cat.mem_match_id, members.mem_match_id)
            members = Catalog(members._ndarray[b])

    del state['gals']
    del state['bkg']
    del state['cbkg']
    del state['zredbkg']
    del state['zredstr']
    del state['depthstr']
    del state['zlambda_corr']
    del state['mask']
    del state['cosmo']
    gc.collect()

    return cat, members

def output_cluster_catalog(cat, members, config, filetype, savemembers=True, withversion=True, clobber=False, outbase=None):
    filename = config.redmapper_filename(filetype + '_catalog', withversion=withversion, outbase=outbase)

    if cat is None:
        logger.info(f"Warning: no catalog generated for {filename}")
        return filename

    logger.info(f"Writing catalog to file: {filename}")
    cat.to_fits_file(filename, clobber=clobber)

    if savemembers:
        if members is None:
            logger.info(f"Warning: no members generated for {filename}")
            return filename
        memfilename = config.redmapper_filename(filetype + '_catalog_members', withversion=withversion, outbase=outbase)
        members.to_fits_file(memfilename, clobber=clobber)

    return filename


