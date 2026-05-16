"""Class for calibrating the color-based red-sequence model.
"""
import os
import numpy as np
import fitsio
import time
import esutil
from scipy.optimize import least_squares

from ..logger import logger
from ..configuration import Configuration
from ..fitters import fit_med_z, fit_red_sequence, fit_red_sequence_off_diagonal, fit_correction, fit_error_bin
from ..redsequence import read_redsequence, redsequence_mstar, redsequence_zindex, plot_redsequence_diag, plot_redsequence_offdiags
from ..color_background import read_color_background, lookup_diagonal, lookup_offdiag, get_colrange
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry
from ..zred_color import compute_zreds
from ..utilities import make_nodes, cubic_spline_compute_y2, cubic_spline_interpolate, interpol, read_redgal_initial_colors, get_redgal_initial_color

def calibrate_red_sequence(config, galfile, rng=None, doRaise=True):
    """
    Run the red-sequence calibration.

    Parameters
    ----------
    config : `redmapper.Configuration`
        Configuration object
    galfile : `str`
        Galaxy file with the required fields
    rng : `np.random.RandomState`, optional
        Random number generator.
    doRaise : `bool`, optional
        Raise an error if background cannot be computed for any galaxies.
        Default is True.
    """
    if not isinstance(config, Configuration):
        config = Configuration(config)

    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    gals = GalaxyCatalog.from_galfile(galfile, zspec=config.centering_use_zspec)

    if config.calib_use_pcol:
        use, = np.where((gals.z > config.zrange[0]) &
                        (gals.z < config.zrange[1]) &
                        (gals.pcol > config.calib_pcut))
    else:
        use, = np.where((gals.z > config.zrange[0]) &
                        (gals.z < config.zrange[1]) &
                        (gals.p > config.calib_pcut))

    if use.size == 0:
        raise RuntimeError("No good galaxies in %s!" % (galfile))

    gals = gals[use]

    # Initialize parameters
    pars = _initialize_red_sequence_pars(config)

    # And a special subset of color galaxies
    if config.calib_use_pcol:
        coluse, = np.where(gals.pcol > config.calib_color_pcut)
    else:
        coluse, = np.where(gals.p > config.calib_color_pcut)

    colgals = gals[coluse]

    # And a placeholder zredstr which allows us to do stuff
    zredstr = read_redsequence(None, config=config)

    # And read the color background
    bkg = read_color_background(config.bkgfile_color)

    # And prepare for luptitude corrections
    if config.b[0] == 0.0:
        do_lupcorr = False
        bnmgy = None
        lupzp = None
    else:
        do_lupcorr = True
        bnmgy = config.b * 1e9
        lupzp = 22.5

    # Compute pivotmags
    _calc_pivotmags(config, colgals, pars, zredstr)

    # Compute median colors
    _calc_medcols(config, colgals, pars)

    # Compute diagonal parameters
    _calc_diagonal_pars(config, gals, pars, bkg, do_lupcorr, bnmgy, lupzp, doRaise=doRaise)

    # Compute off-diagonal parameters
    _calc_offdiagonal_pars(config, gals, pars, bkg, do_lupcorr, bnmgy, lupzp, doRaise=doRaise)

    # Compute volume factor
    _calc_volume_factor(config, pars, config.zrange[1])

    # Write out the parameter file
    save_red_sequence_pars(config, pars, config.parfile, clobber=False)

    # Compute zreds without corrections
    _calc_zreds(config, gals, config.parfile, do_correction=False)

    # Compute correction (mode1)
    _calc_corrections(config, gals, pars)

    # Compute correction (mode2)
    _calc_corrections(config, gals, pars, mode2=True)

    # And re-save the parameter file
    save_red_sequence_pars(config, pars, config.parfile, clobber=True)

    # Recompute zreds with corrections
    _calc_zreds(config, gals, config.parfile, do_correction=True)

    # And want to save galaxies and zreds
    zredfile = os.path.join(config.outpath, os.path.basename(galfile.rstrip('.fit') + '_zreds.fit'))
    gals.to_fits_file(zredfile)

    # Make diagnostic plots
    _make_diagnostic_plots(config, gals, pars, rng=rng)
    _make_red_sequence_evolution_plots(config, pars)
    _make_color_redshift_evolution_plots(config, pars)

def _initialize_red_sequence_pars(config):
    """
    Initialize the red-sequence parameters based on configuration.
    """
    pivotnodes = make_nodes(config.zrange, config.calib_pivotmag_nodesize)
    covmatnodes = make_nodes(config.zrange, config.calib_covmat_nodesize)
    corrnodes = make_nodes(config.zrange, config.calib_corr_nodesize)
    corrslopenodes = make_nodes(config.zrange, config.calib_corr_slope_nodesize)
    volnodes = make_nodes(config.zrange, 0.01)

    nmag = config.nmag
    ncol = nmag - 1

    dtype = [('pivotmag_z', 'f4', pivotnodes.size),
             ('pivotmag', 'f4', pivotnodes.size),
             ('minrefmag', 'f4', pivotnodes.size),
             ('maxrefmag', 'f4', pivotnodes.size),
             ('medcol', 'f4', (pivotnodes.size, ncol)),
             ('medcol_width', 'f4', (pivotnodes.size, ncol)),
             ('medcol_err_ratio', 'f4', (ncol, )),
             ('covmat_z', 'f4', covmatnodes.size),
             ('sigma', 'f4', (ncol, ncol, covmatnodes.size)),
             ('covmat_amp', 'f4', (ncol, ncol, covmatnodes.size)),
             ('covmat_slope', 'f4', (ncol, ncol, covmatnodes.size)),
             ('mag_err_ratio_int', 'f4', (nmag, )),
             ('mag_err_ratio_slope', 'f4', (nmag, )),
             ('mag_err_ratio_pivot', 'f4'),
             ('corr_z', 'f4', corrnodes.size),
             ('corr', 'f4', corrnodes.size),
             ('corr_slope_z', 'f4', corrslopenodes.size),
             ('corr_slope', 'f4', corrslopenodes.size),
             ('corr_r', 'f4', corrslopenodes.size),
             ('corr2', 'f4', corrnodes.size),
             ('corr2_slope', 'f4', corrslopenodes.size),
             ('corr2_r', 'f4', corrslopenodes.size),
             ('volume_factor_z', 'f4', volnodes.size),
             ('volume_factor', 'f4', volnodes.size)]

    node_dict = {}
    for j in range(ncol):
        ztag = 'z%02d' % (j)
        ctag = 'c%02d' % (j)
        zstag = 'zs%02d' % (j)
        stag = 'slope%02d' % (j)

        node_dict[ztag] = make_nodes(config.zrange, config.calib_color_nodesizes[j],
                                     maxnode=config.calib_color_maxnodes[j])
        node_dict[zstag] = make_nodes(config.zrange, config.calib_slope_nodesizes[j],
                                      maxnode=config.calib_color_maxnodes[j])

        dtype.extend([(ztag, 'f4', node_dict[ztag].size),
                      (ctag, 'f4', node_dict[ztag].size),
                      (zstag, 'f4', node_dict[zstag].size),
                      (stag, 'f4', node_dict[zstag].size)])

    pars = Catalog.zeros(1, dtype=dtype)
    pars = pars[0]

    pars.pivotmag_z = pivotnodes
    pars.covmat_z = covmatnodes
    pars.corr_z = corrnodes
    pars.corr_slope_z = corrslopenodes
    pars.volume_factor_z = volnodes
    pars.mag_err_ratio_pivot = config.calib_err_ratio_pivot

    for j in range(ncol):
        ztag = 'z%02d' % (j)
        zstag = 'zs%02d' % (j)
        pars[ztag] = node_dict[ztag]
        pars[zstag] = node_dict[zstag]

    return pars

def _compute_red_sequence_startvals(nodes, z, val, xval=None, err=None, median=False, fit=False, mincomp=3):
    """
    Compute the starting fit values using a simple algorithm.

    Must select one (and only one) of median=True (median fit) or
    fit=True (weighted mean fit).

    Parameters
    ----------
    nodes: `np.array`
       Float array of redshift nodes
    z: `np.array`
       Float array of redshifts
    val: `np.array`
       Float array of values to fit (e.g. refmag, color)
    xval: `np.array`, optional
       X-axis value for color-magnitude relation if fitting slope.
       Usually refmag.
       Default is None, which means not fitting a slope.
    err: `np.array`, optional
       Float array of error on val.  Not used if fitting median.
       Default is None.
    median: `bool`, optional
       Perform median fit.  Default is False.
    fit: `bool`, optional
       Perform weighted mean fit.  Default is False.
    """

    def _linfunc(p, x, y):
        return (p[1] + p[0] * x) - y

    if (not median and not fit) or (median and fit):
        raise RuntimeError("Must select one and only one of median and fit")

    if median:
        mvals = np.zeros(nodes.size)
        scvals = np.zeros(nodes.size)
    else:
        cvals = np.zeros(nodes.size)
        svals = np.zeros(nodes.size)

    if err is not None:
        if err.size != val.size:
            raise ValueError("val and err must be the same length")

        # default all to 0.1
        evals = np.zeros(nodes.size) + 0.1
    else:
        evals = None

    for i in range(nodes.size):
        if i == 0:
            zlo = nodes[0]
        else:
            zlo = (nodes[i - 1] + nodes[i]) / 2.
        if i == nodes.size - 1:
            zhi = nodes[i]
        else:
            zhi = (nodes[i] + nodes[i + 1]) / 2.

        u, = np.where((z > zlo) & (z < zhi))

        if u.size < mincomp:
            if i > 0:
                if median:
                    mvals[i] = mvals[i - 1]
                    scvals[i] = scvals[i - 1]
                else:
                    cvals[i] = cvals[i - 1]
                    svals[i] = svals[i - 1]

                if err is not None:
                    evals[i] = evals[i - 1]
        else:
            if median:
                mvals[i] = np.median(val[u])
                scvals[i] = np.median(np.abs(val[u] - mvals[i]))
            else:
                fit_res = least_squares(_linfunc, [0.0, 0.0], loss='soft_l1', args=(xval[u], val[u]))
                if not np.isfinite(fit_res.x).all():
                    cvals[i] = cvals[i - 1] if i > 0 else 0.0
                    svals[i] = svals[i - 1] if i > 0 else 0.0
                else:
                    cvals[i] = fit_res.x[1]
                    svals[i] = np.clip(fit_res.x[0], None, 0.0)

            if err is not None:
                evals[i] = np.median(err[u])

    # Final safety check for nans
    if median:
        mvals[~np.isfinite(mvals)] = 0.0
        scvals[~np.isfinite(scvals)] = 0.1
        return mvals, scvals
    else:
        cvals[~np.isfinite(cvals)] = 0.0
        svals[~np.isfinite(svals)] = 0.0
        if evals is not None:
            evals[~np.isfinite(evals)] = 0.1
        return cvals, svals, evals

def _compute_single_lupcorr(config, pars, j, cvals, svals, gals, dmags, mags, lups, mind, sign, lupzp, bnmgy):
    """
    Compute the luptitude correction for a single color

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    j: `int`
       Color index
    cvals: `np.array`
       Float array of spline values for color at pivotmag
    svals: `np.array`
       Float array of slope values
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog being fit
    dmags: `np.array`
       Float array of refmag - pivotmag
    mags: `np.array`
       2d Float array of true (model)  magnitudes
    lups: `np.array`
       2d Float array of true (model) luptitudes
    mind: `int`
       magnitude index, currently being worked on.
    sign: `int`, -1 or 1
       Sign of color; -1 if band is redder than ref_ind,
       +1 if band is bluer than ref_ind
    lupzp: `float`
       Luptitude zero point
    bnmgy: `np.array`
       b values in nanomaggies

    Returns
    -------
    lupcorr: `np.array`
       Float array of luptitude color corrections
    """
    ztag = 'z%02d' % (j)
    zstag = 'zs%02d' % (j)

    y2_c = cubic_spline_compute_y2(pars._ndarray[ztag], cvals)
    cv = cubic_spline_interpolate(gals.z, pars._ndarray[ztag], cvals, y2_c)
    y2_s = cubic_spline_compute_y2(pars._ndarray[zstag], svals)
    sv = cubic_spline_interpolate(gals.z, pars._ndarray[zstag], svals, y2_s)

    mags[:, mind] = mags[:, mind + sign] + sign * (cv + sv * dmags)

    flux = 10.**((mags[:, mind] - lupzp) / (-2.5))
    lups[:, mind] = 2.5 * np.log10(1.0 / config.b[mind]) - np.arcsinh(0.5 * flux / bnmgy[mind]) / (0.4 * np.log(10.0))

    magcol = mags[:, j] - mags[:, j + 1]
    lupcol = lups[:, j] - lups[:, j + 1]

    lupcorr = lupcol - magcol

    return lupcorr

def _calc_pivotmags(config, gals, pars, zredstr):
    """
    Calculate the pivot magnitude parameters.

    These are put into pars.pivotmag, pars.maxrefmag, and
    pars.minrefmag

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog with fields required for fit.
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    zredstr: `dict`
       Placeholder red-sequence model
    """

    logger.info("Calculating pivot magnitudes...")

    # With binning, approximate the positions for starting the fit
    pivmags = np.zeros_like(pars.pivotmag_z)

    for i in range(pivmags.size):
        pivmags, _ = _compute_red_sequence_startvals(pars.pivotmag_z, gals.z, gals.refmag, median=True)

    pivmags = fit_med_z(pars.pivotmag_z, gals.z, gals.refmag, pivmags)

    pars.pivotmag = pivmags

    # and min and max...
    pars.minrefmag = redsequence_mstar(zredstr, pars.pivotmag_z) - 2.5 * np.log10(30.0)
    lval_min = np.clip(config.lval_reference - 0.1, 0.001, None)
    pars.maxrefmag = redsequence_mstar(zredstr, pars.pivotmag_z) - 2.5 * np.log10(lval_min)

def _calc_medcols(config, gals, pars):
    """
    Calculate the median color spline parameters.

    Sets pars.medcol, pars.medcol_width

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog with fields required for fit.
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    """

    logger.info("Calculating median colors...")

    ncol = config.nmag - 1

    galcolor = gals.galcol

    for j in range(ncol):
        col = galcolor[:, j]

        # get the start values
        mvals, scvals = _compute_red_sequence_startvals(pars.pivotmag_z, gals.z, col, median=True)

        # compute the median
        mvals = fit_med_z(pars.pivotmag_z, gals.z, col, mvals)

        # and the scatter
        y2 = cubic_spline_compute_y2(pars.pivotmag_z, mvals)
        med = cubic_spline_interpolate(gals.z, pars.pivotmag_z, mvals, y2)
        scvals = fit_med_z(pars.pivotmag_z, gals.z, np.abs(col - med), scvals, min_val=0.01)

        pars.medcol[:, j] = mvals
        pars.medcol_width[:, j] = 1.4826 * scvals

        if (config.calib_fit_err_ratio):
            # Compute overall pulls...
            y2 = cubic_spline_compute_y2(pars.pivotmag_z, pars.medcol[:, j])
            model = cubic_spline_interpolate(gals.z, pars.pivotmag_z, pars.medcol[:, j], y2)
            delta = gals.galcol[:, j] - model
            col_err = gals.galcol_err[:, j]

            # Step over values of r, look for closest to pulls of 1.0
            # change the upper limit from 10 to 50
            err_ratios = np.arange(0.5, 50.0, 0.1)
            sigma_pulls = np.zeros_like(err_ratios)
            # Assume first bin median width is close to sig_int
            sig_int = pars.medcol_width[0, j]

            # Only use galaxies from the first 3 redshift bins

            for i in range(err_ratios.size):
                err = np.sqrt((err_ratios[i]*col_err)**2. + sig_int**2.)
                pulls = delta/err

                sigma_pulls[i] = 1.4826*np.median(np.abs(pulls - np.median(pulls)))

            argmin = np.argmin(np.abs(sigma_pulls - 1.0))
            pars.medcol_err_ratio[j] = err_ratios[argmin]
        else:
            pars.medcol_err_ratio[j] = 1.0

def _calc_diagonal_pars(config, gals, pars, bkg, do_lupcorr, bnmgy, lupzp, doRaise=True):
    """
    Calculate the model parameters and diagonal elements of the covariance
    matrix (one color at a time).

    Sets pars.sigma, pars.covmat_amp, pars.cXX, pars.slopeXX

    Parameters
    ----------
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog with fields required for fit.
    doRaise: `bool`, optional
       Raise if there's a problem with the background?  Default is True.
    """

    # The main routine to compute the red sequence on the diagonal

    ncol = config.nmag - 1

    galcolor = gals.galcol
    galcolor_err = gals.galcol_err

    # compute the pivot mags
    y2 = cubic_spline_compute_y2(pars.pivotmag_z, pars.pivotmag)
    pivotmags = cubic_spline_interpolate(gals.z, pars.pivotmag_z, pars.pivotmag, y2)

    # And set the right probabilities
    if config.calib_use_pcol:
        probs = gals.pcol
    else:
        probs = gals.p

    # Figure out the order of the colors for luptitude corrections
    mags = np.zeros((gals.size, config.nmag))

    col_indices = np.zeros(ncol, dtype=np.int32)
    sign_indices = np.zeros(ncol, dtype=np.int32)
    mind_indices = np.zeros(ncol, dtype=np.int32)

    c = 0
    for j in range(config.ref_ind + 1, config.nmag):
        col_indices[c] = j - 1
        sign_indices[c] = -1
        mind_indices[c] = j
        c += 1
    for j in range(config.ref_ind, 0, -1):
        col_indices[c] = j - 1
        sign_indices[c] = 1
        mind_indices[c] = j - 1
        c += 1

    if do_lupcorr:
        lups = np.zeros_like(mags)

        mags[:, config.ref_ind] = gals.mag[:, config.ref_ind]
        flux = 10.**((mags[:, config.ref_ind] - lupzp) / (-2.5))
        lups[:, config.ref_ind] = 2.5 * np.log10(1.0 / config.b[config.ref_ind]) - np.arcsinh(0.5 * flux / bnmgy[config.ref_ind]) / (0.4 * np.log(10.0))

    # One color at a time along the diagonal
    for c in range(ncol):
        starttime = time.time()

        # The order is given by col_indices, which ensures that we work from the
        # reference mag outward
        j = col_indices[c]
        sign = sign_indices[c]
        mind = mind_indices[c]

        logger.info("Working on diagonal for color %d" % (j))

        col = galcolor[:, j]
        col_err = galcolor_err[:, j]
        mag_err = gals.mag_err[:, j: j + 2].copy()

        ztag = 'z%02d' % (j)
        ctag = 'c%02d' % (j)
        zstag = 'zs%02d' % (j)
        stag = 'slope%02d' % (j)

        # Need to go through the _ndarray because ztag and zstag are strings
        cvals = np.zeros(pars._ndarray[ztag].size)
        svals = np.zeros(pars._ndarray[zstag].size)
        photo_err = np.zeros_like(cvals)

        # Calculate median truncation
        y2_medcol = cubic_spline_compute_y2(pars.pivotmag_z, pars.medcol[:, j])
        med = cubic_spline_interpolate(gals.z, pars.pivotmag_z, pars.medcol[:, j], y2_medcol)
        y2_medcol_width = cubic_spline_compute_y2(pars.pivotmag_z, pars.medcol_width[:, j])
        sc = cubic_spline_interpolate(gals.z, pars.pivotmag_z, pars.medcol_width[:, j], y2_medcol_width)

        # What is the maximum scatter in each node?
        # This is based on the median fit, which does not include photometric
        # error, and should always be larger.  This helps regularize the edges
        # where things otherwise can run away.
        scatter_max = cubic_spline_interpolate(pars.covmat_z, pars.pivotmag_z, pars.medcol_width[:, j], y2_medcol_width)

        # Initial guess for scvals should be halfway between 0.01 and scatter_max
        scvals = (scatter_max - 0.01) / 2.0 + 0.01

        u, = np.where((galcolor[:, j] > (med - config.calib_color_nsig * sc)) &
                      (galcolor[:, j] < (med + config.calib_color_nsig * sc)))
        trunc = config.calib_color_nsig * sc[u]

        dmags = gals.refmag - pivotmags

        # And the starting values...
        # Note that this returns the slope values (svals) at the nodes from the cvals
        # but these might not be the same nodes, so we have to approximate
        cvals_temp, svals_temp, _ = _compute_red_sequence_startvals(pars._ndarray[ztag],
                                                            gals.z[u], col[u],
                                                            xval=dmags[u],
                                                            fit=True, mincomp=5)
        cvals[:] = cvals_temp
        inds = np.searchsorted(pars._ndarray[ztag],
                               pars._ndarray[zstag])
        svals[:] = svals_temp[inds]

        # And do the luptitude correction if necessary.
        if do_lupcorr:
            lupcorr = _compute_single_lupcorr(config, pars, j, cvals, svals, gals, dmags, mags, lups, mind, sign, lupzp, bnmgy)
        else:
            lupcorr = np.zeros(gals.size)

        dmags_err_ratio = gals.refmag - config.calib_err_ratio_pivot
        if config.calib_fit_err_ratio:
            # When we are not doing the first color, we have a mag error modification
            # for one magnitude
            if c > 0:
                err_ratios = pars.mag_err_ratio_int[mind + sign] + pars.mag_err_ratio_slope[mind + sign]*dmags_err_ratio
                if sign == 1:
                    # Apply to the redward mag_err
                    mag_err[:, 1] *= err_ratios
                    # This is the index of color to fit the error ratio
                    fit_err_ratio_ind = [0]
                else:
                    # Apply to the blueward mag_err
                    mag_err[:, 0] *= err_ratios
                    fit_err_ratio_ind = [1]

                # The fit start values from the neighboring color
                err_ratio_pars = [pars.mag_err_ratio_int[mind + sign],
                                  pars.mag_err_ratio_slope[mind + sign]]
            else:
                fit_err_ratio_ind = [0, 1]

                # The fit start values from the median fit
                err_ratio_pars = [pars.medcol_err_ratio[j], 0.0]
        else:
            err_ratio_pars = None
            fit_err_ratio_ind = []

        # We fit in stages: first the mean, then the slope, then the scatter,
        # and finally all three
        lupcorrs_u = lupcorr[u]
        bkgs_u = lookup_diagonal(bkg, j, col[u], gals.refmag[u], doRaise=doRaise)

        # fit the mean
        cvals_list = fit_red_sequence(pars._ndarray[ztag],
                                      gals.z[u], col[u], mag_err[u, :],
                                      cvals, svals, scvals,
                                      fit_mean=True,
                                      dmags=dmags[u],
                                      trunc=trunc,
                                      slope_nodes=pars._ndarray[zstag],
                                      scatter_nodes=pars.covmat_z,
                                      lupcorrs=lupcorrs_u,
                                      probs=probs[u],
                                      bkgs=bkgs_u,
                                      scatter_max=scatter_max, use_scatter_prior=True,
                                      err_ratio_pars=err_ratio_pars, fit_err_ratio_ind=fit_err_ratio_ind,
                                      dmags_err_ratio=dmags_err_ratio[u])
        cvals = cvals_list[0]

        # Update the lupcorr...
        if do_lupcorr:
            lupcorrs_u = _compute_single_lupcorr(config, pars, j, cvals, svals, gals, dmags, mags, lups, mind, sign, lupzp, bnmgy)[u]

        # fit the slope
        svals_list = fit_red_sequence(pars._ndarray[ztag],
                                      gals.z[u], col[u], mag_err[u, :],
                                      cvals, svals, scvals,
                                      fit_slope=True,
                                      dmags=dmags[u],
                                      trunc=trunc,
                                      slope_nodes=pars._ndarray[zstag],
                                      scatter_nodes=pars.covmat_z,
                                      lupcorrs=lupcorrs_u,
                                      probs=probs[u],
                                      bkgs=bkgs_u,
                                      scatter_max=scatter_max, use_scatter_prior=True,
                                      err_ratio_pars=err_ratio_pars, fit_err_ratio_ind=fit_err_ratio_ind,
                                      dmags_err_ratio=dmags_err_ratio[u])
        svals = svals_list[0]

        # fit the scatter
        scvals_list = fit_red_sequence(pars._ndarray[ztag],
                                       gals.z[u], col[u], mag_err[u, :],
                                       cvals, svals, scvals,
                                       fit_scatter=True,
                                       dmags=dmags[u],
                                       trunc=trunc,
                                       slope_nodes=pars._ndarray[zstag],
                                       scatter_nodes=pars.covmat_z,
                                       lupcorrs=lupcorrs_u,
                                       probs=probs[u],
                                       bkgs=bkgs_u,
                                       scatter_max=scatter_max, use_scatter_prior=True,
                                       err_ratio_pars=err_ratio_pars, fit_err_ratio_ind=fit_err_ratio_ind,
                                       dmags_err_ratio=dmags_err_ratio[u])
        scvals = scvals_list[0]

        if config.calib_fit_err_ratio:
            err_ratios = scvals[-2: ]
            scvals = scvals[: -2]

        # fit combined
        cvals, svals, scvals = fit_red_sequence(pars._ndarray[ztag],
                                                gals.z[u], col[u], mag_err[u, :],
                                                cvals, svals, scvals,
                                                fit_mean=True, fit_slope=True, fit_scatter=True,
                                                dmags=dmags[u],
                                                trunc=trunc,
                                                slope_nodes=pars._ndarray[zstag],
                                                scatter_nodes=pars.covmat_z,
                                                lupcorrs=lupcorrs_u,
                                                probs=probs[u],
                                                bkgs=bkgs_u,
                                                scatter_max=scatter_max, use_scatter_prior=True,
                                                err_ratio_pars=err_ratio_pars, fit_err_ratio_ind=fit_err_ratio_ind,
                                                dmags_err_ratio=dmags_err_ratio[u])

        if config.calib_fit_err_ratio:
            err_ratio_int_fit = scvals[-2]
            err_ratio_slope_fit = scvals[-1]
            scvals = scvals[: -2]

            y2_c = cubic_spline_compute_y2(pars._ndarray[ztag], cvals)
            gmean = cubic_spline_interpolate(gals.z, pars._ndarray[ztag], cvals, y2_c)
            y2_s = cubic_spline_compute_y2(pars._ndarray[zstag], svals)
            gslope = cubic_spline_interpolate(gals.z, pars._ndarray[zstag], svals, y2_s)
            y2_sc = cubic_spline_compute_y2(pars.covmat_z, scvals)
            gsig = np.clip(cubic_spline_interpolate(gals.z, pars.covmat_z, scvals, y2_sc), 0.001, None)

            delta_col = gals.galcol[:, j] - (gmean + gslope*dmags)

            err_0 = gals.mag_err[:, j].copy()
            err_1 = gals.mag_err[:, j + 1].copy()
            if 0 not in fit_err_ratio_ind:
                # We are not fitting 0, so we already have the parameters.
                err_0 *= (pars.mag_err_ratio_int[j] +
                          pars.mag_err_ratio_slope[j]*dmags_err_ratio)
            if 1 not in fit_err_ratio_ind:
                # We are not fitting 1, so we already have the parameters.
                err_1 *= (pars.mag_err_ratio_int[j + 1] +
                          pars.mag_err_ratio_slope[j + 1]*dmags_err_ratio)

            ebinpars = fit_error_bin(delta_col,
                                     dmags_err_ratio,
                                     err_0,
                                     err_1,
                                     gsig**2.,
                                     [1.0, 0.0],
                                     scale_indices=fit_err_ratio_ind)

            err_ratio_int = ebinpars[0]
            err_ratio_slope = ebinpars[1]

            if c == 0:
                pars.mag_err_ratio_int[j] = err_ratio_int
                pars.mag_err_ratio_slope[j] = err_ratio_slope
                pars.mag_err_ratio_int[j + 1] = err_ratio_int
                pars.mag_err_ratio_slope[j + 1] = err_ratio_slope
                logger.info('Mag %d/%d error ratio = %.3f + %.3f*(refmag - %.2f)' %
                                        (j, j + 1, err_ratio_int, err_ratio_slope, pars.mag_err_ratio_pivot))
            else:
                pars.mag_err_ratio_int[mind] = err_ratio_int
                pars.mag_err_ratio_slope[mind] = err_ratio_slope
                logger.info('Mag %d error ratio = %.3f + %.3f*(refmag - %.2f)' %
                                        (j, err_ratio_int, err_ratio_slope, pars.mag_err_ratio_pivot))
        else:
            pars.mag_err_ratio_int[j] = 1.0
            pars.mag_err_ratio_slope[j] = 0.0
            pars.mag_err_ratio_int[j + 1] = 1.0
            pars.mag_err_ratio_slope[j + 1] = 0.0

        # And record in the parameters
        pars[ctag] = cvals
        pars[stag] = svals
        pars.sigma[j, j, :] = scvals
        pars.covmat_amp[j, j, :] = scvals**2.

        # And print the time taken
        logger.info('Done in %.2f seconds.' % (time.time() - starttime))

def _calc_offdiagonal_pars(config, gals, pars, bkg, do_lupcorr, bnmgy, lupzp, doRaise=True):
    """
    Set the off-diagonal elements of the covariance matrix.

    These are just set to config.calib_covmat_constant

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog with fields required for fit.
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    bkg: `dict`
       Color background data
    do_lupcorr: `bool`
       Do luptitude corrections
    bnmgy: `np.array`
       b values in nanomaggies
    lupzp: `float`
       Luptitude zero point
    doRaise: `bool`, optional
       Raise if there's a problem with the background?  Default is True.
    """

    ncol = config.nmag - 1

    for j in range(ncol):
        for k in range(j + 1, ncol):
            pars.sigma[j, k, :] = config.calib_covmat_constant
            pars.sigma[k, j, :] = pars.sigma[j, k, :]

            pars.covmat_amp[j, k, :] = config.calib_covmat_constant * pars.sigma[j, j, :] * pars.sigma[k, k, :]
            pars.covmat_amp[k, j, :] = pars.covmat_amp[j, k, :]

def _calc_volume_factor(config, pars, zref):
    """
    Calculate the volume factor (delta-comoving volume in redshift steps)

    Sets pars.volume_factor

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    zref: `float`
       Highest redshift in the model (for reference)
    """

    dz = 0.01

    pars.volume_factor = ((config.cosmo.Dl(0.0, zref + dz) / (1. + (zref + dz)) -
                               config.cosmo.Dl(0.0, zref) / (1. + zref)) /
                               (config.cosmo.Dl(0.0, pars.volume_factor_z + dz) / (1. + (pars.volume_factor_z + dz)) -
                                config.cosmo.Dl(0.0, pars.volume_factor_z) / (1. + pars.volume_factor_z)))



def save_red_sequence_pars(config, pars, filename, clobber=False):
    """
    Save the parameters to a fits file.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    filename: `str`
       Filename to save to.
    clobber: `bool`, optional
       Clobber any existing file?  Default is False.
    """

    if config.calib_redgal_template is not None:
        rg = read_redgal_initial_colors(config.calib_redgal_template)
        zmax = rg['z'].max()
    else:
        zmax = None

    hdr = fitsio.FITSHDR()
    hdr['NCOL'] = config.nmag - 1
    hdr['MSTARSUR'] = config.mstar_survey
    hdr['MSTARBAN'] = config.mstar_band
    hdr['LIMMAG'] = config.limmag_catalog
    # Saved with larger cushion that seems to work well
    hdr['ZRANGE0'] = np.clip(config.zrange[0] - 0.1, 0.01, None)
    hdr['ZRANGE1'] = np.clip(config.zrange[1] + 0.25, None, zmax)
    hdr['ALPHA'] = config.calib_lumfunc_alpha
    hdr['ZBINFINE'] = config.zredc_binsize_fine
    hdr['ZBINCOAR'] = config.zredc_binsize_coarse
    hdr['LOWZMODE'] = 0
    hdr['REF_IND'] = config.ref_ind
    hdr['BANDS'] = ','.join(config.bands)
    if config.calib_redgal_template is not None:
        hdr['TEMPLATE'] = config.calib_redgal_template
    # Only save the b values if they're > 0 (that means we have
    # luptitudes).
    if config.b[0] > 0.0:
        for j, b in enumerate(config.b):
            hdr['BVALUE%d' % (j + 1)] = b

    pars.to_fits_file(filename, header=hdr, clobber=clobber)

def _calc_zreds(config, gals, parfile, do_correction=True):
    """
    Calculate the zreds for a set of galaxies, using the newly fit model.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog being fit
    parfile: `str`
       Parameter file path
    do_corrections: `bool`, optional
       Do redshift afterburner corrections?  Default is True.
    """

    # This is temporary
    zredstr = read_redsequence(parfile)

    gals.add_zred_fields(config.zred_nsamp)

    starttime = time.time()
    compute_zreds(zredstr, gals, do_correction=do_correction)

    logger.info('Computed zreds in %.2f seconds.' % (time.time() - starttime))

def _calc_corrections(config, gals, pars, mode2=False):
    """
    Calculate zred afterburner correction parameters.

    Sets pars.corr, pars.corr_slope, pars.corr_r or
    pars.corr2, pars.corr2_slope, pars.corr2_r

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog being fit.  Must contain zred_uncorr information.
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    mode2: `bool`, optional
       Default is False.  When False, corrections are computed such that
       <zred|ztrue> is unbiased.  When True, corrections are computed
       such that <ztrue|zred> is unbiased.
    """

    # p or pcol
    if config.calib_use_pcol:
        probs = gals.pcol
    else:
        probs = gals.p

    # Set a threshold removing 5% worst lkhd outliers
    st = np.argsort(gals.lkhd)
    thresh = gals.lkhd[st[int(0.05 * gals.size)]]

    # This is an arbitrary 2sigma cut...
    guse, = np.where((gals.lkhd > thresh) &
                     (np.abs(gals.z - gals.zred) < 2. * gals.zred_e))

    y2 = cubic_spline_compute_y2(pars.pivotmag_z, pars.pivotmag)
    pivotmags = cubic_spline_interpolate(gals.z, pars.pivotmag_z, pars.pivotmag, y2)

    w = 1. / (np.exp((thresh - gals.lkhd[guse]) / 0.2) + 1.0)

    # The offset cvals
    cvals = np.zeros(pars.corr_z.size)
    # The slope svals
    svals = np.zeros(pars.corr_slope_z.size)
    # And the r value to be multiplied by error
    rvals = np.ones(pars.corr_slope_z.size)
    # And the background vals
    bkg_cvals = np.zeros(pars.corr_slope_z.size)

    cvals[:], _ = _compute_red_sequence_startvals(pars.corr_z, gals.z, gals.z - gals.zred, median=True)

    # Initial guess for bkg_cvals is trickier and not generalizable (sadly)
    for i in range(pars.corr_slope_z.size):
        if i == 0:
            zlo = pars.corr_slope_z[0]
        else:
            zlo = (pars.corr_slope_z[i - 1] + pars.corr_slope_z[i]) / 2.
        if i == (pars.corr_slope_z.size - 1):
            zhi = pars.corr_slope_z[i]
        else:
            zhi = (pars.corr_slope_z[i] + pars.corr_slope_z[i + 1]) / 2.

        if mode2:
            u, = np.where((gals.zred[guse] > zlo) & (gals.zred[guse] < zhi))
        else:
            u, = np.where((gals.z[guse] > zlo) & (gals.z[guse] < zhi))

        if u.size < 3:
            if i > 0:
                bkg_cvals[i] = bkg_cvals[i - 1]
        else:
            st = np.argsort(probs[guse[u]])
            uu = u[st[0:u.size // 2]]
            bkg_cvals[i] = np.std(gals.z[guse[uu]] - gals.zred[guse[uu]])**2.

    if mode2:
        logger.info("Fitting zred2 corrections...")
        z = gals.zred
    else:
        logger.info("Fitting zred corrections...")
        z = gals.z

    # first fit the mean
    cvals_list = fit_correction(pars.corr_z, z[guse], gals.z[guse] - gals.zred[guse], gals.zred_e[guse],
                                cvals, svals, rvals, bkg_cvals,
                                slope_nodes=pars.corr_slope_z,
                                probs=np.clip(probs[guse], None, 0.99),
                                dmags=gals.refmag[guse] - pivotmags[guse],
                                ws=w, fit_mean=True)
    cvals = cvals_list[0]
    # fit the slope (if desired)
    if not config.calib_corr_nocorrslope:
        svals_list = fit_correction(pars.corr_z, z[guse], gals.z[guse] - gals.zred[guse], gals.zred_e[guse],
                                    cvals, svals, rvals, bkg_cvals,
                                    slope_nodes=pars.corr_slope_z,
                                    probs=np.clip(probs[guse], None, 0.99),
                                    dmags=gals.refmag[guse] - pivotmags[guse],
                                    ws=w, fit_slope=True)
        svals = svals_list[0]
    # Fit r
    rvals_list = fit_correction(pars.corr_z, z[guse], gals.z[guse] - gals.zred[guse], gals.zred_e[guse],
                                cvals, svals, rvals, bkg_cvals,
                                slope_nodes=pars.corr_slope_z,
                                probs=np.clip(probs[guse], None, 0.99),
                                dmags=gals.refmag[guse] - pivotmags[guse],
                                ws=w, fit_r=True)
    rvals = rvals_list[0]
    # Fit bkg
    bkg_cvals_list = fit_correction(pars.corr_z, z[guse], gals.z[guse] - gals.zred[guse], gals.zred_e[guse],
                                    cvals, svals, rvals, bkg_cvals,
                                    slope_nodes=pars.corr_slope_z,
                                    probs=np.clip(probs[guse], None, 0.99),
                                    dmags=gals.refmag[guse] - pivotmags[guse],
                                    ws=w, fit_bkg=True)
    bkg_cvals = bkg_cvals_list[0]

    # Combined fit
    if not config.calib_corr_nocorrslope:
        cvals, svals, rvals, bkg_cvals = fit_correction(pars.corr_z, z[guse], gals.z[guse] - gals.zred[guse], gals.zred_e[guse],
                                                        cvals, svals, rvals, bkg_cvals,
                                                        slope_nodes=pars.corr_slope_z,
                                                        probs=np.clip(probs[guse], None, 0.99),
                                                        dmags=gals.refmag[guse] - pivotmags[guse],
                                                        ws=w, fit_mean=True, fit_slope=True, fit_r=True, fit_bkg=True)
    else:
        cvals, rvals, bkg_cvals = fit_correction(pars.corr_z, z[guse], gals.z[guse] - gals.zred[guse], gals.zred_e[guse],
                                                 cvals, svals, rvals, bkg_cvals,
                                                 slope_nodes=pars.corr_slope_z,
                                                 probs=np.clip(probs[guse], None, 0.99),
                                                 dmags=gals.refmag[guse] - pivotmags[guse],
                                                 ws=w, fit_mean=True, fit_r=True, fit_bkg=True)

    # And record the values
    if mode2:
        pars.corr2 = cvals
        pars.corr2_slope = svals
        pars.corr2_r = rvals
    else:
        pars.corr = cvals
        pars.corr_slope = svals
        pars.corr_r = rvals

def _make_diagnostic_plots(config, gals, pars, rng):
    """
    Make diagnostic plots.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    gals: `redmapper.GalaxyCatalog`
       Galaxy catalog being fit.  Must contain zred information.
    pars: `redmapper.Catalog` (Row)
       Parameter catalog
    rng: `np.random.RandomState`
       Random number generator
    """

    import matplotlib.pyplot as plt

    # what plots do we want?
    # We want to split this out into different modules?

    # For each color, plot
    #  Color(z)
    #  Slope(z)
    #  scatter(z)
    # And a combined
    #  All off-diagonal r value plots

    zredstr = read_redsequence(config.parfile, zbinsize=0.005)

    for j in range(config.nmag - 1):
        fig = plt.figure(figsize=(10, 6))
        fig.clf()

        plot_redsequence_diag(zredstr, fig, j, config.bands)
        fig.savefig(os.path.join(config.outpath, config.plotpath,
                                 '%s_%s-%s.png' % (config.outbase,
                                                   config.bands[j], config.bands[j + 1])))
        plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    fig.clf()
    plot_redsequence_offdiags(zredstr, fig, config.bands)
    fig.savefig(os.path.join(config.outpath, config.plotpath,
                             '%s_offdiags.png' % (config.outbase)))

    # And two panel plot with
    #  left panel is offset, scatter, outliers as f(z)
    #  Right panel is zred vs z (whichever)
    # We need to do this for both zred and zred2.

    zbinsize = 0.02
    pcut = 0.9
    ntrial = 1000

    mlim = redsequence_mstar(zredstr, gals.zred) - 2.5 * np.log10(0.2)

    use, = np.where((gals.p > pcut) &
                    (gals.refmag < mlim) &
                    (gals.zred < 2.0))

    ugals = gals[use]

    zbins = np.arange(config.zrange[0], config.zrange[1], zbinsize)

    dtype = [('ztrue', 'f4'),
             ('zuse', 'f4'),
             ('delta', 'f4'),
             ('delta_err', 'f4'),
             ('delta_std', 'f4'),
             ('zuse_e', 'f4'),
             ('f_out', 'f4')]


    # There are two modes to plot
    for mode in range(2):
        if mode == 0:
            zuse = ugals.z
            dzuse = ugals.zred - ugals.z
            zuse_e = ugals.zred_e
            xlabel = r'$z_{\mathrm{true}}$'
            ylabel_left = r'$z_{\mathrm{red}} - z_{\mathrm{true}}$'
            ylabel_right = r'$z_{\mathrm{red}}$'
            xcol = 'ztrue'
            modename = 'zred'
        else:
            zuse = ugals.zred2
            dzuse = ugals.z - ugals.zred2
            zuse_e = ugals.zred2_e
            xlabel = r'$z_{\mathrm{red2}}$'
            ylabel_left = r'$z_{\mathrm{true}} - z_{\mathrm{red2}}$'
            ylabel_right = r'$z_{\mathrm{true}}$'
            xcol = 'zuse'
            modename = 'zred2'

        zstr = np.zeros(zbins.size, dtype=dtype)

        for i, z in enumerate(zbins):
            # Get all the galaxies in the bin
            u1, = np.where((zuse >= z) & (zuse < (z + zbinsize)))

            if u1.size < 3:
                logger.info('Warning: not enough galaxies in zbin: %.2f, %.2f' % (z, z + zbinsize))
                continue

            med = np.median(dzuse[u1])
            gsigma = 1.4826 * np.abs(dzuse[u1] - med) / zuse_e[u1]

            u2, = np.where(np.abs(gsigma) < 3.0)
            if u2.size < 3:
                logger.info('Warning: not enough galaxies in zbin: %.2f, %.2f' % (z, z + zbinsize))

            use_inner = u1[u2]

            zstr['ztrue'][i] = np.median(ugals.z[use_inner])
            zstr['zuse'][i] = np.median(zuse[use_inner])
            zstr['delta'][i] = np.median(dzuse[use_inner])

            barr = np.zeros(ntrial)
            for t in range(ntrial):
                r = rng.choice(dzuse[use_inner], size=use_inner.size, replace=True)
                barr[t] = np.median(r)

            # Error on median as determined from bootstrap resampling
            zstr['delta_err'][i] = np.std(barr)

            # The typical error
            zstr['delta_std'][i] = 1.4826 * np.median(np.abs(dzuse[use_inner] - zstr['delta'][i]))

            # And outliers ...
            u4, = np.where(np.abs(dzuse[u1] - zstr['delta'][i]) > 4.0 * zstr['delta_std'][i])
            zstr['f_out'][i] = float(u4.size) / float(u1.size)

            zstr['zuse_e'][i] = np.median(zuse_e[use_inner])

        # Cut out bins that didn't get a fit
        cut, = np.where(zstr['ztrue'] > 0.0)
        zstr = zstr[cut]

        # Now we can make the plots
        fig = plt.figure(figsize=(10, 6))
        fig.clf()

        # Left panel is offset, scatter, etc.
        ax = fig.add_subplot(121)
        ax.errorbar(zstr[xcol], zstr['delta'], yerr=zstr['delta_err'], fmt='k^')
        ax.plot(config.zrange, [0.0, 0.0], 'k:')
        ax.plot(zstr[xcol], zstr['delta_std'], 'r-')
        ax.plot(zstr[xcol], zstr['zuse_e'], 'b-')
        ax.plot(zstr[xcol], zstr['f_out'], 'm-')
        ax.set_xlim(config.zrange)
        ax.set_ylim(-0.05, 0.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_left)

        ax = fig.add_subplot(122)
        if mode == 0:
            ax.hexbin(ugals.z, ugals.zred, bins='log', extent=[config.zrange[0], config.zrange[1], config.zrange[0], config.zrange[1]])
        else:
            ax.hexbin(ugals.zred2, ugals.z, bins='log', extent=[config.zrange[0], config.zrange[1], config.zrange[0], config.zrange[1]])
        ax.plot(config.zrange, config.zrange, 'r--')
        ax.set_xlim(config.zrange)
        ax.set_ylim(config.zrange)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_right)

        fig.tight_layout()
        fig.savefig(os.path.join(config.outpath, config.plotpath,
                                 '%s_%s_plots.png' % (config.outbase, modename)))

        plt.close(fig)

    if config.calib_fit_err_ratio:
        # We want to plot the error ratio values

        fig = plt.figure(figsize=(8, 6))
        fig.clf()

        ax = fig.add_subplot(111)
        ax.plot(np.arange(config.nmag), zredstr['mag_err_ratio_intercept'], 'r.')
        ax.plot(np.array([-0.5, config.nmag + 0.5]), [1.0, 1.0], 'k--')
        ax.set_xlim(-0.5, config.nmag - 0.5)
        ax.set_ylim(0.0, np.max(zredstr['mag_err_ratio_intercept']) + 1.0)
        ax.set_xlabel('Magnitude Index')
        ax.set_ylabel('Error Ratio')

        fig.tight_layout()
        fig.savefig(os.path.join(config.outpath, config.plotpath,
                                 '%s_err_ratios.png' % (config.outbase)))
        plt.close(fig)

    # Always make error plots; only do modified version if we fit it.

    # And we make one pull plot per color
    zindex = redsequence_zindex(zredstr, gals.zred)
    for j in range(config.nmag - 1):
        # Make the raw plot (with intrinsic error)

        delta = gals.galcol[:, j] - (zredstr['c'][zindex, j] + zredstr['slope'][zindex, j]*(gals.refmag - zredstr['pivotmag'][zindex]))
        delta_err = np.sqrt(gals.galcol_err[:, j]**2. + zredstr['sigma'][j, j, zindex]**2.)
        pulls = delta/delta_err

        mags = gals.mag[:, j + 1]
        st = np.argsort(mags)
        magrange = [mags[st[int(0.01*mags.size)]], mags[st[int(0.99*mags.size)]]]

        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(111)

        _plot_pulls(ax, mags, pulls, config.bands[j + 1],
                    config.bands[j] + ' - ' + config.bands[j + 1],
                    magrange)
        ax.set_title('Raw Error Ratio')

        fig.tight_layout()
        fig.savefig(os.path.join(config.outpath, config.plotpath,
                                 '%s_raw_error_ratio_%s-%s.png' % (config.outbase,
                                                                   config.bands[j],
                                                                   config.bands[j + 1])))
        plt.close(fig)

        if not config.calib_fit_err_ratio:
            continue

        # Make the scaled plot (with intrinsic error)

        delta = gals.galcol[:, j] - (zredstr['c'][zindex, j] + zredstr['slope'][zindex, j]*(gals.refmag - zredstr['pivotmag'][zindex]))
        err_ratios0 = zredstr['mag_err_ratio_intercept'][j] + zredstr['mag_err_ratio_slope'][j]*(gals.refmag - zredstr['mag_err_ratio_pivot'])
        err_ratios1 = zredstr['mag_err_ratio_intercept'][j + 1] + zredstr['mag_err_ratio_slope'][j + 1]*(gals.refmag - zredstr['mag_err_ratio_pivot'])
        delta_err = np.sqrt((err_ratios0*gals.mag_err[:, j])**2. +
                            (err_ratios1*gals.mag_err[:, j + 1])**2. +
                            zredstr['sigma'][j, j, zindex]**2.)
        pulls = delta/delta_err

        mags = gals.mag[:, j + 1]
        st = np.argsort(mags)
        magrange = [mags[st[int(0.01*mags.size)]], mags[st[int(0.99*mags.size)]]]

        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(111)

        _plot_pulls(ax, mags, pulls, config.bands[j + 1],
                    config.bands[j] + ' - ' + config.bands[j + 1],
                    magrange)
        ax.set_title('Scaled Error Ratio (r_err = %.3f/%.3f, %.3f/%.3f)' %
                     (zredstr['mag_err_ratio_intercept'][j], zredstr['mag_err_ratio_slope'][j],
                      zredstr['mag_err_ratio_intercept'][j + 1], zredstr['mag_err_ratio_slope'][j + 1]))

        fig.tight_layout()
        fig.savefig(os.path.join(config.outpath, config.plotpath,
                                 '%s_scaled_error_ratio_%s-%s.png' % (config.outbase,
                                                                      config.bands[j],
                                                                      config.bands[j + 1])))
        plt.close(fig)

def _plot_pulls(ax, mags, pulls, magname, colname, magrange, binsize=0.5, pullcut=10.0):
    gd, = np.where((np.abs(pulls) < pullcut) & (mags > magrange[0]) & (mags < magrange[1]))

    ax.hexbin(mags[gd], pulls[gd], bins='log', extent=[magrange[0], magrange[1],
                                                       -pullcut, pullcut])

    h, rev = esutil.stat.histogram(mags[gd], binsize=binsize, rev=True)

    binmags = np.zeros(h.size)
    sigs = np.zeros(h.size)
    lo = np.zeros(h.size)
    hi = np.zeros(h.size)
    med = np.zeros(h.size)
    u, = np.where(h > 0)
    for i, ind in enumerate(u):
        i1a = rev[rev[ind]: rev[ind + 1]]
        sigs[i] = 1.4826*np.median(np.abs(pulls[gd[i1a]] - np.median(pulls[gd[i1a]])))
        st = np.argsort(pulls[gd[i1a]])
        lo[i] = pulls[gd[i1a[st[int(0.05*i1a.size)]]]]
        hi[i] = pulls[gd[i1a[st[int(0.95*i1a.size)]]]]
        med[i] = pulls[gd[i1a[st[int(0.50*i1a.size)]]]]
        binmags[i] = np.median(mags[gd[i1a]])

    ok, = np.where(sigs > 0.0)

    ax.plot(binmags[ok], sigs[ok], 'r-', label='Width of Pulls')
    ax.plot(binmags[ok], lo[ok], 'k--', label='5/95th percentiles')
    ax.plot(binmags[ok], hi[ok], 'k--')
    ax.plot(binmags[ok], med[ok], 'r--', label='Median Pull')
    ax.plot(magrange, [1.0, 1.0], 'k:')

    ax.set_xlabel(magname)
    ax.set_ylabel('delta ' + colname)

# make two other QA plots
def _make_red_sequence_evolution_plots(config, pars):
    """
    Make red sequence evolution plots showing color-magnitude relations at different redshifts.
    """

    import matplotlib.pyplot as plt

    # Use config bands to generate color names
    color_list = [f"{b1}-{b2}" for b1, b2 in zip(config.bands[:-1], config.bands[1:])]
    redshift_list = pars.pivotmag_z
    colors = ["purple", "blue", "green", "orange", "red", "brown"]

    ncol = 0
    for i in range(10):
        if f"c{i:02d}" in pars._ndarray.dtype.names:
            ncol += 1
        else:
            break

    all_pivot_mags = []
    all_colors = []

    for target_z in redshift_list:
        pivot_mag = np.interp(target_z, pars.pivotmag_z, pars.pivotmag)
        all_pivot_mags.append(pivot_mag)
        for i in range(min(ncol, len(colors))):
            col_name = f"c{i:02d}"
            z_name = f"z{i:02d}"
            slope_name = f"slope{i:02d}"
            zs_name = f"zs{i:02d}"

            if all(
                name in pars._ndarray.dtype.names
                for name in [col_name, z_name, slope_name, zs_name]
            ):
                z_nodes_color = pars._ndarray[z_name][
                    pars._ndarray[z_name] > 0
                ]
                color_vals = pars._ndarray[col_name][: len(z_nodes_color)]
                z_nodes_slope = pars._ndarray[zs_name][
                    pars._ndarray[zs_name] > 0
                ]
                slope_vals = pars._ndarray[slope_name][: len(z_nodes_slope)]

                if len(z_nodes_color) > 1 and len(z_nodes_slope) > 1:
                    if (
                        target_z >= z_nodes_color.min()
                        and target_z <= z_nodes_color.max()
                    ):
                        color_at_z = np.interp(target_z, z_nodes_color, color_vals)
                        slope_at_z = np.interp(target_z, z_nodes_slope, slope_vals)

                        mag_range_full = np.linspace(
                            pivot_mag - 2, pivot_mag + 2, 100
                        )
                        red_sequence = color_at_z + slope_at_z * (
                            mag_range_full - pivot_mag
                        )
                        all_colors.extend(red_sequence)

    mag_min = min(all_pivot_mags) - 2
    mag_max = max(all_pivot_mags) + 2
    # color_min = min(all_colors) if all_colors else -1
    # color_max = max(all_colors) if all_colors else 2
    color_min, color_max = 0, 2.4

    color_range = color_max - color_min
    color_min -= color_range * 0.1
    color_max += color_range * 0.1

    # Calculate optimal subplot grid with fixed 2 rows
    n_plots = len(redshift_list)
    n_rows = 2
    n_cols = (n_plots + n_rows - 1) // n_rows  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, target_z in enumerate(redshift_list):
        ax = axes[plot_idx]

        pivot_mag = np.interp(target_z, pars.pivotmag_z, pars.pivotmag)
        mag_range = np.linspace(mag_min, mag_max, 100)

        plotted_any = False
        for i in range(min(ncol, len(colors))):
            col_name = f"c{i:02d}"
            z_name = f"z{i:02d}"
            slope_name = f"slope{i:02d}"
            zs_name = f"zs{i:02d}"

            if all(
                name in pars._ndarray.dtype.names
                for name in [col_name, z_name, slope_name, zs_name]
            ):
                z_nodes_color = pars._ndarray[z_name][
                    pars._ndarray[z_name] > 0
                ]
                color_vals = pars._ndarray[col_name][: len(z_nodes_color)]
                z_nodes_slope = pars._ndarray[zs_name][
                    pars._ndarray[zs_name] > 0
                ]
                slope_vals = pars._ndarray[slope_name][: len(z_nodes_slope)]

                if len(z_nodes_color) > 1 and len(z_nodes_slope) > 1:
                    if (
                        target_z >= z_nodes_color.min()
                        and target_z <= z_nodes_color.max()
                    ):
                        color_at_z = np.interp(target_z, z_nodes_color, color_vals)
                        slope_at_z = np.interp(target_z, z_nodes_slope, slope_vals)
                        red_sequence = color_at_z + slope_at_z * (
                            mag_range - pivot_mag
                        )

                        sigma_at_z = None
                        if hasattr(pars, "sigma") and hasattr(
                            pars, "covmat_z"
                        ):
                            try:
                                if (
                                    i < pars.sigma.shape[0]
                                    and i < pars.sigma.shape[1]
                                ):
                                    sigma_vals = pars.sigma[i, i, :]
                                    covmat_z = pars.covmat_z

                                    valid_mask = (covmat_z > 0) & (sigma_vals > 0)
                                    if np.any(valid_mask):
                                        covmat_z_valid = covmat_z[valid_mask]
                                        sigma_vals_valid = sigma_vals[valid_mask]

                                        if (
                                            target_z >= covmat_z_valid.min()
                                            and target_z <= covmat_z_valid.max()
                                        ):
                                            sigma_at_z = np.interp(
                                                target_z,
                                                covmat_z_valid,
                                                sigma_vals_valid,
                                            )

                                            ax.fill_between(
                                                mag_range,
                                                red_sequence - sigma_at_z,
                                                red_sequence + sigma_at_z,
                                                alpha=0.3,
                                                color=colors[i],
                                            )
                            except Exception as e:
                                logger.warning(
                                    f"Error computing sigma for color {i} at z={target_z:.2f}: {e}"
                                )

                        ax.plot(
                            mag_range,
                            red_sequence,
                            "-",
                            color=colors[i],
                            linewidth=2,
                            label=f"{color_list[i]}"
                            + (
                                rf" ($\sigma$={sigma_at_z:.3f})"
                                if sigma_at_z
                                else ""
                            ),
                        )
                        plotted_any = True

        ax.axvline(
            pivot_mag, color="black", linestyle="--", alpha=0.5, label="Pivot Mag"
        )
        ax.set_title(f"z = {target_z:.1f}", fontsize=12)
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Color")

        if plotted_any:
            ax.legend(loc="upper right", fontsize=8)

        ax.set_xlim(mag_min, mag_max)
        ax.set_ylim(color_min, color_max)

    plt.tight_layout()
    plt.suptitle("Red Sequence Model Evolution with Redshift", fontsize=16, y=1.02)

    fig.savefig(
        os.path.join(
            config.outpath,
            config.plotpath,
            "%s_red_sequence_evolution.png" % (config.outbase),
        )
    )
    plt.close(fig)

def _make_color_redshift_evolution_plots(config, pars):
    """
    Make plots showing color evolution with redshift, overlaying BC03 template colors.
    
    This plots the calibrated red sequence colors as a function of redshift
    at mstar, comparing them with the BC03 template colors used
    as initial guesses.
    """
    import matplotlib.pyplot as plt
    from astropy.table import Table
    
    # Use config bands to generate color names
    color_list = [f"{b1}-{b2}" for b1, b2 in zip(config.bands[:-1], config.bands[1:])]
    colors = ["purple", "blue", "green", "orange", "red", "brown"]
    
    # Determine number of colors
    ncol = 0
    for i in range(10):
        if f"c{i:02d}" in pars._ndarray.dtype.names:
            ncol += 1
        else:
            break
    
    # Load BC03 template if available
    bc03_data = None
    
    if config.calib_redgal_template is not None:
        try:
            # Construct path relative to redmapper package location
            # This file is in redmapper/calibration/redsequencecal.py
            # We want redmapper/data/initcolors/...
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(base_dir, 'data', 'initcolors', config.calib_redgal_template)
            
            bc03_data = Table.read(template_path, format="fits")
        except Exception as e:
            logger.warning(f"Could not load BC03 template: {e}")
    
    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    plotted_any = False
    
    # And a placeholder zredstr which allows us to do stuff
    zredstr = read_redsequence(None, config=config)

    # Plot each color
    for i in range(min(ncol, len(colors))):
        col_name = f"c{i:02d}"
        z_name = f"z{i:02d}"
        slope_name = f"slope{i:02d}"
        zs_name = f"zs{i:02d}"
        
        if col_name in pars._ndarray.dtype.names and z_name in pars._ndarray.dtype.names:
            # Get the nodes where colors are defined
            z_nodes = pars._ndarray[z_name][pars._ndarray[z_name] > 0]
            color_vals = pars._ndarray[col_name][:len(z_nodes)]
            
            if len(z_nodes) > 1:
                # We need to correct the calibrated colors to mstar to match the template
                # The calibrated colors are at pivotmag
                # C(mstar) = C(pivot) + slope * (mstar - pivot)
                
                # Get slope
                if slope_name in pars._ndarray.dtype.names and zs_name in pars._ndarray.dtype.names:
                    zs_nodes = pars._ndarray[zs_name][pars._ndarray[zs_name] > 0]
                    slope_vals_raw = pars._ndarray[slope_name][:len(zs_nodes)]
                    # Interpolate slope to z_nodes
                    slope_at_z = np.interp(z_nodes, zs_nodes, slope_vals_raw)
                else:
                    slope_at_z = np.zeros_like(z_nodes)
                    
                # Get pivotmag and mstar
                pivotmag_at_z = np.interp(z_nodes, pars.pivotmag_z, pars.pivotmag)
                mstar_at_z = redsequence_mstar(zredstr, z_nodes)
                
                # Apply correction
                color_vals_mstar = color_vals + slope_at_z * (mstar_at_z - pivotmag_at_z)

                # Plot the calibrated color nodes (at mstar)
                ax.scatter(
                    z_nodes,
                    color_vals_mstar,
                    color=colors[i],
                    s=50,
                    alpha=0.7,
                    zorder=3,
                    label=f"{color_list[i]}"
                )
                
                # Interpolate between nodes for smooth curve
                z_interp = np.linspace(z_nodes.min(), z_nodes.max(), 200)
                color_interp = np.interp(z_interp, z_nodes, color_vals_mstar)

                # Add error region (sigma)
                if hasattr(pars, "sigma") and hasattr(pars, "covmat_z"):
                    try:
                        if i < pars.sigma.shape[0]:
                            sigma_vals = pars.sigma[i, i, :]
                            covmat_z = pars.covmat_z
                            
                            valid_mask = (covmat_z > 0)
                            if np.any(valid_mask):
                                covmat_z_valid = covmat_z[valid_mask]
                                sigma_vals_valid = sigma_vals[valid_mask]
                                
                                # Interpolate sigma to z_interp
                                sigma_interp = np.interp(z_interp, covmat_z_valid, sigma_vals_valid)
                                
                                ax.fill_between(
                                    z_interp,
                                    color_interp - sigma_interp,
                                    color_interp + sigma_interp,
                                    color=colors[i],
                                    alpha=0.2,
                                    zorder=1
                                )
                    except Exception as e:
                        logger.warning(f"Error plotting sigma for color {i}: {e}")
                
                ax.plot(
                    z_interp,
                    color_interp,
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                    zorder=2
                )
                
                plotted_any = True
                
                # Overlay BC03 template if available
                # Use index i directly as it corresponds to the color index in the sequence
                if bc03_data is not None and i < bc03_data["COLOR"].shape[1]:
                    bc03_idx = i
                    
                    # Filter BC03 data to the calibration redshift range
                    z_mask = (bc03_data["Z"] >= config.zrange[0]) & \
                            (bc03_data["Z"] <= config.zrange[1])
                    
                    if np.any(z_mask):
                        ax.plot(
                            bc03_data["Z"][z_mask],
                            bc03_data["COLOR"][z_mask, bc03_idx],
                            color=colors[i],
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.6,
                            zorder=1,
                            label=f"{color_list[i]} (BC03 template)"
                        )
    
    if plotted_any:
        # Configure the plot
        ax.set_xlabel("Redshift (z)", fontsize=14)
        ax.set_ylabel("Color at Mstar", fontsize=14)
        ax.set_title("Red Sequence Color Evolution with Redshift (at Mstar)", fontsize=16)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(config.zrange[0] - 0.02, config.zrange[1] + 0.02)
        
        # Add vertical lines at pivotmag_z nodes for reference
        for z_pivot in pars.pivotmag_z:
            ax.axvline(z_pivot, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(
            config.outpath,
            config.plotpath,
            f"{config.outbase}_color_redshift_evolution.png"
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        logger.warning("No valid color-redshift data found to plot")
        plt.close(fig)