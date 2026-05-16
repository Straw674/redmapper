"""Functions for the color-based red-sequence model.

This module describes the red-sequence parameterization, and contains various
methods for using the model.
"""
import fitsio
import numpy as np
from .chisq_dist import compute_chisq
from .utilities import cubic_spline_compute_y2, cubic_spline_interpolate
from .utilities import read_mstar, get_mstar, read_redgal_initial_colors, get_redgal_initial_color
from .utilities import schechter_pdf

def read_redsequence(filename, zbinsize=None, minsig=0.01, fine=False, zrange=None, config=None, limmag=None):
    """
    Read the red-sequence model from a FITS file or a configuration.

    Parameters
    ----------
    filename: `str`
       Filename of the fits file to load parameters from.
       If None, uses the configuration to create a placeholder.
    zbinsize: `float`, optional
       Redshift binning to interpolate model.
    minsig: `float`, optional
       Minimum intrinsic scatter. Default is 0.01 mag.
    fine: `bool`, optional
       Use fine binning for interpolation. Default is False.
    zrange: `np.array`, optional
       Redshift range to do interpolation.
    config: `redmapper.Config`, optional
       Configuration dictionary. Required if filename=None.
    limmag: `float`, optional
       Maximum magnitude to do red-sequence interpolation.

    Returns
    -------
    redsequence_data: `dict`
       Dictionary containing all red-sequence model parameters.
    """
    if filename is None:
        if config is None:
            raise ValueError("Must have either filename or config")
        if limmag is None:
            limmag = config.limmag_catalog
        if zrange is None:
            zrange = config.zrange
        alpha = config.get('calib_lumfunc_alpha', -1.0)
        mstar_survey = config.get('mstar_survey', 'sdss')
        mstar_band = config.get('mstar_band', 'i')
        ncol = config.nmag - 1
        templatefile = config.get('calib_redgal_template', None)
        bands = config.bands

        zbinsize = 0.001
        has_file = False
        hdr = None
    else:
        pars, hdr = fitsio.read(filename, ext=1, header=True, upper=True)
        try:
            if limmag is None:
                limmag = hdr['LIMMAG']
            if zrange is None:
                zrange = np.array([hdr['ZRANGE0'], hdr['ZRANGE1']])
            alpha = hdr['ALPHA']
            mstar_survey = hdr['MSTARSUR']
            mstar_band = hdr['MSTARBAN']
            ncol = hdr['NCOL']
            if 'TEMPLATE' in hdr:
                templatefile = hdr['TEMPLATE']
                if ',' in hdr['BANDS']:
                    bands = hdr['BANDS'].rstrip().split(',')
                else:
                    bands = list(hdr['BANDS'].rstrip())
            else:
                templatefile = None
        except KeyError as e:
            raise ValueError("Missing field from parameter header: %s" % (e))
        has_file = True

    if len(zrange) != 2:
        raise ValueError("zrange must have 2 elements")

    if zbinsize is None:
        try:
            if fine:
                zbinsize = hdr['ZBINFINE']
            else:
                zbinsize = hdr['ZBINCOAR']
        except (KeyError, TypeError):
            raise ValueError("Missing field from parameter header (ZBINFINE/ZBINCOAR).")

    try:
        lowzmode = hdr['LOWZMODE'] if hdr is not None else 0
    except (KeyError, TypeError):
        lowzmode = 0

    nmag = ncol + 1
    
    if has_file:
        bvalues = np.zeros(nmag)
        try:
            for i in range(nmag):
                bvalues[i] = hdr['BVALUE%1d' % (i+1)]
        except (KeyError, TypeError):
            bvalues[:] = 0.0

        do_lupcorr = bvalues.min() > 0.0

        try:
            ref_ind = hdr['REF_IND']
        except (KeyError, TypeError):
            try:
                ref_ind = hdr['I_IND']
            except (KeyError, TypeError):
                raise ValueError("Need REF_IND or I_IND")

    rg_data = None
    if templatefile is not None:
        rg_data = read_redgal_initial_colors(templatefile)

    nz = np.round((zrange[1]-zrange[0])/zbinsize).astype('i4')
    z_bins = zbinsize*np.arange(nz) + zrange[0]
    z_bins = np.append(z_bins, z_bins[z_bins.size-1])
    nz = nz + 1

    zbinscale = int(1./zbinsize)
    mstar_data = read_mstar(mstar_survey, mstar_band)

    refmagbinsize = 0.01
    if lowzmode:
        refmagrange = np.array([10.0, limmag], dtype='f4')
        lumrefmagrange = np.array([10.0, get_mstar(mstar_data, zrange[1])-2.5*np.log10(0.1)])
    else:
        refmagrange = np.array([12.0, limmag], dtype='f4')
        lumrefmagrange = np.array([12.0, get_mstar(mstar_data, zrange[1])-2.5*np.log10(0.1)])
    
    refmagbins = np.arange(refmagrange[0], refmagrange[1], refmagbinsize, dtype='f8')
    lumrefmagbins = np.arange(lumrefmagrange[0], lumrefmagrange[1], refmagbinsize, dtype='f8')

    refmagbins = np.append(refmagbins, refmagbins[refmagbins.size-1])
    lumrefmagbins = np.append(lumrefmagbins, lumrefmagbins[lumrefmagbins.size-1])

    refmagbinscale = int(1./refmagbinsize)
    refmaginteger = (refmagbins*refmagbinscale).astype(np.int64)
    lumrefmaginteger = (lumrefmagbins*refmagbinscale).astype(np.int64)

    data = {
        'z': z_bins,
        'zbinsize': zbinsize,
        'zbinscale': zbinscale,
        'refmagbins': refmagbins,
        'lumrefmagbins': lumrefmagbins,
        'refmagbinsize': refmagbinsize,
        'refmagbinscale': refmagbinscale,
        'refmaginteger': refmaginteger,
        'lumrefmaginteger': lumrefmaginteger,
        'ncol': ncol,
        'nmag': nmag,
        'alpha': alpha,
        'mstar_survey': mstar_survey,
        'mstar_band': mstar_band,
        'limmag': limmag,
        'extrapolated': np.zeros(nz, dtype=bool),
        'pivotmag': np.zeros(nz, dtype=np.float64),
        'maxrefmag': np.zeros(nz, dtype=np.float64),
        'minrefmag': np.zeros(nz, dtype=np.float64),
        'c': np.zeros((nz, ncol), dtype=np.float64),
        'slope': np.zeros((nz, ncol), dtype=np.float64),
        'sigma': np.zeros((ncol, ncol, nz), dtype=np.float64),
        'covmat': np.zeros((ncol, ncol, nz), dtype=np.float64),
        'volume_factor': np.zeros(nz, dtype=np.float64),
        'corr': np.zeros(nz, dtype=np.float64),
        'corr_slope': np.zeros(nz, dtype=np.float64),
        'corr2': np.zeros(nz, dtype=np.float64),
        'corr2_slope': np.zeros(nz, dtype=np.float64),
        'corr_r': np.ones(nz, dtype=np.float64),
        'corr2_r': np.ones(nz, dtype=np.float64),
        'mag_err_ratio_intercept': np.ones(nmag, dtype=np.float64),
        'mag_err_ratio_slope': np.zeros(nmag, dtype=np.float64),
        'mag_err_ratio_pivot': 20.0,
        'lupcorr': None,
        '_mstar': get_mstar(mstar_data, z_bins),
    }

    if has_file:
        if 'PIVOTMAG_Z' in pars.dtype.names:
            refmag_name = 'REFMAG'
            pivotmag_name = 'PIVOTMAG'
        else:
            refmag_name = 'IMAG'
            pivotmag_name = 'REFMAG'

        loz, = np.where(z_bins < np.min(pars[pivotmag_name+'_Z']))
        hiz, = np.where(z_bins > np.max(pars[pivotmag_name+'_Z']))
        if loz.size > 0: data['extrapolated'][loz] = True
        if hiz.size > 0: data['extrapolated'][hiz] = True

        y2 = cubic_spline_compute_y2(pars[0][pivotmag_name+'_Z'], pars[0][pivotmag_name])
        data['pivotmag'][:] = cubic_spline_interpolate(z_bins, pars[0][pivotmag_name+'_Z'], pars[0][pivotmag_name], y2)

        y2 = cubic_spline_compute_y2(pars[0][pivotmag_name+'_Z'], pars[0]['MAX'+refmag_name])
        data['maxrefmag'][:] = cubic_spline_interpolate(z_bins, pars[0][pivotmag_name+'_Z'], pars[0]['MAX'+refmag_name], y2)
        y2 = cubic_spline_compute_y2(pars[0][pivotmag_name+'_Z'], pars[0]['MIN'+refmag_name])
        data['minrefmag'][:] = cubic_spline_interpolate(z_bins, pars[0][pivotmag_name+'_Z'], pars[0]['MIN'+refmag_name], y2)

        for j in range(ncol):
            jstring = '%02d' % (j)
            y2 = cubic_spline_compute_y2(pars[0]['Z'+jstring], pars[0]['C'+jstring])
            data['c'][:, j] = cubic_spline_interpolate(z_bins, pars[0]['Z'+jstring], pars[0]['C'+jstring], y2)
            y2 = cubic_spline_compute_y2(pars[0]['ZS'+jstring], pars[0]['SLOPE'+jstring])
            data['slope'][:, j] = cubic_spline_interpolate(z_bins, pars[0]['ZS'+jstring], pars[0]['SLOPE'+jstring], y2)

        if 'MAG_ERR_RATIO_INT' in pars.dtype.names:
            data['mag_err_ratio_intercept'][:] = pars[0]['MAG_ERR_RATIO_INT'][:]
            data['mag_err_ratio_slope'][:] = pars[0]['MAG_ERR_RATIO_SLOPE'][:]
            data['mag_err_ratio_pivot'] = pars[0]['MAG_ERR_RATIO_PIVOT']

        for j in range(ncol):
            y2 = cubic_spline_compute_y2(pars[0]['COVMAT_Z'], pars[0]['SIGMA'][j, j, :])
            data['sigma'][j, j, :] = np.clip(cubic_spline_interpolate(z_bins, pars[0]['COVMAT_Z'], pars[0]['SIGMA'][j, j, :], y2), minsig, None)
            data['covmat'][j, j, :] = data['sigma'][j, j, :]**2

        for j in range(ncol):
            for k in range(j+1, ncol):
                y2 = cubic_spline_compute_y2(pars[0]['COVMAT_Z'], pars[0]['SIGMA'][j, k, :])
                data['sigma'][j, k, :] = np.clip(cubic_spline_interpolate(z_bins, pars[0]['COVMAT_Z'], pars[0]['SIGMA'][j, k, :], y2), -0.99, 0.99)
                data['sigma'][k, j, :] = data['sigma'][j, k, :]
                data['covmat'][j, k, :] = data['sigma'][k, j, :] * data['sigma'][j, j, :] * data['sigma'][k, k, :]
                data['covmat'][k, j, :] = data['covmat'][j, k, :]

        y2 = cubic_spline_compute_y2(pars[0]['VOLUME_FACTOR_Z'], pars[0]['VOLUME_FACTOR'])
        data['volume_factor'][:] = cubic_spline_interpolate(z_bins, pars[0]['VOLUME_FACTOR_Z'], pars[0]['VOLUME_FACTOR'], y2)

        y2 = cubic_spline_compute_y2(pars[0]['CORR_Z'], pars[0]['CORR'])
        data['corr'][:] = cubic_spline_interpolate(z_bins, pars[0]['CORR_Z'], pars[0]['CORR'], y2)
        y2 = cubic_spline_compute_y2(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR_SLOPE'])
        data['corr_slope'][:] = cubic_spline_interpolate(z_bins, pars[0]['CORR_SLOPE_Z'], pars[0]['CORR_SLOPE'], y2)

        y2 = cubic_spline_compute_y2(pars[0]['CORR_Z'], pars[0]['CORR2'])
        data['corr2'][:] = cubic_spline_interpolate(z_bins, pars[0]['CORR_Z'], pars[0]['CORR2'], y2)
        y2 = cubic_spline_compute_y2(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR2_SLOPE'])
        data['corr2_slope'][:] = cubic_spline_interpolate(z_bins, pars[0]['CORR_SLOPE_Z'], pars[0]['CORR2_SLOPE'], y2)

        if 'CORR_R' in pars.dtype.names:
            if pars[0]['CORR_R'][0] > 0.0:
                y2 = cubic_spline_compute_y2(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR_R'])
                data['corr_r'][:] = np.clip(cubic_spline_interpolate(z_bins, pars[0]['CORR_SLOPE_Z'], pars[0]['CORR_R'], y2), 0.5, None)
            if pars[0]['CORR2_R'][0] > 0.0:
                y2 = cubic_spline_compute_y2(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR2_R'])
                data['corr2_r'][:] = np.clip(cubic_spline_interpolate(z_bins, pars[0]['CORR_SLOPE_Z'], pars[0]['CORR2_R'], y2), 0.5, None)

        if rg_data is not None:
            not_extrap, = np.where(~data['extrapolated'])
            data['pivotmag'][loz] = data['pivotmag'][not_extrap[0]]
            data['pivotmag'][hiz] = data['pivotmag'][not_extrap[-1]]
            for j in range(ncol):
                data['slope'][loz, j] = data['slope'][not_extrap[0], j]
                data['slope'][hiz, j] = data['slope'][not_extrap[-1], j]
                try:
                    delta = get_redgal_initial_color(rg_data, bands[j], bands[j+1], z_bins[not_extrap[0]]) - data['c'][not_extrap[0], j]
                    data['c'][loz, j] = get_redgal_initial_color(rg_data, bands[j], bands[j+1], z_bins[loz]) - delta
                    delta = get_redgal_initial_color(rg_data, bands[j], bands[j+1], z_bins[not_extrap[-1]]) - data['c'][not_extrap[-1], j]
                    data['c'][hiz, j] = get_redgal_initial_color(rg_data, bands[j], bands[j+1], z_bins[hiz]) - delta
                except (ValueError, IndexError):
                    pass

    # luminosity function integrations
    data['lumnorm'] = np.zeros((lumrefmagbins.size, nz))
    for i in range(nz):
        f = schechter_pdf(lumrefmagbins, alpha=alpha, mstar=data['_mstar'][i])
        data['lumnorm'][:, i] = refmagbinsize*np.cumsum(f, dtype=np.float64)

    if has_file:
        data['lupcorr'] = np.zeros((refmagbins.size, nz, ncol), dtype='f8')
        if do_lupcorr:
            bnmgy = bvalues*1e9
            for i in range(nz):
                mags = np.zeros((refmagbins.size, nmag))
                lups = np.zeros((refmagbins.size, nmag))
                mags[:, ref_ind] = refmagbins
                for j in range(ref_ind+1, nmag):
                    mags[:, j] = mags[:, j-1] - (data['c'][i, j-1]+data['slope'][i, j-1]*(mags[:, ref_ind]-data['pivotmag'][i]))
                for j in range(ref_ind-1, -1, -1):
                    mags[:, j] = mags[:, j+1] + (data['c'][i, j]+data['slope'][i, j]*(mags[:, ref_ind]-data['pivotmag'][i]))
                for j in range(nmag):
                    flux = 10.**((mags[:, j]-22.5)/(-2.5))
                    lups[:, j] = 2.5*np.log10(1.0/bvalues[j]) - np.arcsinh(0.5*flux/bnmgy[j])/(0.4*np.log(10.0))
                data['lupcorr'][:, i, :] = (lups[:, 0:ncol] - lups[:, 1:ncol+1]) - (mags[:, 0:ncol] - mags[:, 1:ncol+1])

    data['z'][data['z'].size-1] = 1000.0
    data['zinteger'] = np.round(data['z']*zbinscale).astype(np.int64)
    data['refmagbins'][data['refmagbins'].size-1] = 1000.0
    data['refmaginteger'] = np.round(data['refmagbins']*refmagbinscale).astype(np.int64)
    data['lumrefmagbins'][data['lumrefmagbins'].size-1] = 1000.0
    data['lumrefmaginteger'] = np.round(data['lumrefmagbins']*refmagbinscale).astype(np.int64)

    return data

def redsequence_zindex(redsequence_data, z):
    """
    Look up the redshift index for the red-sequence model.

    Parameters
    ----------
    redsequence_data: `dict`
       Dictionary containing red-sequence model parameters.
    z: `np.array`
       Float array of redshifts.

    Returns
    -------
    zindex: `np.array`
       Integer array of redshift indices.
    """
    zind = np.searchsorted(redsequence_data['zinteger'], np.round(np.atleast_1d(z)*redsequence_data['zbinscale']).astype(np.int64))
    return np.ndarray.item(zind) if zind.size == 1 else zind

def redsequence_refmagindex(redsequence_data, refmag):
    """
    Look up the reference magnitude index for the red-sequence model.

    Parameters
    ----------
    redsequence_data: `dict`
       Dictionary containing red-sequence model parameters.
    refmag: `np.array`
       Float array of refmag values.

    Returns
    -------
    indices: `np.array`
       Integer array of refmag indices.
    """
    refmagind = np.searchsorted(redsequence_data['refmaginteger'], np.round(np.atleast_1d(refmag)*redsequence_data['refmagbinscale']).astype(np.int64))
    return np.ndarray.item(refmagind) if refmagind.size == 1 else refmagind

def redsequence_lumrefmagindex(redsequence_data, lumrefmag):
    """
    Look up the luminosity table reference magnitude index.

    Parameters
    ----------
    redsequence_data: `dict`
       Dictionary containing red-sequence model parameters.
    lumrefmag: `np.array`
       Float array of refmags for luminosity table.

    Returns
    -------
    indices: `np.array`
       Integer array of lumrefmag indices.
    """
    lumrefmagind = np.searchsorted(redsequence_data['lumrefmaginteger'], np.round(np.atleast_1d(lumrefmag)*redsequence_data['refmagbinscale']).astype(np.int64))
    return np.ndarray.item(lumrefmagind) if lumrefmagind.size == 1 else lumrefmagind

def redsequence_mstar(redsequence_data, z):
    """
    Look up mstar at a set of redshifts.

    Parameters
    ----------
    redsequence_data: `dict`
       Dictionary containing red-sequence model parameters.
    z: `np.array`
       Float array of redshifts.

    Returns
    -------
    mstar: `np.array`
       Float array of mstar values.
    """
    zind = redsequence_zindex(redsequence_data, z)
    return redsequence_data['_mstar'][zind]

def compute_redsequence_chisq_redshifts(redsequence_data, galaxy, zs, calc_lkhd=False, z_is_index=False):
    """
    Compute chisq for a single galaxy at a series of redshifts.

    Parameters
    ----------
    redsequence_data: `dict`
       Dictionary containing red-sequence model parameters.
    galaxy: `astropy.table.Row`
       The galaxy to compute chisq values.
    zs: `np.array`
       Float array of redshifts or integer array of bins.
    calc_lkhd: `bool`, optional
       Calculate likelihood rather than chisq. Default is False.
    z_is_index: `bool`, optional
       The zs are indices and not redshifts. Default is False.

    Returns
    -------
    chisqs: `np.array`
       Float array of chisq values.
    """
    calc_chisq = not calc_lkhd
    zinds = zs if z_is_index else redsequence_zindex(redsequence_data, zs)
    magind = redsequence_refmagindex(redsequence_data, galaxy['refmag'])

    if 'galcol' in galaxy.colnames if hasattr(galaxy, 'colnames') else hasattr(galaxy, 'galcol'):
        galcolor = galaxy['galcol'] if hasattr(galaxy, 'colnames') else galaxy.galcol
    else:
        from .galaxy import compute_colors
        galcolor = compute_colors(galaxy['mag'])

    mag_err = galaxy['mag_err'].copy()
    dmag = galaxy['refmag'] - redsequence_data['mag_err_ratio_pivot']
    mag_err *= (redsequence_data['mag_err_ratio_intercept'][:] + redsequence_data['mag_err_ratio_slope'][:]*dmag)

    lupcorr = redsequence_data['lupcorr'][magind, zinds, :] if redsequence_data['lupcorr'] is not None else None

    return compute_chisq(redsequence_data['covmat'][:, :, zinds], redsequence_data['c'][zinds, :],
                         redsequence_data['slope'][zinds, :], redsequence_data['pivotmag'][zinds],
                         np.array(galaxy['refmag']), mag_err,
                         galcolor,
                         refmagerr=np.array(galaxy['refmag_err']),
                         lupcorr=lupcorr,
                         calc_chisq=calc_chisq, calc_lkhd=calc_lkhd)

def compute_redsequence_chisq(redsequence_data, galaxies, z, calc_lkhd=False, z_is_index=False):
    """
    Compute chisq for galaxies at redshift(s).

    Parameters
    ----------
    redsequence_data: `dict`
       Dictionary containing red-sequence model parameters.
    galaxies: `astropy.table.Table` or `astropy.table.Row`
       Catalog of galaxies to compute chisq values.
    z: `np.array`
       Float array of redshifts or integer array of redshift indices.
    calc_lkhd: `bool`, optional
       Calculate likelihood rather than chisq. Default is False.
    z_is_index: `bool`, optional
       The zs are indices and not redshifts. Default is False.

    Returns
    -------
    chisqs: `np.array`
       Float array of chisq values.
    """
    calc_chisq = not calc_lkhd
    zind = z if z_is_index else redsequence_zindex(redsequence_data, z)
    magind = redsequence_refmagindex(redsequence_data, galaxies['refmag'])

    if 'galcol' in galaxies.colnames if hasattr(galaxies, 'colnames') else hasattr(galaxies, 'galcol'):
        galcolor = galaxies['galcol'] if hasattr(galaxies, 'colnames') else galaxies.galcol
    else:
        from .galaxy import compute_colors
        galcolor = compute_colors(galaxies['mag'])

    if np.atleast_1d(zind).size == 1 and (len(galaxies) == 1 if hasattr(galaxies, '__len__') else True):
        zind = np.atleast_1d(zind)
        magind = np.atleast_1d(magind)

    mag_err = galaxies['mag_err'].copy()
    dmags = galaxies['refmag'] - redsequence_data['mag_err_ratio_pivot']
    for j in range(redsequence_data['nmag']):
        mag_err[..., j] *= redsequence_data['mag_err_ratio_intercept'][j] + redsequence_data['mag_err_ratio_slope'][j]*dmags

    lupcorr = redsequence_data['lupcorr'][magind, zind, :] if redsequence_data['lupcorr'] is not None else None

    return compute_chisq(redsequence_data['covmat'][:, :, zind], redsequence_data['c'][zind, :],
                         redsequence_data['slope'][zind, :], redsequence_data['pivotmag'][zind],
                         galaxies['refmag'], mag_err,
                         galcolor,
                         refmagerr=galaxies['refmag_err'],
                         lupcorr=lupcorr,
                         calc_chisq=calc_chisq, calc_lkhd=calc_lkhd)

def plot_redsequence_diag(redsequence_data, fig, ind, bands):
    """
    Plot the diagonal elements of the red-sequence model.
    """
    not_extrap, = np.where(~redsequence_data['extrapolated'])
    z = redsequence_data['z']

    ax = fig.add_subplot(221)
    ax.plot(z[:-1], redsequence_data['c'][:-1, ind], 'r--')
    ax.plot(z[not_extrap], redsequence_data['c'][not_extrap, ind], 'r-')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('<%s - %s>' % (bands[ind], bands[ind + 1]))

    ax = fig.add_subplot(222)
    ax.plot(z[:-1], redsequence_data['slope'][:-1, ind], 'r--')
    ax.plot(z[not_extrap], redsequence_data['slope'][not_extrap, ind], 'r-')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('(%s - %s) slope' % (bands[ind], bands[ind + 1]))

    ax = fig.add_subplot(223)
    ax.plot(z[:-1], redsequence_data['sigma'][ind, ind, :-1], 'r--')
    ax.plot(z[not_extrap], redsequence_data['sigma'][ind, ind, not_extrap], 'r-')
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('(%s - %s) sigma' % (bands[ind], bands[ind + 1]))

    fig.tight_layout()

def plot_redsequence_offdiags(redsequence_data, fig, bands):
    """
    Plot the off-diagonal elements of the red-sequence model.
    """
    ncol = redsequence_data['ncol']
    noff = (ncol * ncol - ncol) // 2
    nrow = (noff + 1) // 2
    not_extrap, = np.where(~redsequence_data['extrapolated'])
    z = redsequence_data['z']

    ctr = 1
    for j in range(ncol):
        for k in range(j + 1, ncol):
            ax = fig.add_subplot(nrow, 2, ctr)
            ax.plot(z[:-1], redsequence_data['sigma'][j, k, :-1], 'r--')
            ax.plot(z[not_extrap], redsequence_data['sigma'][j, k, not_extrap], 'r-')
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlabel('Redshift')
            ax.set_ylabel('Corr %s-%s / %s-%s' % (bands[j], bands[j + 1], bands[k], bands[k + 1]))
            ctr += 1

    fig.tight_layout()

