"""Functional interfaces for making pretty diagnostic plots for redmapper catalogs
"""
import os
import numpy as np
import fitsio
import esutil
import scipy.optimize
import scipy.ndimage
import copy
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .utilities import gaussFunction, cubic_spline_compute_y2, cubic_spline_interpolate, interpol
from .logger import logger

def plot_spec_comparison(config, z_spec, z_phot, z_phot_e, name=r'z_\lambda', specname=r'z_{\mathrm{spec}}',
                         title=None, figure_return=False, calib_zrange=None, withversion=False,
                         binsize=0.02, nsig=4.0):
    """
    Make a pretty spectroscopic plot from an arbitrary list of values.

    Parameters
    ----------
    config: `dict`
       Configuration dictionary
    z_spec: `np.array`
       Float array of spectroscopic redshifts
    z_phot: `np.array`
       Float array of photometric redshifts
    z_phot_e: `np.array`
       Float array of photometric redshift errors
    name: `str`, optional
       Name of photo-z field for label.  Default is 'z_lambda'.
    specname: `str`, optional
       Name of spec-z field for label.
    title: `str`, optional
       Title string.
    figure_return: `bool`, optional
       Return the figure instead of saving a png.
    calib_zrange: `np.array` or `list`, optional
       2-element array with calibration redshift range to mark.
    withversion: `bool`, optional
       Plots should be saved with the version string.
    binsize: `float`, optional
       Redshift smoothing bin size.
    nsig: `float`, optional
       Number of sigma to be considered a redshift outlier.
    """
    name_clean = name.replace('_', '').replace('^', '')
    use, = np.where(z_spec > 0.0)

    plot_xrange = np.array([0.0, config['zrange'][1] + 0.1])

    fig = plt.figure(figsize=(8, 6))
    fig.clf()

    ax = fig.add_subplot(211)

    z_bins, z_map = _make_photoz_map(z_spec[use], z_phot[use], zrange=[0.0, config['zrange'][1] + 0.1])

    nlevs = 5
    levbinsize = (1.0 - 0.1) / nlevs
    levs = np.arange(nlevs) * levbinsize + 0.1
    levs = np.append(levs, 10)
    levs = -levs[::-1]

    colors = ['#000000', '#333333', '#666666', '#9A9A9A', '#CECECE', '#FFFFFF']

    ax.contourf(z_bins, z_bins, -z_map.T, levs, colors=colors)
    ax.plot(plot_xrange, plot_xrange, 'b--', linewidth=3)
    ax.set_xlim(plot_xrange)
    ax.set_ylim(plot_xrange)
    ax.tick_params(axis='y', which='major', labelsize=14, length=5, left=True, right=True, direction='in')
    ax.tick_params(axis='y', which='minor', left=True, right=True, direction='in')
    ax.tick_params(axis='x', labelbottom=False, which='major', length=5, direction='in', bottom=True, top=True)
    ax.tick_params(axis='x', which='minor', bottom=True, top=True, direction='in')
    minorLocator = MultipleLocator(0.05)
    ax.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.02)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.set_ylabel(r'$%s$' % (specname), fontsize=16)

    if calib_zrange is not None:
        if len(calib_zrange) == 2:
            ylim = ax.get_ylim()
            ax.plot([calib_zrange[0], calib_zrange[0]], ylim, 'k:')
            ax.plot([calib_zrange[1], calib_zrange[1]], ylim, 'k:')

    bad, = np.where(np.abs(z_phot[use] - z_spec[use]) / z_phot_e[use] > nsig)
    if bad.size > 0:
        ax.plot(z_phot[use[bad]], z_spec[use[bad]], 'r*')

    fracout = float(bad.size) / float(use.size)
    fout_label = r'$f_\mathrm{out} = %7.4f$' % (fracout)
    ax.annotate(fout_label, (plot_xrange[0] + 0.1, config['zrange'][1]),
                xycoords='data', ha='left', va='top', fontsize=16)

    ax.set_title(title)

    h, rev = esutil.stat.histogram(z_phot[use], min=0.0, max=config['zrange'][1]-0.001, rev=True, binsize=binsize)
    bins = np.arange(h.size) * binsize + 0.0 + binsize/2.

    bias = np.zeros(h.size)
    scatter = np.zeros_like(bias)
    errs = np.zeros_like(bias)

    gd, = np.where(h >= 3)
    for ind in gd:
        i1a = rev[rev[ind]: rev[ind + 1]]
        bias[ind] = np.median(z_spec[use[i1a]] - z_phot[use[i1a]])
        scatter[ind] = 1.4862 * np.median(np.abs((z_spec[use[i1a]] - z_phot[use[i1a]]) - bias[ind]))
        errs[ind] = np.median(z_phot_e[use[i1a]])

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(plot_xrange, [0.0, 0.0], 'b--', linewidth=2)
    ax2.plot(bins, bias, 'm-.', label=r'$\mathrm{Bias}$', linewidth=3)
    ax2.plot(bins, scatter/(1. + bins), 'r:', label=r'$\sigma_z / (1 + z)$', linewidth=3)
    ax2.plot(bins, errs/(1. + bins), 'c--', label=r'$\sigma_{%s} / (1 + z)$' % (name_clean), linewidth=3)

    ax2.legend(loc=4, fontsize=14)
    ax2.set_xlim(plot_xrange)
    ax2.set_ylim([-0.03, 0.024])

    if calib_zrange is not None:
        if len(calib_zrange) == 2:
            ax2.plot([calib_zrange[0], calib_zrange[0]], [-0.03, 0.024], 'k:')
            ax2.plot([calib_zrange[1], calib_zrange[1]], [-0.03, 0.024], 'k:')

    ax2.tick_params(axis='both', which='major', labelsize=14, length=5, left=True, right=True, top=True, bottom=True, direction='in')
    ax2.tick_params(axis='y', which='minor', left=True, right=True, direction='in')
    ax2.tick_params(axis='x', which='minor', bottom=True, top=True, direction='in')
    ax2.set_xlabel(r'$%s$' % (name), fontsize=16)
    ax2.set_ylabel(r'$%s - %s$' % (specname, name), fontsize=16)
    minorLocator = MultipleLocator(0.02)
    ax2.xaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.002)
    ax2.yaxis.set_minor_locator(minorLocator)

    fig.subplots_adjust(hspace=0.0)
    fig.tight_layout()

    if not figure_return:
        from .configuration import get_redmapper_filename
        filename = get_redmapper_filename(config, 'zspec', paths=(config.get('plotpath'),),
                                          withversion=withversion, filetype='png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
        plt.close(fig)
        return filename
    else:
        return fig

def _make_photoz_map(z_spec, z_photo, dzmax=0.1, nbins_coarse=20, nbins_fine=200, zrange=[0.0, 1.2]):
    zmin, zmax = z_photo.min(), z_photo.max()
    zbinsize = (zmax - zmin) / nbins_coarse
    zbins = np.arange(nbins_coarse) * zbinsize + zmin
    dzbinsize = (2. * dzmax)/(nbins_coarse)
    dzbins = np.arange(nbins_coarse) * dzbinsize - dzmax
    z, dz = zbins + 0.5 * zbinsize, dzbins + 0.5 * dzbinsize
    spl = np.zeros((nbins_coarse, nbins_coarse))
    for i in range(nbins_coarse):
        use, = np.where((z_photo >= zbins[i]) & (z_photo < (zbins[i] + zbinsize)))
        if use.size == 0: continue
        dzvec = z_spec[use] - z_photo[use]
        for j in range(nbins_coarse):
            use2, = np.where((dzvec >= dzbins[j]) & (dzvec < (dzbins[j] + dzbinsize)))
            spl[i, j] = use2.size
        p0 = [use.size, np.median(dzvec), np.std(dzvec)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction, dz, spl[i, :], p0=p0)
        except:
            coeff = p0
        spl[i, :] /= coeff[0]
    maxdz = 0.5 * (z[1] - z[0])
    zbinsize = (zrange[1] - zrange[0]) / nbins_fine
    z_bins = np.arange(nbins_fine) * zbinsize + zrange[0]
    z_map_values = np.zeros((nbins_fine, nbins_fine))
    for i in range(nbins_fine):
        delta = np.abs(z_bins[i] - z)
        index = np.argmin(delta)
        if delta[index] < maxdz:
            dz1 = z_bins - z_bins[i]
            y2 = cubic_spline_compute_y2(dz, spl[index, :])
            y = cubic_spline_interpolate(dz1, dz, spl[index, :], y2)
            y[(np.abs(dz1) > 0.1) | (y < 0.0)] = 0.0
            if y.max() > 0:
                z_map_values[i, :] = y / y.max()
    z_map_values[z_map_values < 1e-3] = 0.0
    return z_bins, scipy.ndimage.uniform_filter(z_map_values, size=5, mode='nearest')

def plot_nz(config, z, areastr, zrange, xlabel=None, ylabel=None, title=None,
            redmapper_name='nz', calib_zrange=None, withversion=False, binsize=0.02):
    """
    Plot the n(z) for an arbitrary list of objects.
    """
    hist = esutil.stat.histogram(z, min=zrange[0], max=zrange[1]-0.0001, binsize=binsize, more=True)
    h, zbins = hist['hist'], hist['center']
    indices = np.clip(np.searchsorted(areastr.z, zbins), 0, areastr.size - 1)
    vol = np.zeros(zbins.size)
    for i in range(zbins.size):
        vol[i] = (config['cosmo'].V(np.clip(zbins[i] - binsize/2., zrange[0], None),
                                     np.clip(zbins[i] + binsize/2., None, zrange[1])) *
                  (areastr.area[indices[i]] / 41252.961))
    dens = h.astype(np.float32) / vol
    err = np.sqrt(dens * vol) / vol
    u, = np.where((dens > 0.0) & (err > 0.0))
    u2, = np.where((dens[u] / err[u]) > 5.0)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(zbins[u[u2]], dens[u[u2]]*1e4, yerr=err[u[u2]]*1e4, fmt='r.', markersize=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=16)
    if ylabel: ax.set_ylabel(ylabel, fontsize=16)
    if title: ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if calib_zrange and len(calib_zrange) == 2:
        ylim = ax.get_ylim()
        ax.plot([calib_zrange[0], calib_zrange[0]], ylim, 'k:')
        ax.plot([calib_zrange[1], calib_zrange[1]], ylim, 'k:')
    fig.tight_layout()
    from .configuration import get_redmapper_filename
    filename = get_redmapper_filename(config, redmapper_name, paths=(config.get('plotpath'),),
                                      withversion=withversion, filetype='png')
    fig.savefig(filename)
    plt.close(fig)
    return filename

def plot_nlambda(config, lam, xlabel=None, ylabel=None, title=None, 
                 redmapper_name='nlambda', withversion=False, binsize=1.0):
    """
    Plot the n(lambda) for an arbitrary list of objects.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(lam, bins=np.arange(0, 200, binsize))
    ax.set_yscale('log')
    if xlabel: ax.set_xlabel(xlabel, fontsize=16)
    if ylabel: ax.set_ylabel(ylabel, fontsize=16)
    if title: ax.set_title(title, fontsize=16)
    fig.tight_layout()
    from .configuration import get_redmapper_filename
    filename = get_redmapper_filename(config, redmapper_name, paths=(config.get('plotpath'),),
                                      withversion=withversion, filetype='png')
    fig.savefig(filename)
    plt.close(fig)
    return filename

def plot_positions(config, ra, dec, title=None, redmapper_name='positions', withversion=False):
    """
    Plot the ra/dec positions.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    delta_ra_range = ra.max() - ra.min()
    ra_rot = (ra + 180) % 360.0
    delta_ra_rot_range = ra_rot.max() - ra_rot.min()
    ra_to_plot = ra_rot if delta_ra_rot_range < delta_ra_range else ra
    ax.plot(ra_to_plot, dec, 'r.')
    ax.invert_xaxis()
    ax.set_xlabel('RA', fontsize=16)
    ax.set_ylabel('Dec', fontsize=16)
    if title: ax.set_title(title, fontsize=16)
    fig.tight_layout()
    from .configuration import get_redmapper_filename
    filename = get_redmapper_filename(config, redmapper_name, paths=(config.get('plotpath'),),
                                      withversion=withversion, filetype='png')
    fig.savefig(filename)
    plt.close(fig)
    return filename

def plot_redmagic_nz(config, cat, name, eta, n0, areastr, sample=True,
                     zrange=None, calib_zrange=None, extraname=None,
                     withversion=False, binsize=0.02):
    """
    Plot the n(z) for a redmagic catalog.
    """
    zsamp = cat.zredmagic_samp
    if zsamp.ndim > 1:
        zsamp = zsamp[:, 0]
    if zrange is None:
        zrange = config.get('redmagic_zrange')
    if extraname is None:
        redmapper_name = 'redmagic_%s_%3.1f-%02d_nz' % (name, eta, int(n0))
    else:
        redmapper_name = 'redmagic_%s_%s_%3.1f-%02d_nz' % (extraname, name, eta, int(n0))

    return plot_nz(config, zsamp, areastr, zrange,
                   xlabel=r'$z_{\mathrm{redmagic}}$',
                   ylabel=r'$n\,(1e4\,\mathrm{galaxies} / \mathrm{Mpc}^{3})$',
                   title='%s: %3.1f-%02d' % (name, eta, int(n0)),
                   redmapper_name=redmapper_name,
                   calib_zrange=calib_zrange, withversion=withversion, binsize=binsize)


