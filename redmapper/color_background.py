"""Color-only galaxy background class for redmapper.

This file contains classes to describe the b(x) background terms using colors only.
"""
import fitsio
import numpy as np
import esutil
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .catalog import Entry
from .utilities import interpol
from . import depthmap
from .utilities import cic
from .galaxy import GalaxyCatalog
from .logger import logger


def read_color_background(filename, usehdrarea=False):
    """
    Instantiate a color background

    Parameters
    ----------
    filename: `string`
       Color background filename
    usehdrarea: `bool`, optional
       Use area from the header instead of as configured.  Default is False.
    """

    refmagbinsize = 0.01
    colbinsize = 0.01
    area = 1.0

    fits = fitsio.FITS(filename)

    bkgs = {}

    started = False

    for ext in fits[1: ]:
        extname = ext.get_extname()

        parts = extname.split('_')
        iind = int(parts[0])
        jind = int(parts[1])

        key = iind * 100 + jind

        obkg = Entry.from_fits_ext(ext)

        # Create the refmag bins
        refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], refmagbinsize)
        nrefmagbins = refmagbins.size

        # Deal with area if necessary
        if not started:
            if usehdrarea:
                if 'areas' in obkg.dtype.names:
                    areas = interpol(obkg.areas, obkg.refmagbins, refmagbins)
                else:
                    hdr = ext.read_header()
                    areas = np.zeros(nrefmagbins) + hdr['AREA']
            else:
                areas = np.zeros(nrefmagbins) + area

            started = True

        if iind == jind:
            # We are on a diagonal

            # First do the refmag
            ncolbins = obkg.colbins.size
            bc_new = np.zeros((nrefmagbins, ncolbins))
            for i in range(ncolbins):
                bc_new[:, i] = np.clip(interpol(obkg.bc[:, i], obkg.refmagbins, refmagbins), 0.0, None)

            bc = bc_new.copy()

            # Now do the color
            colbins = np.arange(obkg.colrange[0], obkg.colrange[1], colbinsize)
            ncolbins = colbins.size

            bc_new = np.zeros((nrefmagbins, ncolbins))
            for j in range(nrefmagbins):
                bc_new[j, :] = np.clip(interpol(bc[j, :], obkg.colbins, colbins), 0.0, None)

            n = np.sum(bc, axis=1, dtype=np.float64) * (colbinsize / obkg.colbinsize)

            sigma_g = bc_new.copy()
            for j in range(ncolbins):
                sigma_g[:, j] = bc_new[:, j] / areas

            bkgs[key] = {'col1': iind,
                              'col2': jind,
                              'refmagindex': 1,
                              'colbins': colbins,
                              'colrange': obkg.colrange,
                              'colbinsize': colbinsize,
                              'refmagbins': refmagbins,
                              'refmagrange': obkg.refmagrange,
                              'refmagbinsize': refmagbinsize,
                              'bc': bc_new,
                              'n': n,
                              'sigma_g': sigma_g}
        else:
            # We are on an off-diagonal

            # start with the refmag
            ncol1bins = obkg.col1bins.size
            ncol2bins = obkg.col2bins.size
            bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
            for i in range(ncol1bins):
                for j in range(ncol2bins):
                    bc_new[:, j, i] = np.clip(interpol(obkg.bc[:, j, i], obkg.refmagbins, refmagbins), 0.0, None)

            bc = bc_new.copy()

            # color1
            col1bins = np.arange(obkg.col1range[0], obkg.col1range[1], colbinsize)
            ncol1bins = col1bins.size

            bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
            for j in range(ncol2bins):
                for k in range(nrefmagbins):
                    bc_new[k, j, :] = np.clip(interpol(bc[k, j, :], obkg.col1bins, col1bins), 0.0, None)

            bc = bc_new.copy()

            col2bins = np.arange(obkg.col2range[0], obkg.col2range[1], colbinsize)
            ncol2bins = col2bins.size

            bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
            for i in range(ncol1bins):
                for k in range(nrefmagbins):
                    bc_new[k, :, i] = np.clip(interpol(bc[k, :, i], obkg.col2bins, col2bins), 0.0, None)

            temp = np.sum(bc_new, axis=1, dtype=np.float64) * colbinsize
            n = np.sum(temp, axis=1, dtype=np.float64) * colbinsize

            sigma_g = bc_new.copy()

            for j in range(ncol1bins):
                for k in range(ncol2bins):
                    sigma_g[:, k, j] = bc_new[:, k, j] / areas

            bkgs[key] = {'col1': iind,
                              'col2': jind,
                              'refmag_index': 2,
                              'col1bins': col1bins,
                              'col1range': obkg.col1range,
                              'col1binsize': colbinsize,
                              'col2bins': col2bins,
                              'col2range': obkg.col2range,
                              'col2binsize': colbinsize,
                              'refmagbins': refmagbins,
                              'refmagrange': obkg.refmagrange,
                              'refmagbinsize': refmagbinsize,
                              'bc': bc_new,
                              'n': n,
                              'sigma_g': sigma_g}

    return bkgs

def sigma_g_diagonal(color_background_data, bkg_index, colors, refmags):
    """
    Compute sigma_g(color, refmag) for a diagonal (single-color) background.

    Parameters
    ----------
    bkg_index: `int`
       Color index along the diagonal
    colors: `np.array`
       Float array of colors
    refmags: `np.array`
       Float array of reference magnitudes

    Returns
    -------
    sigma_g: `np.array`
       Sigma_g(x) for input values
    """
    bkg = color_background_data[bkg_index * 100 + bkg_index]

    colindex = np.searchsorted(bkg['colbins'], colors - bkg['colbinsize'])
    refmagindex = np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize'])
    # check for overruns
    badcol, = np.where((colindex < 0) | (colindex >= bkg['colbins'].size))
    colindex[badcol] = 0
    badrefmag, = np.where((refmagindex < 0) | (refmagindex >= bkg['refmagbins'].size))
    refmagindex[badrefmag] = 0

    sigma_g = bkg['sigma_g'][refmagindex, colindex]
    sigma_g[badcol] = np.inf
    sigma_g[badrefmag] = np.inf
    badcombo, = np.where(sigma_g == 0.0)
    sigma_g[badcombo] = np.inf

    return sigma_g

def lookup_diagonal(color_background_data, bkg_index, colors, refmags, doRaise=True):
    """
    Look up the normalized background value b(color, refmag) for a diagonal
    (single-color) background.

    Parameters
    ----------
    bkg_index: `int`
       Color index along the diagonal
    colors: `np.array`
       Float array of colors
    refmags: `np.array`
       Float array of reference magnitudes
    doRaise: `bool`, optional
       Raise exception if color background does not exist.  Default is True.

    Returns
    -------
    b: `np.array`
       Background(x) for input values
    """
    try:
        bkg = color_background_data[bkg_index * 100 + bkg_index]
    except KeyError:
        if doRaise:
            raise KeyError("Could not find a color background for %d" % (bkg_index))
        else:
            return np.zeros(colors.size)

    refmagindex = np.clip(np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize']), 0, bkg['refmagbins'].size - 1)
    col_index = np.clip(np.searchsorted(bkg['colbins'], colors - bkg['colbinsize']), 0, bkg['colbins'].size - 1)

    return bkg['bc'][refmagindex, col_index] / bkg['n'][refmagindex]

def get_colrange(color_background_data, bkg_index):
    """
    Get the background color range for a given background index.

    Parameters
    ----------
    bkg_index: `int`
       Color index along the diagonal

    Returns
    -------
    range: `np.array`
       2-element float array with color min, max
    """
    bkg = color_background_data[bkg_index * 100 + bkg_index]
    return bkg['colrange']

def lookup_offdiag(color_background_data, bkg_index1, bkg_index2, colors1, colors2, refmags, doRaise=True):
    """
    Look up the normalized background value b(color1, color2, refmag) for
    an off-diagonal (two-color) background.

    Parameters
    ----------
    bkg_index1: `int`
       Color index for color 1
    bkg_index2: `int`
       Color index for color 2
    colors1: `np.array`
       Float array of colors (1)
    colors2: `np.array`
       Float array of colors (2)
    refmags: `np.array`
       Float array of reference mags
    doRaise: `bool`, optional
       Raise exception if color background does not exist.  Default is True.
    """
    key = bkg_index1 * 100 + bkg_index2
    if key not in color_background_data:
        key = bkg_index2 * 100 + bkg_index1

    try:
        bkg = color_background_data[key]
    except KeyError:
        if doRaise:
            raise KeyError("Could not find a color background for %d, %d" % (bkg_index1, bkg_index1))
        else:
            return np.zeros(colors1.size)

    refmagindex = np.clip(np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize']), 0, bkg['refmagbins'].size - 1)
    col_index1 = np.clip(np.searchsorted(bkg['col1bins'], colors1 - bkg['col1binsize']), 0, bkg['col1bins'].size - 1)
    col_index2 = np.clip(np.searchsorted(bkg['col2bins'], colors2 - bkg['col2binsize']), 0, bkg['col2bins'].size - 1)

    return bkg['bc'][refmagindex, col_index2, col_index1] / bkg['n'][refmagindex]

def generate_color_background(config, minrangecheck=1000, clobber=False):
    """
    Generate the color galaxyt background.  The output filename is
    specified in config.bkgfile_color.

    Parameters
    ----------
    clobber: `bool`, optional
       Overwrite any existing config.bkgfile_color file.  Default is False.
    """

    # Check if it's already there...
    if not clobber and os.path.isfile(config.bkgfile_color):
        logger.info("Found %s and clobber is False" % (config.bkgfile_color))
        return

    # read in the galaxies
    gals = GalaxyCatalog.from_galfile(config.galfile,
                                      nside=config.nside,
                                      hpix=config.hpix,
                                      border=config.border)

    # Generate ranges based on the data
    refmagbinsize = 0.05

    refmagrange = np.array([12.0, config.limmag_ref])

    nmag = config.nmag
    ncol = nmag - 1

    col = gals.galcol

    colrange_default = np.array([-2.0, 5.0])

    colranges = np.zeros((2, ncol))
    colbinsize = 0.02
    for i in range(ncol):
        use, = np.where((col[:, i] > colrange_default[0]) &
                        (col[:, i] < colrange_default[1]) &
                        (gals.refmag < (config.limmag_ref - 0.5)))

        h = esutil.stat.histogram(col[use, i], min=colrange_default[0],
                                  max=colrange_default[1], binsize=colbinsize)
        bins = np.arange(h.size) * colbinsize + colrange_default[0]

        good, = np.where(h > minrangecheck)

        colranges[0, i] = np.min(bins[good])
        colranges[1, i] = np.max(bins[good]) + colbinsize

    nrefmag = np.ceil((refmagrange[1] - refmagrange[0]) / refmagbinsize).astype(np.int32)
    refmagbins = np.arange(nrefmag) * refmagbinsize + refmagrange[0]

    if config.depthfile is not None:
        depth_data = depthmap.read_depth_map(config)
        areas = depthmap.compute_areas(depth_data, refmagbins)
    else:
        areas = np.zeros(refmagbins.size) + config.area

    # Prepare QA plotting
    do_qa = getattr(config, 'more_qa_plots', False)
    if do_qa:
        plotpath = getattr(config, 'plotpath', '.')
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
        logger.info("QA plots will be saved to %s" % plotpath)

    for i in range(ncol):
        for j in range(i, ncol):
            if (i == j):
                # diagonal
                ncoldiag = np.ceil((colranges[1, i] - colranges[0, i]) / colbinsize).astype(np.int32)
                coldiagbins = np.arange(ncoldiag) * colbinsize + colranges[0, i]
                binsizes = refmagbinsize * colbinsize

                bad = ((col[:, i] < colranges[0, i]) |
                       (col[:, i] >= colranges[1, i]) |
                       (gals.refmag < refmagrange[0]) |
                       (gals.refmag >= refmagrange[1]))

                colpos = (col[~bad, i] - colranges[0, i]) * ncoldiag / (colranges[1, i] - colranges[0, i])
                refmagpos = (gals[~bad].refmag - refmagrange[0]) * nrefmag / (refmagrange[1] - refmagrange[0])

                value = np.ones(np.sum(~bad))
                logger.info("Running CIC on %d, %d" % (i, j))
                field = cic(value, colpos, ncoldiag, refmagpos, nrefmag, isolated=True)
                # need to fix cic...
                bad = ~np.isfinite(field)
                field[bad] = 0.0

                bc = field.astype(np.float32) / binsizes
                n = np.sum(bc, axis=1, dtype=np.float64) * colbinsize

                # QA plot for diagonal
                if do_qa:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Plot 1: bc as 2D histogram
                    im1 = axes[0, 0].imshow(bc.T, aspect='auto', origin='lower',
                                            extent=[refmagbins[0], refmagbins[-1],
                                                   coldiagbins[0], coldiagbins[-1]],
                                            cmap='Reds', norm=LogNorm(vmin=bc[bc>0].min() if np.any(bc>0) else 1e-10))
                    axes[0, 0].set_xlabel('refmag')
                    axes[0, 0].set_ylabel('color_%d' % i)
                    axes[0, 0].set_title('Background bc (color %d)' % i)
                    plt.colorbar(im1, ax=axes[0, 0], label='density')
                    
                    # Plot 2: n vs refmag
                    axes[0, 1].plot(refmagbins, n, 'b-')
                    axes[0, 1].set_xlabel('refmag')
                    axes[0, 1].set_ylabel('n (integrated counts)')
                    axes[0, 1].set_title('Integrated counts vs refmag')
                    axes[0, 1].grid(alpha=0.3)
                    
                    # Plot 3: slice at median refmag
                    mid_refmag = len(refmagbins) // 2
                    axes[1, 0].plot(coldiagbins, bc[mid_refmag, :], 'r-')
                    axes[1, 0].set_xlabel('color_%d' % i)
                    axes[1, 0].set_ylabel('bc')
                    axes[1, 0].set_title('Color slice at refmag=%.2f' % refmagbins[mid_refmag])
                    axes[1, 0].grid(alpha=0.3)
                    
                    # Plot 4: areas vs refmag
                    axes[1, 1].plot(refmagbins, areas, 'g-')
                    axes[1, 1].set_xlabel('refmag')
                    axes[1, 1].set_ylabel('area (deg²)')
                    axes[1, 1].set_title('Survey area vs refmag')
                    axes[1, 1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    qafile = os.path.join(plotpath, 'colorbkg_diag_%02d.png' % i)
                    plt.savefig(qafile, dpi=300)
                    plt.close()
                    logger.info("Saved QA plot: %s" % qafile)

                outstr = np.zeros(1, dtype=[('COL1', 'i2'),
                                            ('COL2', 'i2'),
                                            ('COL_INDEX', 'i2'),
                                            ('REFMAG_INDEX', 'i2'),
                                            ('COLBINS', 'f4', coldiagbins.size),
                                            ('COLRANGE', 'f4', 2),
                                            ('COLBINSIZE', 'f4'),
                                            ('REFMAGBINS', 'f4', refmagbins.size),
                                            ('REFMAGRANGE', 'f4', 2),
                                            ('REFMAGBINSIZE', 'f4'),
                                            ('AREAS', 'f4', areas.size),
                                            ('BC', 'f4', bc.shape),
                                            ('N', 'f4', n.size)])

                outstr['COL1'] = i
                outstr['COL2'] = j
                outstr['COL_INDEX'] = 0
                outstr['REFMAG_INDEX'] = 1
                outstr['COLBINS'][:] = coldiagbins
                outstr['COLRANGE'][:] = colranges[:, i]
                outstr['COLBINSIZE'] = colbinsize
                outstr['REFMAGBINS'][:] = refmagbins
                outstr['REFMAGRANGE'][:] = refmagrange
                outstr['REFMAGBINSIZE'] = refmagbinsize
                outstr['AREAS'][:] = areas
                outstr['BC'][:, :] = bc
                outstr['N'][:] = n

            else:
                # off-diagonal

                ncol1 = np.ceil((colranges[1, i] - colranges[0, i]) / colbinsize).astype(np.int32)
                col1bins = np.arange(ncol1) * colbinsize + colranges[0, i]
                ncol2 = np.ceil((colranges[1, j] - colranges[0, j]) / colbinsize).astype(np.int32)
                col2bins = np.arange(ncol2) * colbinsize + colranges[0, j]

                binsizes = refmagbinsize * colbinsize * colbinsize

                bad = ((col[:, i] < colranges[0, i]) |
                       (col[:, i] >= colranges[1, i]) |
                       (col[:, j] < colranges[0, j]) |
                       (col[:, j] >= colranges[1, j]) |
                       (gals.refmag < refmagrange[0]) |
                       (gals.refmag >= refmagrange[1]))

                col1pos = (col[~bad, i] - colranges[0, i]) * ncol1 / (colranges[1, i] - colranges[0, i])
                col2pos = (col[~bad, j] - colranges[0, j]) * ncol2 / (colranges[1, j] - colranges[0, j])
                refmagpos = (gals[~bad].refmag - refmagrange[0]) * nrefmag / (refmagrange[1] - refmagrange[0])

                value = np.ones(np.sum(~bad))
                logger.info("Running CIC on %d, %d" % (i, j))
                field = cic(value, col1pos, ncol1, col2pos, ncol2, refmagpos, nrefmag, isolated=True)
                bad = ~np.isfinite(field)
                field[bad] = 0.0

                bc = field.astype(np.float32) / binsizes
                temp = np.sum(bc, axis=2, dtype=np.float64) * colbinsize
                n = np.sum(temp, axis=1, dtype=np.float64) * colbinsize

                # QA plot for off-diagonal
                if do_qa:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Plot 1: bc slice at median refmag (color1 vs color2)
                    mid_refmag = len(refmagbins) // 2
                    im1 = axes[0, 0].imshow(bc[mid_refmag, :, :].T, aspect='auto', origin='lower',
                                            extent=[col2bins[0], col2bins[-1],
                                                   col1bins[0], col1bins[-1]],
                                            cmap='Reds', norm=LogNorm(vmin=bc[bc>0].min() if np.any(bc>0) else 1e-10))
                    axes[0, 0].set_xlabel('color_%d' % j)
                    axes[0, 0].set_ylabel('color_%d' % i)
                    axes[0, 0].set_title('Background at refmag=%.2f' % refmagbins[mid_refmag])
                    plt.colorbar(im1, ax=axes[0, 0], label='density')
                    
                    # Plot 2: n vs refmag
                    axes[0, 1].plot(refmagbins, n, 'b-')
                    axes[0, 1].set_xlabel('refmag')
                    axes[0, 1].set_ylabel('n (integrated counts)')
                    axes[0, 1].set_title('Integrated counts vs refmag')
                    axes[0, 1].grid(alpha=0.3)
                    
                    # Plot 3: color1 marginal at median refmag
                    marginal_col1 = np.sum(bc[mid_refmag, :, :], axis=0)
                    axes[1, 0].plot(col1bins, marginal_col1, 'r-')
                    axes[1, 0].set_xlabel('color_%d' % i)
                    axes[1, 0].set_ylabel('marginal density')
                    axes[1, 0].set_title('Color %d marginal' % i)
                    axes[1, 0].grid(alpha=0.3)
                    
                    # Plot 4: color2 marginal at median refmag
                    marginal_col2 = np.sum(bc[mid_refmag, :, :], axis=1)
                    axes[1, 1].plot(col2bins, marginal_col2, 'g-')
                    axes[1, 1].set_xlabel('color_%d' % j)
                    axes[1, 1].set_ylabel('marginal density')
                    axes[1, 1].set_title('Color %d marginal' % j)
                    axes[1, 1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    qafile = os.path.join(plotpath, 'colorbkg_offdiag_%02d_%02d.png' % (i, j))
                    plt.savefig(qafile, dpi=300)
                    plt.close()
                    logger.info("Saved QA plot: %s" % qafile)

                outstr = np.zeros(1, dtype=[('COL1', 'i2'),
                                            ('COL2', 'i2'),
                                            ('COL1_INDEX', 'i2'),
                                            ('COL2_INDEX', 'i2'),
                                            ('REFMAG_INDEX', 'i2'),
                                            ('COL1BINS', 'f4', col1bins.size),
                                            ('COL1RANGE', 'f4', 2),
                                            ('COL1BINSIZE', 'f4'),
                                            ('COL2BINS', 'f4', col2bins.size),
                                            ('COL2RANGE', 'f4', 2),
                                            ('COL2BINSIZE', 'f4'),
                                            ('REFMAGBINS', 'f4', refmagbins.size),
                                            ('REFMAGRANGE', 'f4', 2),
                                            ('REFMAGBINSIZE', 'f4'),
                                            ('AREAS', 'f4', areas.size),
                                            ('BC', 'f4', bc.shape),
                                            ('N', 'f4', n.size)])

                outstr['COL1'] = i
                outstr['COL2'] = j
                outstr['COL1_INDEX'] = 0
                outstr['COL2_INDEX'] = 1
                outstr['REFMAG_INDEX'] = 2
                outstr['COL1BINS'][:] = col1bins
                outstr['COL1RANGE'][:] = colranges[:, i]
                outstr['COL1BINSIZE'] = colbinsize
                outstr['COL2BINS'][:] = col2bins
                outstr['COL2RANGE'][:] = colranges[:, j]
                outstr['COL2BINSIZE'] = colbinsize
                outstr['REFMAGBINS'][:] = refmagbins
                outstr['REFMAGRANGE'][:] = refmagrange
                outstr['REFMAGBINSIZE'] = refmagbinsize
                outstr['AREAS'][:] = areas
                outstr['BC'][:, :] = bc
                outstr['N'][:] = n


            extname = "%02d_%02d_REF" % (i, j)
            hdr = fitsio.FITSHDR()
            hdr['AREA'] = config.area
            if i == 0 and j == 0:
                startClobber = True
            else:
                startClobber = False

            fitsio.write(config.bkgfile_color, outstr, extname=extname, header=hdr, clobber=startClobber)



