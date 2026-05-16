"""Classes and functions for describing a galaxy catalog for redmapper.

This module contains the logic for reading, using, and making galaxy tables,
transitioning to a functional style using astropy.table.Table.
"""
import fitsio
import esutil
from esutil.htm import Matcher
import numpy as np
import hpgeom as hpg
import hpgeom.healpy_compat as hpc
import os
import glob
import re
import tempfile
import copy
from collections.abc import Iterable
from astropy.table import Table, Row

from .catalog import Entry, Catalog
from .mask import get_mask, get_mask_values
from .utilities import decode_string, make_lockfile


def zred_extra_dtype(nsamp):
    """
    Return the zred dtype to append.

    Parameters
    ----------
    nsamp: `int`
       Number of samples of zred to record
    """

    return [('ZRED', 'f4'),
            ('ZRED_E', 'f4'),
            ('ZRED2', 'f4'),
            ('ZRED2_E', 'f4'),
            ('ZRED_UNCORR', 'f4'),
            ('ZRED_UNCORR_E', 'f4'),
            ('ZRED_SAMP', 'f4', nsamp),
            ('LKHD', 'f4'),
            ('CHISQ', 'f4')]

def compute_colors(mag_array):
    """
    Compute galaxy colors from magnitudes.

    Parameters
    ----------
    mag_array: `np.ndarray`
        Magnitude array of shape (ngal, nmag) or (nmag,)

    Returns
    -------
    colors: `np.ndarray`
        Color array of shape (ngal, nmag-1) or (nmag-1,)
    """
    if mag_array.ndim == 1:
        return mag_array[:-1] - mag_array[1:]
    return mag_array[:, :-1] - mag_array[:, 1:]

def compute_color_errors(mag_err_array):
    """
    Compute galaxy color errors from magnitude errors.

    Parameters
    ----------
    mag_err_array: `np.ndarray`
        Magnitude error array of shape (ngal, nmag) or (nmag,)

    Returns
    -------
    color_errors: `np.ndarray`
        Color error array of shape (ngal, nmag-1) or (nmag-1,)
    """
    if mag_err_array.ndim == 1:
        return np.sqrt(mag_err_array[:-1]**2. + mag_err_array[1:]**2.)
    return np.sqrt(mag_err_array[:, :-1]**2. + mag_err_array[:, 1:]**2.)

def get_subpixel_indices(galtable, hpix=[], border=0.0, nside=0):
    """
    Routine to get subpixel indices from a galaxy table.

    Parameters
    ----------
    galtable: `astropy.table.Table` or `redmapper.Entry`
       A redmapper galaxy table master catalog
    hpix: `list`, optional
       Healpix number (ring format) of sub-region.  Default is [] (full catalog)
    border: `float`, optional
       Border around hpix (in degrees) to find pixels.  Default is 0.0.
       Only works if hpix is a single-length list
    nside: `int`, optional
       Nside of healpix subregion.  Default is 0 (full catalog).

    Returns
    -------
    indices: `np.array`
       Integer array of indices of galaxy table pixels in the subregion.
    """
    if not isinstance(hpix, Iterable):
        _hpix = [hpix]
    else:
        _hpix = hpix

    def _get(obj, key):
        if isinstance(obj, Table):
            return obj[key]
        return getattr(obj, key)

    filenames = _get(galtable, 'filenames')

    if len(_hpix) == 0 or nside == 0:
        return np.arange(len(filenames))

    theta, phi = hpg.pixel_to_angle(_get(galtable, 'nside'), _get(galtable, 'hpix'), lonlat=False, nest=False)
    ipring_big = hpg.angle_to_pixel(nside, theta, phi, lonlat=False, nest=False)

    _, indices = esutil.numpy_util.match(_hpix, ipring_big)

    # Ignore border if using full catalog
    if border > 0.0 and len(_hpix) > 0:
        if len(_hpix) != 1:
            raise NotImplementedError("Cannot do boundary around a pixel list.")

        # now we need to find the extra boundary...
        boundaries = hpc.boundaries(nside, _hpix[0], step=_get(galtable, 'nside') // nside)
        inhpix = _get(galtable, 'hpix')[indices]
        for i in range(boundaries.shape[1]):
            pixint = hpc.query_disc(_get(galtable, 'nside'), boundaries[:, i],
                                    border*np.pi/180., inclusive=True, fact=8)
            inhpix = np.append(inhpix, pixint)
        inhpix = np.unique(inhpix)
        _, indices = esutil.numpy_util.match(inhpix, _get(galtable, 'hpix'))

    return indices

class Galaxy(Entry):
    """
    Class to describe a single galaxy, based on Entry (astropy Row).
    """
    @property
    def galcol(self):
        """Get the array of galaxy colors."""
        return compute_colors(self['mag'])

class GalaxyCatalog(Catalog):
    """
    Class to describe a redmapper galaxy Catalog, based on Catalog (astropy Table).
    """
    _RowClass = Galaxy
    RowClass = Galaxy

    def __init__(self, *args, **kwargs):
        depth = kwargs.pop('depth', 10)
        super().__init__(*args, **kwargs)
        self._RowClass = Galaxy
        self.RowClass = Galaxy
        self._htm_matcher = None
        self.depth = depth

    @property
    def galcol(self):
        """Get the array of galaxy colors."""
        return compute_colors(self['mag'])

    @property
    def galcol_err(self):
        """Get the array of galaxy color errors."""
        return compute_color_errors(self['mag_err'])

    def add_zred_fields(self, nsamp):
        """Add zred fields (compatibility method)."""
        self.add_fields(zred_extra_dtype(nsamp))

    @classmethod
    def from_galfile(cls, filename, zredfile=None, nside=0, hpix=[], border=0.0, truth=False,
                     use_tempfile=False, refmag_range=[-1000.0, 1000.0], chisq_max=1e30, zspec=False):
        """
        Generate a GalaxyCatalog from a redmapper "galfile."
        """
        if zredfile is not None:
            use_zred = True
        else:
            use_zred = False

        if not isinstance(hpix, Iterable):
            _hpix = [hpix]
        else:
            _hpix = hpix

        if len(_hpix) > 0 and nside == 0:
            raise ValueError("If hpix is specified, must also specify nside")
        if border < 0.0:
            raise ValueError("Border must be >= 0.0.")
        
        if nside > 0:
            npix = hpg.nside_to_npixel(int(nside))
            if len(_hpix) > 0:
                for _hp in _hpix:
                    if (_hp < 0) or (_hp >= npix):
                        raise ValueError("hpix %d is out of range." % (_hp))

        if border > 0.0 and len(_hpix) > 1:
            raise NotImplementedError("Cannot read a boundary around a pixel list.")

        hdr = fitsio.read_header(filename, ext=1)
        pixelated = hdr.get("PIXELS", 0)
        fitsformat = hdr.get("FITS", 0)

        if use_zred:
            zhdr = fitsio.read_header(zredfile, ext=1)
            zpixelated = zhdr.get("PIXELS", 0)

        if not pixelated:
            cat = fitsio.read(filename, ext=1, upper=True, lower=True)
            if use_zred:
                zcat = fitsio.read(zredfile, ext=1, upper=True, lower=True)
                if zcat.size != cat.size:
                    raise ValueError("zredfile different length than catfile")
                # Merge catalogs
                for name in zcat.dtype.names:
                    if name not in cat.dtype.names:
                        cat = esutil.numpy_util.add_fields(cat, [(name, zcat.dtype[name])])
                        cat[name] = zcat[name]
                return cls(cat)
            else:
                return cls(cat)
        else:
            if use_zred and not zpixelated:
                raise ValueError("galfile is pixelated but zredfile is not")

        if not fitsformat:
            raise ValueError("Input galfile must describe fits files.")

        tab = Entry.from_fits_file(filename, ext=1)
        nside_tab = tab.nside
        if nside > nside_tab:
            raise ValueError("Requested nside (%d) > table nside (%d)." % (nside, nside_tab))

        if use_zred:
            ztab = Entry.from_fits_file(zredfile, ext=1)
            zpath = os.path.dirname(zredfile)

        path = os.path.dirname(os.path.abspath(filename))
        indices = get_subpixel_indices(tab, hpix=_hpix, border=border, nside=nside)

        if use_zred:
            mark = np.zeros(indices.size, dtype=bool)
            for i, f in enumerate(ztab.filenames[indices]):
                fname = os.path.join(zpath, f if isinstance(f, str) else decode_string(f))
                if os.path.isfile(fname):
                    mark[i] = True
            bad, = np.where(~mark)
            if bad.size == indices.size:
                raise ValueError("No zred files associated with galaxy pixels.")
            indices = np.delete(indices, bad)

        trim_border = False
        if len(_hpix) == 1 and nside > 0 and border > 0.0:
            trim_border = True
            nside_cutref = 512
            boundaries = hpc.boundaries(nside, _hpix[0], step=nside_cutref//nside)
            bit_shift = 2*int(np.round(np.log2(nside_cutref/nside)))
            inhpix_nest = np.arange(2**bit_shift, dtype=np.int32) + np.left_shift(hpg.ring_to_nest(nside, _hpix[0]), bit_shift)
            inhpix = hpg.nest_to_ring(nside_cutref, inhpix_nest)
            for i in range(boundaries.shape[1]):
                pixint = hpc.query_disc(nside_cutref, boundaries[:, i], np.radians(border), inclusive=True, fact=8)
                inhpix = np.append(inhpix, pixint)
            inhpix = np.unique(inhpix)

        first_fname = os.path.join(path, tab.filenames[indices[0]] if isinstance(tab.filenames[indices[0]], str) else decode_string(tab.filenames[indices[0]]))
        elt = fitsio.read(first_fname, ext=1, rows=0, lower=True)
        dtype_in = elt.dtype.descr
        if not truth:
            mark = [dt[0] not in ('ztrue', 'm200', 'central', 'halo_id') for dt in dtype_in]
            dtype = [dt for i, dt in enumerate(dtype_in) if mark[i]]
            columns = [dt[0] for dt in dtype]
        else:
            dtype = list(dtype_in)
            columns = None

        cat_fields = [dt[0] for dt in dtype]

        if use_zred:
            fname = os.path.join(zpath, ztab.filenames[indices[0]] if isinstance(ztab.filenames[indices[0]], str) else decode_string(ztab.filenames[indices[0]]))
            zelt = fitsio.read(fname, ext=1, rows=0, lower=True)
            zcat_fields = [dt[0] for dt in zelt.dtype.descr]
            dtype.extend(zelt.dtype.descr)

        if use_tempfile:
            fd, tempFile = tempfile.mkstemp(suffix='.fits')
            os.close(fd)
            tempfits = fitsio.FITS(tempFile, mode='rw', clobber=True)
            tempfits.create_table_hdu(dtype=dtype)
        else:
            final_cat_array = np.zeros(np.sum(tab.ngals[indices]), dtype=dtype)

        ctr = 0
        for index in indices:
            fname = os.path.join(path, tab.filenames[index] if isinstance(tab.filenames[index], str) else decode_string(tab.filenames[index]))
            if use_tempfile:
                tempcat = np.zeros(tab.ngals[index], dtype=dtype)
                tempcat[cat_fields][:] = fitsio.read(fname, ext=1, lower=True, columns=columns)
            else:
                final_cat_array[cat_fields][ctr: ctr + tab.ngals[index]] = fitsio.read(fname, ext=1, lower=True, columns=columns)

            if use_zred:
                fname = os.path.join(zpath, ztab.filenames[index] if isinstance(ztab.filenames[index], str) else decode_string(ztab.filenames[index]))
                if use_tempfile:
                    tempcat[zcat_fields][:] = fitsio.read(fname, ext=1, lower=True)
                else:
                    final_cat_array[zcat_fields][ctr: ctr + tab.ngals[index]] = fitsio.read(fname, ext=1, lower=True)

            if use_tempfile:
                guse = ((tempcat['refmag'] > refmag_range[0]) & (tempcat['refmag'] < refmag_range[1]))
                if use_zred:
                    guse &= (tempcat['chisq'] < chisq_max)
                if guse.sum() > 0:
                    if trim_border:
                        ipring = hpg.angle_to_pixel(nside_cutref, tempcat['ra'], tempcat['dec'], nest=False)
                        _, matches = esutil.numpy_util.match(inhpix, ipring[guse])
                        tempfits[1].append(tempcat[guse][matches])
                    else:
                        tempfits[1].append(tempcat[guse])
            ctr += tab.ngals[index]

        if use_tempfile:
            tempfits.close()
            final_cat_array = fitsio.read(tempFile, ext=1)
            os.remove(tempFile)
            return cls(final_cat_array)
        else:
            if trim_border:
                ipring = hpg.angle_to_pixel(nside_cutref, final_cat_array['ra'], final_cat_array['dec'], nest=False)
                _, matches = esutil.numpy_util.match(inhpix, ipring)
                return cls(final_cat_array[matches])
            return cls(final_cat_array)

    def match_one(self, ra, dec, radius):
        """Match one ra/dec position to the galaxy catalog."""
        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self['ra'], self['dec'])
        _, indices, dists = self._htm_matcher.match(ra, dec, radius, maxmatch=0)
        return indices, dists

    def match_many(self, ras, decs, radius, maxmatch=0):
        """Match many ra/dec positions to the galaxy catalog."""
        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self['ra'], self['dec'])
        return self._htm_matcher.match(ras, decs, radius, maxmatch=maxmatch)

class FakeMaskConfig(object):
    """A simple fake config to read in a mask."""
    def __init__(self, maskfile, mask_mode):
        self.maskfile = maskfile
        self.mask_mode = mask_mode
        class TempD(object):
            def __init__(self):
                self.hpix = 0
                self.nside = 0
        self.d = TempD()

class GalaxyCatalogMaker(object):
    """Class to generate a redmapper galaxy catalog from an input catalog."""

    def __init__(self, outbase, info_dict, nside=32, maskfile=None, mask_mode=0,
                 parallel=False, generate_unique_ids=False, ingest_truth=False, ingest_zspec=False):
        self.parallel = parallel
        self.generate_unique_ids = generate_unique_ids
        self.outbase = outbase
        self.nside = nside
        self.lim_ref = info_dict['LIM_REF']
        self.ref_ind = info_dict['REF_IND']
        self.area = info_dict['AREA']
        self.nmag = info_dict['NMAG']
        self.mode = info_dict['MODE']
        self.zeropoint = info_dict['ZP']
        self.b = info_dict.get('B', np.zeros(self.nmag))
        self.u_ind = info_dict.get('U_IND')
        self.g_ind = info_dict.get('G_IND')
        self.r_ind = info_dict.get('R_IND')
        self.i_ind = info_dict.get('I_IND')
        self.z_ind = info_dict.get('Z_IND')
        self.y_ind = info_dict.get('Y_IND')

        self.filename = '%s_master_table.fit' % (self.outbase)
        if os.path.basename(self.filename) == self.filename:
            raise RuntimeError("outbase %s must contain a path" % (self.outbase))

        self.outpath = os.path.dirname(self.filename)
        self.outbase_nopath = os.path.basename(self.outbase)

        if not os.path.exists(self.outpath):
            try:
                os.makedirs(self.outpath)
            except FileExistsError:
                pass

        if os.path.isfile(self.filename):
            raise RuntimeError("Final file %s already exists." % (self.filename))

        self.ngals = np.zeros(hpg.nside_to_npixel(int(self.nside)), dtype=np.int32)
        self.mask = None
        if maskfile is not None:
            fake_config = FakeMaskConfig(maskfile, mask_mode)
            self.mask = get_mask(fake_config, include_maskgals=False)

        self.is_finalized = False
        self.ingest_truth = ingest_truth
        self.ingest_zspec = ingest_zspec

    def split_galaxies(self, gals):
        """Split a full galaxy catalog into pixels."""
        if self.is_finalized:
            raise RuntimeError("Catalog already finalized.")
        self.append_galaxies(gals)
        self.finalize_catalog()

    def append_galaxies(self, gals):
        """Append a set of galaxies to a galaxy catalog."""
        if self.is_finalized:
            raise RuntimeError("Catalog already finalized.")
        
        # Convert to numpy array if it's a Table for checking
        if isinstance(gals, Table):
            gals_arr = gals.as_array()
        else:
            gals_arr = gals

        self._check_galaxies(gals_arr)

        if self.mask is not None:
            good = get_mask_values(self.mask['mask_data'], gals_arr['ra'], gals_arr['dec'], rng=self.mask['rng'], config=self.mask['config'])
            gals_arr = gals_arr[good]

        ipring = hpg.angle_to_pixel(self.nside, gals_arr['ra'], gals_arr['dec'], nest=False)
        h, rev = esutil.stat.histogram(ipring, min=0, max=self.ngals.size - 1, rev=True)

        gdpix, = np.where(h > 0)
        for pix in gdpix:
            i1a = rev[rev[pix]: rev[pix + 1]]
            fname = os.path.join(self.outpath, '%s_%07d.fit' % (self.outbase_nopath, pix))

            if self.parallel:
                lockfile = fname + '.lock'
                if not make_lockfile(lockfile, block=True, maxtry=300, waittime=2):
                    raise RuntimeError("Could not get lock!")
                
                if not os.path.isfile(fname):
                    fitsio.write(fname, gals_arr[i1a])
                else:
                    with fitsio.FITS(fname, mode='rw') as fits:
                        fits[1].append(gals_arr[i1a])
                os.remove(lockfile)
            else:
                if (self.ngals[pix] == 0) and (os.path.isfile(fname)):
                    raise RuntimeError("File exists for pixel %d but ngals=0" % pix)
                if self.ngals[pix] == 0:
                    fitsio.write(fname, gals_arr[i1a])
                else:
                    with fitsio.FITS(fname, mode='rw') as fits:
                        fits[1].append(gals_arr[i1a])
            self.ngals[pix] += i1a.size

    def finalize_catalog(self):
        """Finish writing a galaxy catalog master table."""
        if self.parallel:
            if os.path.isfile(self.filename):
                return
            lockfile = self.filename + '.lock'
            if not make_lockfile(lockfile, block=False):
                return
            self.ngals[:] = 0
            files = sorted(glob.glob('%s/%s_???????.fit' % (self.outpath, self.outbase_nopath)))
            for f in files:
                m = re.search(r'_(\d{7})', f)
                pix = int(m.groups()[0])
                with fitsio.FITS(f) as fits:
                    self.ngals[pix] = fits[1].get_nrows()

        hpix, = np.where(self.ngals > 0)
        filename_dtype = 'U%d' % (len(self.outbase_nopath) + 15)

        dtype = [('nside', 'i2'), ('hpix', 'i4', (hpix.size, )), ('ra_pix', 'f8', (hpix.size, )),
                 ('dec_pix', 'f8', (hpix.size, )), ('ngals', 'i4', (hpix.size, )),
                 ('filenames', filename_dtype, (int(np.clip(hpix.size, 2, None)), )),
                 ('lim_ref', 'f4'), ('ref_ind', 'i2'), ('area', 'f8'), ('nmag', 'i4'),
                 ('mode', 'S10'), ('b', 'f8', (int(np.clip(self.nmag, 2, None)), )),
                 ('zeropoint', 'f4'), ('has_truth', 'i2'), ('has_zspec', 'i2')]
        
        for name in ['u_ind', 'g_ind', 'r_ind', 'i_ind', 'z_ind', 'y_ind']:
            if getattr(self, name) is not None:
                dtype.append((name, 'i2'))

        tab = Entry(np.zeros(1, dtype=dtype))
        tab.nside = self.nside
        tab.hpix = hpix
        tab.ra_pix, tab.dec_pix = hpg.pixel_to_angle(self.nside, hpix, nest=False)
        tab.ngals = self.ngals[hpix]
        for i, pix in enumerate(hpix):
            tab.filenames[i] = '%s_%07d.fit' % (self.outbase_nopath, pix)
        tab.lim_ref, tab.ref_ind, tab.area, tab.nmag, tab.mode = self.lim_ref, self.ref_ind, self.area, self.nmag, self.mode
        tab.b[: self.b.size] = self.b
        tab.zeropoint, tab.has_truth, tab.has_zspec = self.zeropoint, self.ingest_truth, self.ingest_zspec

        for name in ['u_ind', 'g_ind', 'r_ind', 'i_ind', 'z_ind', 'y_ind']:
            val = getattr(self, name)
            if val is not None:
                setattr(tab, name, val)

        hdr = fitsio.FITSHDR()
        hdr['PIXELS'] = 1
        hdr['FITS'] = 1
        tab.to_fits_file(self.filename, header=hdr, clobber=True)

        if self.generate_unique_ids:
            ctr = 1
            for f in tab.filenames:
                fname = os.path.join(self.outpath, f if isinstance(f, str) else decode_string(f))
                with fitsio.FITS(fname, mode=fitsio.READWRITE) as fits:
                    ids = np.arange(ctr, ctr + fits[1].get_nrows(), dtype=np.int64)
                    fits[1].write_column('id', ids)
                    ctr += ids.size
        else:
            # Check for unique IDs
            all_ids = []
            for i, f in enumerate(tab.filenames):
                if f == "": continue
                fname = os.path.join(self.outpath, f if isinstance(f, str) else decode_string(f))
                all_ids.append(fitsio.read(fname, columns=['id']))
            ids = np.concatenate(all_ids)
            if len(np.unique(ids)) < len(ids):
                raise RuntimeError("Input galaxy IDs must be unique.")

        self.is_finalized = True
        if self.parallel:
            os.remove(lockfile)

    def _check_galaxies(self, gals):
        """Check the galaxies for the data-type and for NaNs and illegal values."""
        dtype_required = self.get_galaxy_dtype(self.nmag, truth=self.ingest_truth, zspec=self.ingest_zspec)
        names = [d[0].lower() for d in gals.dtype.descr]
        
        for d in dtype_required:
            if d[0] not in names:
                raise RuntimeError("Required column %s missing." % d[0])
            if not np.isfinite(gals[d[0]]).all():
                raise RuntimeError("Non-finite values in %s." % d[0])
            if d[0] in ('mag', 'mag_err'):
                if (gals[d[0]] <= 0).any():
                    raise RuntimeError("Non-positive values in %s." % d[0])
                if d[0] == 'mag' and (gals[d[0]] >= 90.0).any():
                    raise RuntimeError("Input magnitude column %s contains elements with >= 90." % d[0])
                if d[0] == 'mag_err' and self.b[0] == 0.0 and (gals[d[0]] >= 90.0).any():
                    raise RuntimeError("Input mag_err column %s contains elements with >= 90." % d[0])

        if (gals['ra'] < 0.0).any() or (gals['ra'] > 360.0).any():
            raise RuntimeError("RA out of range.")
        if (gals['dec'] < -90.0).any() or (gals['dec'] > 90.0).any():
            raise RuntimeError("Dec out of range.")

        if not self.generate_unique_ids:
            if len(np.unique(gals['id'])) < len(gals['id']):
                raise RuntimeError("Input galaxy IDs must be unique.")

    @staticmethod
    def get_galaxy_dtype(nmag, truth=False, zspec=False):
        """Get the recommended galaxy dtype."""
        dtype = [('id', 'i8'), ('ra', 'f8'), ('dec', 'f8'), ('refmag', 'f4'),
                 ('refmag_err', 'f4'), ('mag', 'f4', nmag), ('mag_err', 'f4', nmag), ('ebv', 'f4')]
        if truth:
            dtype.extend([('ztrue', 'f4'), ('m200', 'f4'), ('central', 'i2'), ('halo_id', 'i8')])
        if zspec:
            dtype.extend([('zspec', 'f4'), ('zspec_err', 'f4')])
        return dtype
