import fitsio
import numpy as np

def read_fits_catalog(filename, ext=1, rows=None, lower=True):
    """Read a FITS catalog into a numpy structured array."""
    return fitsio.read(filename, ext=ext, rows=rows, lower=lower, trim_strings=True)

def write_fits_catalog(filename, array, clobber=False, header=None, extname=None):
    """Write a numpy structured array to a FITS file."""
    fitsio.write(filename, array, clobber=clobber, header=header, extname=extname)
