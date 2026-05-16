import xarray as xr
import numpy as np
import fitsio
from astropy.table import Table

def get_galaxy_schema(nmag, truth=False, zspec=False):
    """
    Get the standardized galaxy table schema.
    
    Parameters
    ----------
    nmag : int
        Number of magnitude bands.
    truth : bool, optional
        Include truth columns. Default is False.
    zspec : bool, optional
        Include spectroscopic redshift columns. Default is False.
        
    Returns
    -------
    list
        List of (name, dtype) or (name, dtype, shape) tuples.
    """
    dtype = [
        ('id', 'i8'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('refmag', 'f4'),
        ('refmag_err', 'f4'),
        ('mag', 'f4', nmag),
        ('mag_err', 'f4', nmag),
        ('ebv', 'f4')
    ]
    if truth:
        dtype.extend([
            ('ztrue', 'f4'),
            ('m200', 'f4'),
            ('central', 'i2'),
            ('halo_id', 'i8')
        ])
    if zspec:
        dtype.extend([
            ('zspec', 'f4'),
            ('zspec_err', 'f4')
        ])
    return dtype

def get_cluster_schema():
    """
    Get the standardized cluster table schema.
    
    Returns
    -------
    list
        List of (name, dtype) tuples.
    """
    return [
        ('mem_match_id', 'i4'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('z', 'f4'),
        ('refmag', 'f4'),
        ('refmag_err', 'f4'),
        ('lambda', 'f4'),
        ('lambda_e', 'f4'),
        ('z_lambda', 'f4'),
        ('z_lambda_e', 'f4'),
        ('cg_spec_z', 'f4'),
        ('z_spec_init', 'f4'),
        ('z_init', 'f4'),
        ('r_lambda', 'f4'),
        ('r_mask', 'f4'),
        ('scaleval', 'f4'),
        ('maskfrac', 'f4'),
        ('zred', 'f4'),
        ('zred_e', 'f4'),
        ('zred_chisq', 'f4'),
        ('chisq', 'f4'),
        ('z_lambda_niter', 'i2'),
        ('ebv_mean', 'f4'),
        ('lnlamlike', 'f4'),
        ('lncglike', 'f4'),
        ('lnlike', 'f4'),
        ('ra_orig', 'f8'),
        ('dec_orig', 'f8'),
        ('w', 'f4'),
        ('dlambda_dz', 'f4'),
        ('dlambda_dz2', 'f4'),
        ('dlambdavar_dz', 'f4'),
        ('dlambdavar_dz2', 'f4'),
        ('z_lambda_raw', 'f4'),
        ('z_lambda_e_raw', 'f4'),
        ('bkg_local', 'f4'),
        ('lim_exptime', 'f4'),
        ('lim_limmag', 'f4'),
        ('lim_limmag_hard', 'f4'),
        ('lambda_c', 'f4'),
        ('lambda_ce', 'f4'),
        ('ncent_good', 'i2'),
        ('maskgal_index', 'i2')
    ]

def get_member_schema():
    """
    Get the standardized member table schema.
    
    Returns
    -------
    list
        List of (name, dtype) tuples.
    """
    return [
        ('mem_match_id', 'i4'),
        ('id', 'i8'),
        ('z', 'f4'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('r', 'f4'),
        ('p', 'f4'),
        ('pfree', 'f4'),
        ('pcol', 'f4'),
        ('theta_i', 'f4'),
        ('theta_r', 'f4'),
        ('refmag', 'f4'),
        ('refmag_err', 'f4'),
        ('zred', 'f4'),
        ('zred_e', 'f4'),
        ('zred_chisq', 'f4'),
        ('chisq', 'f4'),
        ('ebv', 'f4'),
        ('zspec', 'f4')
    ]

def get_zred_schema(nsamp):
    """
    Get the standardized zred table schema.
    
    Parameters
    ----------
    nsamp : int
        Number of samples for zred_samp.
        
    Returns
    -------
    list
        List of (name, dtype) or (name, dtype, shape) tuples.
    """
    return [
        ('zred', 'f4'),
        ('zred_e', 'f4'),
        ('zred2', 'f4'),
        ('zred2_e', 'f4'),
        ('zred_uncorr', 'f4'),
        ('zred_uncorr_e', 'f4'),
        ('zred_samp', 'f4', nsamp),
        ('lkhd', 'f4'),
        ('chisq', 'f4')
    ]

def load_background_model(filename):
    """
    Load the background model into an xarray Dataset.
    
    This replaces the stateful Background class.
    """
    # The original code reads the CHISQBKG extension
    # We'll do something similar but return an xarray Dataset
    
    # For now, let's just implement a placeholder that shows the structure
    # In a real scenario, we'd copy the interpolation logic from Background.__init__
    
    with fitsio.FITS(filename) as fits:
        if 'CHISQBKG' not in [ext.get_extname() for ext in fits[1:]]:
            raise ValueError(f"Background file {filename} does not have CHISQBKG extension")
        
        data = fits['CHISQBKG'].read(lower=True)
        # Assuming data is a structured array with fields:
        # refmagrange, chisqrange, zrange, refmagbins, chisqbins, zbins, sigma_g, sigma_lng, etc.
        
        # In the original code, it does some interpolation on initialization.
        # With xarray, we can keep the raw data and use xarray's interpolation.
        
        # This is a simplified version:
        ds = xr.Dataset(
            data_vars={
                "sigma_g": (["refmag", "chisq", "z"], data['sigma_g'][0]),
                "sigma_lng": (["refmag", "chisq", "z"], data['sigma_lng'][0]),
            },
            coords={
                "refmag": data['refmagbins'][0],
                "chisq": data['chisqbins'][0],
                "z": data['zbins'][0],
            }
        )
        return ds

def background_lookup(model_ds, z, chisq, refmag):
    """
    Pure function to look up background values.
    """
    return model_ds.sigma_g.interp(z=z, chisq=chisq, refmag=refmag, method="linear")
