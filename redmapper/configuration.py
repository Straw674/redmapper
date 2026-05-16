"""Configuration module for redmapper.

This module provides functional interfaces for redmapper configuration management.
"""
import yaml
import fitsio
import copy
from esutil.cosmology import Cosmo
import numpy as np
import re
import os
from ._version import __version__
from .logger import logger, add_file_handler
from .utilities import decode_string
from .core.config import DEFAULT_CONFIG

def read_yaml(filename):
    """Read a yaml file into a dictionary."""
    with open(filename, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
    return yaml_data if yaml_data is not None else {}

def get_galfile_stats(galfile, refmag):
    """Get statistics from the galaxy file."""
    hdr = fitsio.read_header(galfile, ext=1)
    pixelated = hdr.get("PIXELS", 0)
    fitsformat = hdr.get("FITS", 0)

    if not fitsformat:
        raise ValueError("Input galfile must describe fits files.")

    gal_stats = {}
    if not pixelated:
        gal_stats['galfile_pixelized'] = False
        hdrmode = hdr.get("MODE", "").rstrip()
        if hdrmode == 'SDSS':
            gal_stats['survey_mode'] = 0
        elif hdrmode == 'DES':
            gal_stats['survey_mode'] = 1
        elif hdrmode == 'LSST':
            gal_stats['survey_mode'] = 2
        else:
            raise ValueError("Input galaxy file with unknown mode: %s" % (hdrmode))

        gal_stats['area'] = hdr.get('AREA', -100.0)
        gal_stats['limmag_ref'] = hdr.get('LIM_REF')
        gal_stats['nmag'] = hdr.get('NMAG')
        gal_stats['zeropoint'] = hdr.get('ZP')
        gal_stats['ref_ind'] = hdr.get(refmag.upper()+'_IND')
        gal_stats['b'] = None
        gal_stats['galfile_nside'] = 0
        gal_stats['bands'] = [None]*gal_stats['nmag']

        for name in hdr:
            m = re.search('(.*)_IND', name)
            if m is None or m.groups()[0] == 'REF':
                continue
            band = m.groups()[0].lower()
            gal_stats['bands'][hdr[name]] = band

        elt = fitsio.read(galfile, ext=1, rows=0, lower=True)
        gal_stats['galfile_has_truth'] = 'ztrue' in elt.dtype.names
        gal_stats['galfile_has_zspec'] = 'zspec' in elt.dtype.names
    else:
        gal_stats['galfile_pixelized'] = True
        main = fitsio.read(galfile, ext=1, lower=True)
        try:
            mode = decode_string(main['mode'][0].rstrip())
        except AttributeError:
            mode = main['mode'][0].rstrip()
            
        if mode == 'SDSS':
            gal_stats['survey_mode'] = 0
        elif mode == 'DES':
            gal_stats['survey_mode'] = 1
        elif mode == 'LSST':
            gal_stats['survey_mode'] = 2
        else:
            raise ValueError("Input galaxy file with unknown mode: %s" % (mode))

        gal_stats['area'] = main['area'][0]
        gal_stats['limmag_ref'] = main['lim_ref'][0]
        gal_stats['nmag'] = main['nmag'][0]
        gal_stats['b'] = main['b'][0] if 'b' in main.dtype.names else None
        gal_stats['zeropoint'] = main['zeropoint'][0]
        gal_stats['ref_ind'] = main[refmag.lower()+'_ind'][0]
        gal_stats['galfile_nside'] = main['nside'][0]
        gal_stats['bands'] = [None]*gal_stats['nmag']

        for name in main.dtype.names:
            m = re.search('(.*)_ind', name)
            if m is None or m.groups()[0] == 'ref':
                continue
            band = m.groups()[0].lower()
            gal_stats['bands'][main[name][0]] = band

        try:
            gal_stats['galfile_has_truth'] = bool(main['has_truth'][0])
            gal_stats['galfile_has_zspec'] = bool(main['has_zspec'][0])
        except:
            path = os.path.dirname(os.path.abspath(galfile))
            try:
                first_fname = os.path.join(path, decode_string(main['filenames'][0][0]))
            except AttributeError:
                first_fname = os.path.join(path, main['filenames'][0][0])
            elt = fitsio.read(first_fname, ext=1, rows=0, lower=True)
            gal_stats['galfile_has_truth'] = 'ztrue' in elt.dtype.names
            gal_stats['galfile_has_zspec'] = 'zspec' in elt.dtype.names

    if any(x is None for x in gal_stats['bands']):
        gal_stats.pop('bands', None)

    return gal_stats

def get_wcen_vals(wcenfile):
    """Load wcen values from wcenfile."""
    if wcenfile is None or not os.path.isfile(wcenfile):
        return {}

    wcen = fitsio.read(wcenfile, ext=1, lower=True)
    vals = {'wcen_Delta0': wcen[0]['delta0'],
            'wcen_Delta1': wcen[0]['delta1'],
            'wcen_sigma_m': wcen[0]['sigma_m'],
            'wcen_pivot': wcen[0]['pivot'],
            'lnw_fg_mean': wcen[0]['lnw_fg_mean'],
            'lnw_fg_sigma': wcen[0]['lnw_fg_sigma'],
            'lnw_sat_mean': wcen[0]['lnw_sat_mean'],
            'lnw_sat_sigma': wcen[0]['lnw_sat_sigma'],
            'lnw_cen_mean': wcen[0]['lnw_cen_mean'],
            'lnw_cen_sigma': wcen[0]['lnw_cen_sigma']}

    if 'phi1_mmstar_m' in wcen[0].dtype.names:
        vals['phi1_mmstar_m'] = wcen[0]['phi1_mmstar_m']
        vals['phi1_mmstar_slope'] = wcen[0]['phi1_mmstar_m']
        vals['phi1_msig_m'] = wcen[0]['phi1_msig_m']
        vals['phi1_msig_slope'] = wcen[0]['phi1_msig_slope']

    return vals

class Config(dict):
    """
    A dictionary subclass that allows dot-access.
    This is a temporary bridge to maintain compatibility while the Configuration class is eliminated.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        # A simple copy that avoids re-running __init__ logic
        new_config = self.__class__.__new__(self.__class__)
        new_config.update(self)
        return new_config

    def get_zrange_cushioned(self):
        return get_zrange_cushioned(self)

    def redmapper_filename(self, *args, **kwargs):
        return get_redmapper_filename(self, *args, **kwargs)

    def compute_border(self, cosmo=None):
        return compute_border(self, cosmo=cosmo)

    def start_file_logging(self, filename=None):
        start_config_logging(self, filename=filename)

    def stop_file_logging(self):
        from .logger import remove_file_handlers
        remove_file_handlers()

    def check_files(self, **kwargs):
        check_config_files(self, **kwargs)

    def output_yaml(self, filename):
        write_config_yaml(self, filename)

class Configuration(Config):
    """
    Legacy-compatible Configuration class that loads from a file on initialization.
    """
    def __init__(self, configfile=None, outpath=None, **kwargs):
        # Initialize as a dictionary first
        super(Config, self).__init__()
        
        from .cluster import cluster_dtype_base, member_dtype_base

        # 1. Load defaults
        for key, value in DEFAULT_CONFIG.items():
            self[key] = copy.copy(value) if isinstance(value, (list, np.ndarray, dict)) else value

        # 2. Basic fields
        self['version'] = __version__
        self.update({
            'galfile': None, 'zredfile': None, 'halofile': None, 'randfile': None,
            'catfile': None, 'specfile': None, 'specfile_train': None, 'outbase': None,
            'parfile': None, 'bkgfile': None, 'bkgfile_color': None, 'zlambdafile': None,
            'maskfile': None, 'depthfile': None, 'wcenfile': None, 'redgalfile': None,
            'redgalmodelfile': None, 'seedfile': None, 'zmemfile': None, 'redmagicfile': None,
            'refmag': None, 'nmag': None, 'area': None, 'limmag_catalog': None, 'limmag_ref': None,
            'zeropoint': None, 'survey_mode': None, 'b': None, 'galfile_nside': None,
            'bands': None, 'has_truth': False, 'galfile_has_truth': False, 'galfile_has_zspec': False,
            'zrange': None, 'percolation_lmask': None, 'percolation_memradius': None,
            'percolation_memlum': None, 'calib_redgal_template': None, 'galfile_pixelized': None
        })

        # 3. Override with yaml and kwargs
        if configfile:
            confdict = read_yaml(configfile)
            self.update(confdict)
            self['configpath'] = os.path.dirname(os.path.abspath(configfile))
            self['configfile'] = os.path.basename(configfile)

        if outpath:
            self['outpath'] = outpath

        self.update(kwargs)

        # 4. Ensure array fields
        array_fields = ['hpix', 'b', 'zrange', 'calib_color_nodesizes', 'calib_slope_nodesizes',
                        'calib_color_maxnodes', 'calib_covmat_maxnodes', 'calib_colormem_zbounds',
                        'calib_colormem_colormodes', 'calib_colormem_sigint', 'bkg_local_annuli',
                        'wcen_cal_zrange', 'consolidate_lambda_cuts', 'redmagic_zrange',
                        'redmagic_n0s', 'redmagic_etas', 'redmagic_zmaxes', 'redmagic_constchis']
        for field in array_fields:
            if field in self and self[field] is not None:
                self[field] = np.atleast_1d(self[field])

        # 5. Post-init logic
        if self['galfile'] and self['refmag']:
            gal_stats = get_galfile_stats(self['galfile'], self['refmag'])
            for key, value in gal_stats.items():
                if self.get(key) is None:
                    self[key] = value
            
            if self['area'] is not None and self['depthfile'] is not None:
                logger.warning("You should not need to set area in the config file when you have a depth map.")

        if self['wcenfile']:
            wcen_vals = get_wcen_vals(self['wcenfile'])
            for key, value in wcen_vals.items():
                self[key] = value

        if self.get('specfile_train') is None:
            self['specfile_train'] = self.get('specfile')

        if self.get('limmag_catalog') is None:
            self['limmag_catalog'] = self.get('limmag_ref')

        if len(self.get('redmagic_zrange', [])) == 0 and self.get('zrange') is not None:
            self['redmagic_zrange'] = copy.copy(self['zrange'])

        # 6. dtypes
        if self['nmag'] is not None and self['npzbins'] is not None and self['percolation_maxcen'] is not None:
            c_dtype = copy.copy(cluster_dtype_base)
            c_dtype.extend([('MAG', 'f4', self['nmag']),
                            ('MAG_ERR', 'f4', self['nmag']),
                            ('PZBINS', 'f4', self['npzbins']),
                            ('PZ', 'f4', self['npzbins']),
                            ('RA_CENT', 'f8', self['percolation_maxcen']),
                            ('DEC_CENT', 'f8', self['percolation_maxcen']),
                            ('ID_CENT', 'i8', self['percolation_maxcen']),
                            ('LAMBDA_CENT', 'f4', self['percolation_maxcen']),
                            ('ZLAMBDA_CENT', 'f4', self['percolation_maxcen']),
                            ('P_CEN', 'f4', self['percolation_maxcen']),
                            ('Q_CEN', 'f4', self['percolation_maxcen']),
                            ('P_FG', 'f4', self['percolation_maxcen']),
                            ('Q_MISS', 'f4'),
                            ('P_SAT', 'f4', self['percolation_maxcen']),
                            ('P_C', 'f4', self['percolation_maxcen'])])
            self['cluster_dtype'] = c_dtype
            
            m_dtype = copy.copy(member_dtype_base)
            m_dtype.extend([('MAG', 'f4', self['nmag']),
                            ('MAG_ERR', 'f4', self['nmag'])])
            self['member_dtype'] = m_dtype
        else:
            self['cluster_dtype'] = None
            self['member_dtype'] = None

        self['cosmo'] = Cosmo()

def create_config(configfile=None, outpath=None, **kwargs):
    """
    Functional entry point for creating a redmapper configuration.
    """
    return Configuration(configfile=configfile, outpath=outpath, **kwargs)

def get_zrange_cushioned(config):
    """Return the zrange with cushions."""
    zrange = config.get('zrange')
    if zrange is None: return None
    zrange_cushioned = np.array(zrange, dtype=float)
    zrange_cushioned[0] = np.clip(zrange_cushioned[0] - config['calib_zrange_cushion'], 0.05, None)
    zrange_cushioned[1] += config['calib_zrange_cushion']
    return zrange_cushioned

def get_redmapper_filename(config, redmapper_name, paths=None, filetype='fit',
                           withversion=False, outbase=None):
    """Generate a redmapper filename."""
    if outbase is None:
        outbase = config['outbase']

    if withversion:
        outbase += '_redmapper_v%s' % (config['version'])

    if paths is None:
        return os.path.join(config['outpath'],
                            '%s_%s.%s' % (outbase, redmapper_name, filetype))
    else:
        pars = [config['outpath']]
        pars.extend(paths)
        pars.append('%s_%s.%s' % (outbase, redmapper_name, filetype))
        return os.path.join(*pars)

def compute_border(config, cosmo=None):
    """Compute the border radius."""
    if config.get('zrange') is None: return 0.0
    if cosmo is None:
        cosmo = Cosmo()
    maxdist = 1.05 * config['percolation_rmask_0'] * (300. / 100.)**config['percolation_rmask_beta']
    radius = maxdist / (np.radians(1.) * cosmo.Da(0, config['zrange'][0]))
    return 3.0 * radius

def start_config_logging(config, filename=None):
    """Start logging to a file based on configuration."""
    if config.get('printlogging'):
        logger.info("Logging is set to be to console only.")
        return

    if filename is None:
        hpix_val = config['hpix'][0] if len(config['hpix']) > 0 else 0
        logfilename = os.path.join(config['outpath'], config['logpath'],
                                   f'redmapper_{config["outbase"]}_{hpix_val:04}.log')
    else:
        logfilename = os.path.join(config['outpath'], config['logpath'], os.path.basename(filename))
    
    add_file_handler(logfilename)

def check_config_files(config, check_zredfile=False, check_bkgfile=False, check_bkgfile_components=False,
                       check_parfile=False, check_zlambdafile=False, check_randfile=False):
    """Validate that required files exist."""
    if check_zredfile and (config.get('zredfile') is None or not os.path.isfile(config['zredfile'])):
        raise ValueError("zredfile %s not found." % (config.get('zredfile')))
    if check_bkgfile and (config.get('bkgfile') is None or not os.path.isfile(config['bkgfile'])):
        raise ValueError("bkgfile %s not found." % (config.get('bkgfile')))
    if check_bkgfile_components:
        if config.get('bkgfile') is None or not os.path.isfile(config['bkgfile']):
            raise ValueError("bkgfile not found.")
        with fitsio.FITS(config['bkgfile']) as fits:
            extnames = [ext.get_extname() for ext in fits[1:]]
            if 'CHISQBKG' not in extnames:
                raise ValueError("bkgfile %s does not have CHISQBKG extension." % (config['bkgfile']))
            if 'ZREDBKG' not in extnames:
                raise ValueError("bkgfile %s does not have ZREDBKG extension." % (config['bkgfile']))
    if check_parfile and (config.get('parfile') is None or not os.path.isfile(config['parfile'])):
        raise ValueError("parfile %s not found." % (config.get('parfile')))
    if check_zlambdafile and (config.get('zlambdafile') is None or not os.path.isfile(config['zlambdafile'])):
        raise ValueError("zlambdafile %s not found." % (config.get('zlambdafile')))
    if check_randfile and (config.get('randfile') is None or not os.path.isfile(config['randfile'])):
        raise ValueError("randfile %s not found." % (config.get('randfile')))

def write_config_yaml(config, filename):
    """Write configuration to a yaml file."""
    out_dict = {}
    for key, value in config.items():
        if key.startswith('_') or key in ['cosmo', 'logger', 'cluster_dtype', 'member_dtype']: continue
        if isinstance(value, np.ndarray):
            out_dict[key] = value.tolist()
        elif isinstance(value, list):
            out_dict[key] = [list(x) if isinstance(x, tuple) else x for x in value]
        else:
            try:
                out_dict[key] = value.item()
            except (ValueError, AttributeError, TypeError):
                if isinstance(value, tuple):
                    out_dict[key] = list(value)
                else:
                    out_dict[key] = value
    with open(filename, 'w') as f:
        yaml.dump(out_dict, stream=f, Dumper=yaml.SafeDumper)

def setup_config_environment(config):
    """Initialize environment (directories)."""
    if config['outpath'] and not os.path.exists(config['outpath']):
        os.makedirs(config['outpath'], exist_ok=True)
    if config['outpath'] and config.get('plotpath') and not os.path.exists(os.path.join(config['outpath'], config['plotpath'])):
        os.makedirs(os.path.join(config['outpath'], config['plotpath']), exist_ok=True)
