import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from ._version import __version__, __version_tuple__

version = __version__

from .logger import logger, set_log_level
from . import calibration
from . import pipeline
from . import redmagic

from .configuration import create_config, get_zrange_cushioned, get_redmapper_filename, compute_border, start_config_logging, check_config_files, write_config_yaml, setup_config_environment, Configuration
from .solver_nfw import Solver
from .catalog import Entry, Catalog
from .redsequence import read_redsequence, redsequence_zindex, redsequence_refmagindex, redsequence_lumrefmagindex, redsequence_mstar, compute_redsequence_chisq, compute_redsequence_chisq_redshifts
from .chisq_dist import compute_chisq
from .background import read_background, read_zred_background, compute_background, compute_zred_background, generate_background, generate_zred_background
from .cluster import Cluster, ClusterCatalog
from .galaxy import Galaxy, GalaxyCatalog, GalaxyCatalogMaker
from .mask import get_mask, get_mask_values, read_mask, read_maskgals, select_maskgals_sample, gen_maskgals, compute_maskgals_mark, calc_maskcorr
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction

from .zred_color import compute_zreds, compute_zred
from .centering import CENTERING_FUNCS
from .color_background import read_color_background, sigma_g_diagonal, lookup_diagonal, get_colrange, lookup_offdiag, generate_color_background
from .fitters import (med_z_cost, fit_med_z,
                       red_sequence_cost, fit_red_sequence,
                       red_sequence_off_diagonal_cost, fit_red_sequence_off_diagonal,
                       fit_correction, fit_ecgmm, fit_error_bin)
from .zred_runner import run_zred_catalog, run_zred_pixels
from .redmapper_run import redmapper_run
from .depth_fitting import compute_depthlim_pars, apply_depthlim, applyErrorModel
from .plotting import plot_spec_comparison, plot_nz, plot_nlambda, plot_positions, plot_redmagic_nz
from .volumelimit import create_volume_limit_mask, create_volume_limit_mask_fixed, calc_zmax, get_volume_limit_areas
from .utilities import read_members
from .randoms import generate_randoms, RandomCatalog, RandomCatalogMaker, weight_randoms
from .run_randoms_zmask import run_randoms_zmask
