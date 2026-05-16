"""Select red galaxies from the spectroscopic catalog"""

from ..core.calibration import select_spec_red_galaxies
from ..logger import logger

def select_spec_red_galaxies_wrapper(config):
    """
    Wrapper for the functional select_spec_red_galaxies.
    Maintains compatibility with legacy code.
    """
    return select_spec_red_galaxies(config, logger=logger)


