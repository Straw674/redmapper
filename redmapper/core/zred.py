from ..redsequence import redsequence_zindex, compute_redsequence_chisq

def calculate_chisq(galaxies, redshift, zredstr, calc_lkhd=False):
    """
    Pure function version of calculate_chisq.
    """
    return compute_redsequence_chisq(zredstr, galaxies, redshift, calc_lkhd=calc_lkhd)
