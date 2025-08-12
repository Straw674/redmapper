#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "solver_nfw.h"

int nfw_weights(double inlambda, double r0, double beta, long ngal,
		double *ucounts, double *bcounts, double *r, double *w,
		double *p, double *wt, double *rlambda, double *theta_r, double rsig,
                int update_all)
{
  double nfwnorm,logrlambda;
  int i;
  double rsig_const;

  // pre-compute rsig_const
  rsig_const = 1./(sqrt(2)*rsig);

  // calculate rlambda
  *rlambda=r0*pow((inlambda/100.),beta);
  
  // and find the norm  -- hard coded
  logrlambda = log(*rlambda);
  nfwnorm = exp(1.65169 - 0.547850*logrlambda + 0.138202*pow(logrlambda,2)
		-0.0719021*pow(logrlambda,3) - 0.0158241*pow(logrlambda,4)
		-0.000854985*pow(logrlambda,5));
    
  for (i=0;i<ngal;i++) {
    // calculate p[i]
      if (update_all) {
          p[i] = (inlambda*ucounts[i]*nfwnorm)/(inlambda*ucounts[i]*nfwnorm+bcounts[i]);
      }

    // check for infinity/NaN
    if (!isfinite(p[i])) p[i]=0.0;
    
    // and check radius and do weights
    if (rsig <= 0.0) {
      // old skool... hard cut.
      if (r[i] < *rlambda) {
         if (update_all) {
	    wt[i]=p[i]*w[i];
         }
	 theta_r[i] = 1.0; // added by Tom
      } else {
         wt[i]=0.0;
      }
    } else { // rsig > 0.0
      /*
	TOM - Note: on line 152 of calclambda_chisq_cluster_lambda.pro
	we can see that this case always results in theta_r[i] being equal
	to the function with the erf() in it.
      */
      // Now we have a soft cut.
      if (r[i] < (*rlambda - 5.0*rsig)) {
        // r correction value is 1.0
          theta_r[i] = 1.0;
          //wt[i] = p[i]*w[i]*1.0;
      } else if (r[i] > (*rlambda + 5.0*rsig)) {
	// r correction value is 0.0
          theta_r[i] = 0.0;
          //wt[i] = 0.0;
      } else {
	// need to calculate r correction...
          theta_r[i] = (0.5+0.5*erf((*rlambda - r[i])*rsig_const));
          //wt[i] = p[i]*w[i]*(0.5+0.5*erf((*rlambda - r[i])*rsig_const));
      }
      if (update_all) {
         wt[i] = p[i] * w[i] * theta_r[i];
      }
    }
  }
  
  return 0;
}
