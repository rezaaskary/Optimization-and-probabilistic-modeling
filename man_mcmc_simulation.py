import numpy as np
from Probablity_distributions import *

def vc(x):
    if isinstance(x, list):
        return np.array(x)
    else:
        raise Exception('The format of the input variable is not supported!')




def log_posteriori_fcn():
    parameter1 = Uniform(a=0, b=4, return_der_pdf=False, return_der_logpdf=False)
    parameter2 = Normal(sigma=3, mu=0, return_der_pdf=False, return_der_logpdf=False)
    CC=parameter1.pdf(np.array(.1))

    return

T = log_posteriori_fcn()
