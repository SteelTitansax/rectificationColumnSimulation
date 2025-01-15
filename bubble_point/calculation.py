import numpy as np
from scipy.optimize import fsolve
def bubble_point_temperature( x, K_i,T ):
    """
    Function to calculate the sum of y_i = K_i(T) * x_i for the bubble-point condition.
    We want the sum to equal 1.0.
    
    Arguments:
    T -- Temperature (in Kelvin)
    x -- Mole fractions of the liquid phase (as a dictionary)
    
    Returns:
    The difference from 1.0 that we want to solve for (should be zero at the bubble point).
    """
    sum_y = 0.0
    for component, mole_fraction in x.items():
        sum_y += K_i(T, component) * mole_fraction
        return sum_y - 1.0  # We want the sum to be equal to 1

    
