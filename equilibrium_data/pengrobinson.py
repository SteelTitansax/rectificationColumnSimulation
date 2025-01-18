import numpy as np


class PengRobinson:
    def __init__(self, compound, T_c, P_c, omega, T, P, verbose=False):
        
        """
        :param compound: Name of the compound.
        :param T_c: Critical temperature in Kelvin.
        :param P_c: Critical pressure in Pa.
        :param omega: Acentric factor.
        :param verbose: If True, prints the parameters.
        """
        self.compound = compound
        self.T_c = T_c
        self.P_c = P_c
        self.omega = omega
        
        self.R = 8.314  # Ideal gas constant in J/(mol·K)
	        
        # Parameters of the Peng-Robinson equation    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        
        self.b = 0.07780 * self.R * T_c / P_c
        
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        
        alpha = (1 + kappa * (1 - (T / T_c)**0.5))**2
        
        self.a = 0.45724 * (self.R**2 * T_c**2 / P_c) * alpha
        
        Psat = (self.R * T / (self.b * 2)) * (1 - (self.a / (self.R * T))**0.5)
        K_result = Psat / P
        if verbose:
            print(f"Peng-Robinson parameters for {self.compound}:")
            print(f"     a = {self.a:.2f}  b = {self.b:.2f}")
            
            print('     K value at ' + str(T_c) +' K and '+ str(P_c) + ' Pa : ' + str(K_result))

    
    def _calculate_single_K(self, T_c, P_c, omega, T, P):
        """
        Calculates the K-value for a single value of T_c, P_c, and omega.
        """
        R = 8.31446261815324  # Gas constant in J/(mol*K)

        # Check that parameters are floating point numbers
        try:
            T_c = float(T_c)
            P_c = float(P_c)
            omega = float(omega)
            T = float(T)
            P = float(P)
        except ValueError as e:
            raise ValueError("Parameters must be numeric values.") from e

        # Calculate the b parameter
        b = 0.07780 * R * T_c / P_c

        # Calculate kappa
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

        # Calculate alpha
        alpha = (1 + kappa * (1 - (T / T_c)**0.5))**2

        # Calculate a
        a = 0.45724 * (R**2 * T_c**2 / P_c) * alpha

        # Calculate Psat
        Psat = (R * T / (b * 2)) * (1 - (a / (R * T))**0.5)

        # Calculate the K-value
        K_value = Psat / P

        return K_value


    def calculate_K(self, T_c, P_c, omega, T, P):
    
    #Calculates the K-value using the Peng-Robinson equation.
    #:param T_c: Critical temperature in K (can be a single value or a list of values)
    #:param P_c: Critical pressure in Pa (can be a single value or a list of values)
    #:param omega: Acentric factor (can be a single value or a list of values)
    #:param T: Temperature in K (single value)
    #:param P: Pressure in Pa (single value)
    #:return: K-value for the component (can be a single value or a list of values)        
    # Check if T_c, P_c, omega are sequences (lists or arrays) and handle them accordingly
    
        if isinstance(T_c, (list, tuple)):
            # If T_c is a list, treat it as such and calculate K for each value
            K_values = []
            for t_c, p_c, w in zip(T_c, P_c, omega):
                K_values.append(self._calculate_single_K(t_c, p_c, w, T, P))
            return K_values
        else:
            # If it is not a list, simply calculate K for the single value
            return self._calculate_single_K(T_c, P_c, omega, T, P)
