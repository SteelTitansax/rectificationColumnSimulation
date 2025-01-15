import numpy as np


class PengRobinson:
    def __init__(self, compound, T_c, P_c, omega, T, P, verbose=False):
        
        """
        :param compound: Nombre del compuesto.
        :param T_c: Temperatura crítica en Kelvin.
        :param P_c: Presión crítica en Pa.
        :param omega: Factor acéntrico.
        :param verbose: Si es True, imprime los parámetros.
        """
        self.compound = compound
        self.T_c = T_c
        self.P_c = P_c
        self.omega = omega
        
        self.R = 8.314  # Constante de los gases ideales en J/(mol·K)
	        
        # Parámetros de la ecuación de Peng-Robinson    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        
        self.b = 0.07780 * self.R * T_c / P_c
        
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        
        alpha = (1 + kappa * (1 - (T / T_c)**0.5))**2
        
        self.a = 0.45724 * (self.R**2 * T_c**2 / P_c) * alpha
        
        Psat = (self.R * T / (self.b * 2)) * (1 - (self.a / (self.R * T))**0.5)
        K_result = Psat / P
        if verbose:
            print(f"Parametros de Peng-Robinson para {self.compound}:")
            print(f"     a = {self.a:.2f}  b = {self.b:.2f}")
            
            print('     K value at ' + str(T_c) +' K and '+ str(P_c) + ' Pa : ' + str(K_result))

    
    def calculate_K(self, T_c, P_c, omega, T, P):
        """
        Calcula el K-value utilizando la ecuación de Peng-Robinson.
        
        :param T: Temperatura en K
        :param P: Presión en Pa
        :return: K-value para el componente.
        """
        self.b = 0.07780 * self.R * T_c / P_c
        
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        
        alpha = (1 + kappa * (1 - (T / T_c)**0.5))**2
        
        self.a = 0.45724 * (self.R**2 * T_c**2 / P_c) * alpha
        
        Psat = (self.R * T / (self.b * 2)) * (1 - (self.a / (self.R * T))**0.5)
        K_value = Psat / P
        
        return K_value



