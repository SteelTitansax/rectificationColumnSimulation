import numpy as np


class PengRobinson:
    def __init__(self, compound, T_c, P_c, omega, T, verbose=False):
        
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
	        
        # Parámetros de la ecuación de Peng-Robinson
        self.a = 0.45724 * (self.R**2 * self.T_c**2) / self.P_c * (1 + (0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2) * (1 - np.sqrt(T_c / T)))**2
        self.b = 0.07780 * self.R * self.T_c / self.P_c
        
        if verbose:
            print(f"Parametros de Peng-Robinson para {self.compound}:")
            print(f"     a = {self.a:.2f}  b = {self.b:.2f}")
            K_result = self.calculate_K(T, P_c)
            print('     K value at ' + str(T_c) +' K and '+ str(P_c) + ' Pa : ' + str(K_result[0]))

    
    def calculate_K(self, T, P):
        """
        Calcula el K-value utilizando la ecuación de Peng-Robinson.
        
        :param T: Temperatura en K
        :param P: Presión en Pa
        :return: K-value para el componente.
        """
        # Ajuste de temperatura y presión
        Tr = T / self.T_c  # Temperatura reducida
        Pr = P / self.P_c  # Presión reducida
        
        # Calcular los coeficientes de Peng-Robinson
        a_T = self.a * (1 + (0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2) * (1 - np.sqrt(Tr)))**2
        b_T = self.b * (1 - np.sqrt(Pr))
        
        # Obtener el K-value de la mezcla
        K_value = np.exp(-a_T / (self.R * T) + b_T * P)
        
        return K_value, P



