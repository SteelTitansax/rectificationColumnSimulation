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

    
    def _calculate_single_K(self, T_c, P_c, omega, T, P):
        """
        Realiza el cálculo del K-value para un solo valor de T_c, P_c y omega.
        """
        R = 8.31446261815324  # Constante de los gases en J/(mol*K)

        # Verificar que los parámetros sean flotantes
        try:
            T_c = float(T_c)
            P_c = float(P_c)
            omega = float(omega)
            T = float(T)
            P = float(P)
        except ValueError as e:
            raise ValueError("Los parámetros deben ser valores numéricos.") from e

        # Cálculo del parámetro b
        b = 0.07780 * R * T_c / P_c

        # Cálculo de kappa
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

        # Cálculo de alpha
        alpha = (1 + kappa * (1 - (T / T_c)**0.5))**2

        # Cálculo de a
        a = 0.45724 * (R**2 * T_c**2 / P_c) * alpha

        # Cálculo de Psat
        Psat = (R * T / (b * 2)) * (1 - (a / (R * T))**0.5)

        # Cálculo del K-value
        K_value = Psat / P

        return K_value


    def calculate_K(self, T_c, P_c, omega, T, P):
    
    #Calcula el K-value utilizando la ecuación de Peng-Robinson.
    #:param T_c: Temperatura crítica en K (puede ser un valor único o una lista de valores)
    #:param P_c: Presión crítica en Pa (puede ser un valor único o una lista de valores)
    #:param omega: Factor acéntrico (puede ser un valor único o una lista de valores)
    #:param T: Temperatura en K (valor único)
    #:param P: Presión en Pa (valor único)
    #:return: K-value para el componente (puede ser un valor único o una lista de valores)        
    # Verificar si T_c, P_c, omega son secuencias (listas o arrays) y tratarlos adecuadamente
    
        if isinstance(T_c, (list, tuple)):
            # Si T_c es una lista, lo tratamos como tal y calculamos K para cada valor
            K_values = []
            for t_c, p_c, w in zip(T_c, P_c, omega):
                K_values.append(self._calculate_single_K(t_c, p_c, w, T, P))
            return K_values
        else:
            # Si no es una lista, simplemente calculamos K para el único valor
            return self._calculate_single_K(T_c, P_c, omega, T, P)


    