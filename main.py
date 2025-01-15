from scipy.sparse import linalg, diags
from equilibrium_data.heat_capacity_liquid import CpL
from equilibrium_data.heat_capacity_vapor import CpV
from equilibrium_data.heats_of_vaporization import dH_vap
from equilibrium_data.pengrobinson import PengRobinson
from bubble_point.calculation import bubble_point
import os 

import numpy as np 

class Model : 

    def __init__(self, components: list = None, F: float = 0., P: float = 607950 , 
                 z_feed: list = None, RR: float = 1, D: float = 0, N: int = 1, feed_stage: int = 0,
                 T_feed_guess: float = 79 ):
    
        """Distillation column with partial reboiler and total condenser.
        
        Feed is saturated liquid.        

        :param components: list of component names
        :param F: feed molar flow rate
        :param P: pressure (constant throughout column), [Pa]
        :param z_feed: mole fractions of each component, ordered
        :param RR: reflux ratio (L/D)
        :param D: distillate molar flow rate
        :param N: number of equilibrium contacts
        :param feed_stage: stage where feed is input
        :param T_feed_guess: guess temperature for feed stage
        
        """

        self.flow_rate_tol = 1.e-4
        self.temperature_tol = 1.e-2
        self.components = components
        self.F = F
        self.F_feed = F
        self.P = P
        self.P_feed = P
        self.z_feed = {key: val for key, val in zip(components, z_feed)}        
        self.RR = RR
        self.B = F - D
        self.D = D
        self.N = N
        self.feed_stage = feed_stage
        self.T_feed_guess = T_feed_guess
        self.K_func = {}
        self.CpL_func = {}
        self.CpV_func = {}
        self.dH_func = {}
        self.T_ref = {}

        # Matrices variables creation following the Amundson, NR and Pontinen, AJ. Multicomponent Distillation
        # Calculations on a Large Digital Computer. 
        # Ind. Eng. Chem. 1958;50:730–736.

        self.num_stages = self.N + 1 # plus distilator stage
        self.stages = range(self.num_stages)
        self.L = np.zeros(self.num_stages)
        self.V = np.zeros(self.num_stages)
        self.L_old = np.zeros(self.num_stages)
        self.V_old = np.zeros(self.num_stages)
        self.F = np.zeros(self.num_stages)
        self.F[self.feed_stage] = self.F_feed

        # We set up feed_stage concentration for every component

        self.z = {
            key: np.zeros(self.num_stages) for key in components
        }
        self.l = {
            key: np.zeros(self.num_stages) for key in components
        }
        for component in self.components:
            self.z[component][feed_stage] = self.z_feed[component] # molar concentrations
        
        self.T_feed = self.T_feed_guess
        self.T = np.zeros(self.num_stages)
        self.T_old = np.zeros(self.num_stages)
        self.K = {key: np.zeros(self.num_stages) for key in self.components}

        # solver parameters
        self.df = 1.  # Dampening factor to prevent excessive oscillation of temperatures
    def set_parameters(self, verbose=False):
        
        print('Setting parameters ...')

        """Add thermodynamic parameters for calculation

        .. note::

            K values from PengRobinson model 
            CpL from Perrys
            CpV assumes ideal gas
            dH_vap from NIST Webbook
        """

        T_feed = self.T_feed
        P=self.P
        self.K_func = {}
        ROOT_DIR = os.getcwd()

        for key in self.components:

            # Get the critical parameters for the compound

            f_name = os.path.join(ROOT_DIR, 'equilibrium_data', 'pengrobinson.csv')
            data = read_csv_data(f_name)
            assert key in data.keys(), f'Compound {key} not found!'
            compound_data = data[key]
            
            T_c = float(compound_data['Tc (K)'])
            P_c = float(compound_data['Pc (Pa)'])
            omega = float(compound_data['Omega\n'])
            
            # Create a PengRobinson instance with all required parameters
            
            self.K_func[key] = PengRobinson(key, T_c, P_c, omega,T_feed,P, True)

            self.CpL_func = {
                key: CpL(key, verbose) for key in self.components
            }

            self.CpV_func = {
                key: CpV(key, verbose) for key in self.components
            }
            self.dH_func = {
                key: dH_vap(key, verbose) for key in self.components
            }
            self.T_ref = {
                key: val.T_ref for key, val in self.dH_func.items()
            }


        print('Parameters set sucessfully...')

    def bubble_T_feed(self):
        P_c_values = []  # Lista para almacenar los valores de presión crítica
        T_c_values = []  # Lista para almacenar los valores de presión crítica
        omega_values = []  # Lista para almacenar los valores de presión crítica
        T = self.T_feed
        P = self.P
        for key in self.components:

            ROOT_DIR = os.getcwd()
            f_name = os.path.join(ROOT_DIR, 'equilibrium_data', 'pengrobinson.csv')
            data = read_csv_data(f_name)
            assert key in data.keys(), f'Compound {key} not found!'
            compound_data = data[key]
            P_c = float(compound_data['Pc (Pa)'])
            P_c_values.append(P_c)
            T_c = float(compound_data['Tc (K)'])
            T_c_values.append(T_c)
            omega = float(compound_data['Omega\n'])
            omega_values.append(omega)

            print(P_c_values)  # Agregar P_c a la lista
            print(T_c_values)  # Agregar P_c a la lista

        K_values = [
        self.K_func[component].calculate_K(T_c_values[idx], P_c_values[idx],omega,T,P)
        for idx, component in enumerate(self.components)
        ]

        
        print(K_values)
        return bubble_point(
            [self.z_feed[i] for i in self.components],
            K_values,
            self.P_feed, 
            self.T_feed_guess
        )


def read_csv_data(f_name):
    data = {}
    with open(f_name, 'r') as f:
        header = next(f).split(',')
        for line in f:
            compound, *vals = line.split(',')
            data[compound] = {
                key: val for key, val in zip(header[1:], vals)
            }

    return data




if __name__ == '__main__':
    model = Model(
        ['Argon','Nytrogen', 'Oxygen'],
        F=1305., # kmol/h
        P=538000, # KPa
        z_feed = [0.00005, 0.7808, 0.2095],
        T_feed_guess = 79, # Operation temperature 
        RR=1.,
        D=400.,
        N=40,
        feed_stage=15,

    )
    
    # Setting parameters 

    model.set_parameters()

    # Calculate T using Bubble point 

    print(model.T_feed)
    model.T_feed = model.bubble_T_feed()
    print(model.T_feed)



