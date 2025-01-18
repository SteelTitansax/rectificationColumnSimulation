from scipy.sparse import linalg, diags
from equilibrium_data.heat_capacity_liquid import CpL
from equilibrium_data.heat_capacity_vapor import CpV
from equilibrium_data.heats_of_vaporization import dH_vap
from equilibrium_data.pengrobinson import PengRobinson
import os 
import numpy as np
from scipy.optimize import fsolve, OptimizeResult
from solvers import solve_diagonal

class Model : 

    def __init__(self, components: list = None, F: float = 0., P: float = 607950 , 
                 z_feed: list = None, RR: float = 1, D: float = 0, N: int = 1, feed_stage: int = 0,
                 T_feed_guess: float = 79 ):
    
        """
        
        Distillation column with partial reboiler and total condenser.
        
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

        """
        Add thermodynamic parameters for calculation

        .. Note::

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

    def initialize_flow_rates(self):
        
        # initialize L, V with CMO

        self.L[:self.feed_stage] = self.RR * self.D
        self.L[self.feed_stage:self.N] = self.RR * self.D + self.F_feed
        self.L[self.N] = self.B
        self.V[1:] = self.RR * self.D + self.D

    def solve_component_mass_bal(self, component):
        
        A, B, C, D = make_ABC(
            self.V, self.L, self.K[component], self.F, self.z[component], self.D, self.B, self.N
        )
        self.l[component][:] = solve_diagonal(A, B, C, D)

    def h_pure_rule(self, c, T):
        """rule for liquid enthalpy of pure component"""
        return self.CpL_func[c].integral_dT(self.T_ref[c], T)

    def h_j_rule(self, stage):
        """Enthalpy of liquid on stage *j*.
        Calculated for ideal mixture

        .. math::

            h_j = \\sum_i x_{ij}h^*_i(T_j)

        where the asterisk indicates the pure component enthalpy

        :return: :math:`h_j` [J/kmol]
        """
        return sum(
            self.x_ij_expr(c, stage) * self.h_pure_rule(c, self.T[stage]) for c in self.components
        )

    def x_ij_expr(self, i, j):
        """

        :param i: component name
        :param j: stage number
        :return: mole fraction on stage
        """
        return self.l[i][j] / self.L[j]

    def h_feed_rule(self, stage):
        """Enthalpy of liquid in feed mixture
        Calculated for ideal mixture

        .. math::

            h = \\sum_i x_{ij}h^*_i(T_j)

        where the asterisk indicates the pure component enthalpy

        :return: :math:`h` [J/kmol]
        """
        return sum(
            self.z[c][stage] * self.h_pure_rule(c, self.T_feed) for c in self.components
        )

    def H_pure_rule(self, c, T):
        """Rule for vapor enthalpy of pure component"""
        return self.CpV_func[c].integral_dT(self.T_ref[c], T) + self.dH_func[c].eval()

    def H_j_rule(self, stage):
        """Enthalpy of vapor on stage *j*.
        Calculated for ideal mixture

        .. math::
            H_j = \\sum_i y_{ij}H^*_i(T_j)

        where the asterisk indicates the pure component enthalpy

        .. todo::
            convert y mole fractions to dynamic expression

        :return: :math:`H_j` [J/kmol]
        """
        return sum(
            self.y_ij_expr(c, stage) * self.H_pure_rule(c, self.T[stage]) for c in self.components
        )

    def y_ij_expr(self, i, j):
        """

        :param i: component name
        :param j: stage number
        :return: gas-phase mole fraction on stage
        """

        
        """
        TO BE FINETUNED
        
        ROOT_DIR = os.getcwd()
        f_name = os.path.join(ROOT_DIR, 'equilibrium_data', 'pengrobinson.csv')
        data = read_csv_data(f_name)
        compound_data = data[i]
        P_c = float(compound_data['Pc (Pa)'])
        T_c = float(compound_data['Tc (K)'])
        omega = float(compound_data['Omega\n'])
        
        P = self.P
        T = self.T_feed
        l_total = sum(self.l[c][stage] for c in self.components)
        z = [self.l[c][stage]/l_total for c in self.components]

        # Calculate K values for each component at the current temperature
        K_value = self.K_func[i].calculate_K(T_c, P_c, omega, self.T[j], self.P_feed)

        return K_value * self.x_ij_expr(i, j)

        """
        
    
    def solve_energy_balances(self):
        """Solve energy balances"""

        self.L_old[:] = self.L[:]
        self.V_old[:] = self.V[:]

        BE = np.zeros(self.num_stages)
        CE = np.zeros(self.num_stages)
        DE = np.zeros(self.num_stages)

        # total condenser
        BE[0] = 0.
        CE[0] = self.h_j_rule(0) - self.H_j_rule(1)
        DE[0] = self.F[0] * self.h_feed_rule(0) + self.Q_condenser_rule()

        # stages 1 to N-1
        for j in range(1, self.N):
            BE[j] = self.H_j_rule(j) - self.h_j_rule(j - 1)
            CE[j] = self.h_j_rule(j) - self.H_j_rule(j + 1)
            DE[j] = self.F[j] * self.h_feed_rule(j) - self.D * (self.h_j_rule(j - 1) - self.h_j_rule(j)) \
                    - sum(self.F[k] for k in range(j + 1)) * self.h_j_rule(j) \
                    + sum(self.F[k] for k in range(j)) * self.h_j_rule(j - 1)

        # partial reboiler
        BE[self.N] = self.H_j_rule(self.N) - self.h_j_rule(self.N - 1)
        DE[self.N] = self.F[self.N] * self.h_feed_rule(self.N) + self.Q_reboiler_rule() \
                     - self.B * (self.h_j_rule(self.N - 1) - self.h_j_rule(self.N)) \
                     - self.F[self.N - 1] * self.h_j_rule(self.N - 1)

        A = diags(
            diagonals=[BE[1:], CE[1:-1]],
            offsets=[0, 1],
            shape=(self.N, self.N),
            format='csr'
        )
        self.V[1:] = linalg.spsolve(A, DE[1:])
        self.L[0] = self.RR * self.D
        for i in range(1, self.N):
            self.L[i] = self.V[i + 1] - self.D + sum(self.F[k] for k in range(i + 1))
        self.L[self.N] = self.B

    def Q_reboiler_rule(self):
        return self.D * self.h_j_rule(0) + self.B * self.h_j_rule(self.N) \
               - self.F_feed * self.h_feed_rule(self.feed_stage) - self.Q_condenser_rule()
    
    def Q_condenser_rule(self):
        """Condenser requirement can be determined from balances around total condenser"""
        return self.D * (1. + self.RR) * (self.h_j_rule(0) - self.H_j_rule(1))


    def update_K_values(self):
        P_c_values = []
        T_c_values = []
        omega_values = []

        """Calculate the bubble-point temperature using K-values and mole fractions"""
    
        for key in self.components:
            ROOT_DIR = os.getcwd()
            f_name = os.path.join(ROOT_DIR, 'equilibrium_data', 'pengrobinson.csv')
            data = read_csv_data(f_name)
            assert key in data.keys(), f'Compound {key} not found!'
            compound_data = data[key]
            P_c = float(compound_data['Pc (Pa)'])
            T_c = float(compound_data['Tc (K)'])         
            omega = float(compound_data['Omega\n'])
            
            for i in enumerate(self.T):
                print('P_c : ',P_c)
                print('T_c : ',T_c)
                print('omega : ',omega)
                print('T : ',i[1])
                print('P : ',self.P_feed)
                self.K[key][i[0]] = self.K_func[key].calculate_K(T_c, P_c, omega, i[1], self.P_feed)
            
        print(self.K)
        self.T_old[:] = self.T[:]

    def bubble_T(self, stage):

        
        P_c_values = []
        T_c_values = []
        omega_values = []

        """Calculate the bubble-point temperature using K-values and mole fractions"""
    
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
            
        P = self.P
        T = self.T_feed
        l_total = sum(self.l[c][stage] for c in self.components)
        z = [self.l[c][stage]/l_total for c in self.components]

        # Calculate K values for each component at the current temperature
        K_values = [
            self.K_func[component].calculate_K(T_c_values[idx], P_c_values[idx], omega_values[idx], T, P)
            for idx, component in enumerate(self.components)
        ]



        # Use solver to solve for the temperature that satisfies the bubble-point condition
        T_bubble_point = self.bubble_point_eq(T_c_values,P_c_values,omega_values,T,P) 
        
        return T_bubble_point        

        
    def bubble_point_eq(self,T_c_values,P_c_values,omega_values,T,P):
            
            """
            This function calculates the bubble-point temperature using K-values and mole fractions from Tmin to Tmax.
            The bubble point condition is satisfied when the sum of the K-values multiplied by the mole fractions is equal to 1.
            """
            
            T_min = T - 30  # Minimum temperature
            T_max = T + 30  # Maximum temperature
            step = 0.1  # Iteration step 

            for T in np.arange(T_min, T_max, step):
                sum_y = 0.0
                for i, component in enumerate(self.components):
                    K_value = self.K_func[component].calculate_K(T_c_values[i], P_c_values[i], omega_values[i], T, P)
                    sum_y += K_value * self.z_feed[component]  # K_i(T) * z_i
                    print("Sum_y",sum_y)
                #  Sum close to 1.0 (bubble point conditión)
    
                if abs(sum_y - 1.0):  # Convergence Tolerance
                    print(f"Bubble temperature found: {T} K")
                    return T

            # Si no se encuentra un valor de T que cumpla la condición
    
            print(f"Bubble temperature could not be found in the {T_min} K - {T_max} K range")
            return None
    

    def bubble_T_feed(self):
        P_c_values = []
        T_c_values = []
        omega_values = []

        """Calculate the bubble-point temperature using K-values and mole fractions"""
    
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
            
        T = self.T_feed
        z = self.z_feed
        P = self.P

        # Calculate K values for each component at the current temperature
        K_values = [
            self.K_func[component].calculate_K(T_c_values[idx], P_c_values[idx], omega_values[idx], T, P)
            for idx, component in enumerate(self.components)
        ]


        # Use solver to solve for the temperature that satisfies the bubble-point condition
        T_bubble_point = self.bubble_point_eq(T_c_values,P_c_values,omega_values,T,P) 
        
        return T_bubble_point        

    def T_is_converged(self):
        """
          :return: True if T is converged, else False
        """
        eps = np.abs(self.T - self.T_old)
        return eps.max() < self.temperature_tol        

    
        
        
def make_ABC(V: np.array, L: np.array, K: np.array, F: np.array, z: np.array,
             Distillate: float, Bottoms: float, N: int):
        """
        Distillation column with partial reboiler and total condenser

        .. note::
            K_j is assumed to depend on *T* and *p*, but not composition

        :param V: vapor molar flow rate out of stage 0 to *N*
        :param L: liquid molar flow rate out of stage 0 to *N*
        :param K: equilibrium expressions for stage 0 to *N*
        :param F: feed flow rate into stage for stage 0 to *N*
        :param z: feed composition into stage for stage 0 to *N*
        :param Distillate: distillate flow rate
        :param Bottoms: bottoms flow rate
        :param N: number of equilibrium stages

        :return: A, B, C, D
        """
        B = np.zeros(N + 1)  # diagonal
        A = -1 * np.ones(N)  # lower diagonal
        C = np.zeros(N)  # upper diagonal
        D = np.zeros(N + 1)

        assert abs(V[0]) < 1e-8, 'Vapor flow rate out of total condenser is non-zero!'
        # total condenser
        B[0] = 1. + Distillate / L[0]
        C[0] = -V[1] * K[1] / L[1]
        D[0] = F[0] * z[0]
        # reboiler
        B[N] = 1 + V[N] * K[N] / Bottoms
        D[N] = F[N] * z[N]

        D[1:N] = F[1:N] * z[1:N]
        B[1:N] = 1 + V[1:N] * K[1:N] / L[1:N]
        C[1:N] = -V[2:(N + 1)] * K[2:(N + 1)] / L[2:(N + 1)]
        return A, B, C, D        

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
        P=506625, # KPa
        z_feed = [0.00005, 0.7808, 0.2095],
        T_feed_guess = 77, # Operation temperature 
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

    # Initialize T 

    for i in model.stages:
        model.T[i] = model.T_feed
        print(model.T)
    
    # Initialize Flow Rates 
    
    print(model.L)
    print(model.V)
    
    model.initialize_flow_rates()
    
    print(model.L)
    print(model.V)

    # Calculate K values
    
    model.update_K_values()

    # Solve component mass balances
    
    for i in model.components:
        print(i, model.l[i])
    for i in model.components:
        model.solve_component_mass_bal(i)
    for i in model.components:
        print(i, model.l[i])

    # Solve the bubble point for every stage 
    print("Setting bubble point for every stage")
    print(model.T)
    for stage in model.stages:
        model.T[stage] = model.bubble_T(stage)
    print(model.T)

    # Print if is converged

    print(model.T_is_converged())

    # Iteration 

    iter = 0
    while not model.T_is_converged():
        model.update_K_values()
        for i in model.components:
            model.solve_component_mass_bal(i)
        for stage in model.stages:
            model.T[stage] = model.bubble_T(stage)
        print(iter, model.T)
        iter += 1

    print(model.T_is_converged())

    """# Energy balance
    print(model.L)
    print(model.V)
    model.solve_energy_balances()
    print(model.L)
    print(model.V)"""

    # Convergence of flow rates 

    """print(model.flow_rates_converged())"""

    # Iteration process of flow rate convergence 
    """
    outer_loop = 0
    inner_loop = 0
    while not model.flow_rates_converged():
        outer_loop += 1
        for i in model.components:
            model.solve_component_mass_bal(i)
        for stage in model.stages:
            model.T[stage] = model.bubble_T(stage)
        while not model.T_is_converged():
            inner_loop += 1
            model.update_K_values()
            for i in model.components:
                model.solve_component_mass_bal(i)
            for stage in model.stages:
                model.T[stage] = model.bubble_T(stage)
        model.solve_energy_balances()
        print(outer_loop, inner_loop, model.V)"""

    # X Flows and plot behaviours

    """x = {}
    for i in model.components:
        x[i] = model.l[i][:]/model.L[:]
    print(x)

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(model.stages, model.T, 'o')
    ax.set_xlabel('Stage Number')
    ax.set_ylabel('Temperature [K]')

    # plot liquid-phase mole fractions
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    # calculate mole fractions
    for i in model.components:
        ax2.plot(model.stages, x[i], label=i)
    ax2.set_ylabel('Liquid phase mole fraction')
    ax2.set_xlabel('Stage Number')
    ax2.legend()
    """