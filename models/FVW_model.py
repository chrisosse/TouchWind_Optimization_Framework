import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio.v3 as iio
import time
import math

class FVW_model:
    '''
    _summary_

    Returns
    -------
    _type_
        _description_
    '''

    def __init__(
        self,
        model_params,
    ):
        #####################################################################
        #                         INPUT VARIABLES                           #
        #####################################################################
        # Model parameters
        self.n_r = model_params['n_r'] # number of vortex rings
        self.n_e = model_params['n_e'] #number of element on each ring
        self.n_p = model_params['n_p'] #number of points on each ring            #!!! IN PAPER IS SAYS Np = Ne + 1
        
        self.k_tot = 100 #number of discrete simulation time steps
        self.stretching = False #vortex filament strechting correction !only applicable if diffusion is True 
        self.ground_option = 2 #(0 = none, 1 = mirror turbine, 2 = solid ground)
        self.diffusion = True #vortex diffusion and stretching

        # diffusion variables
        self.alpha = 1.256443
        self.nu = 1.5e-5
        self.a1 = 0.005 #[orde: 0.0010 - 0.0001]
    
        #plotting options
        self.make_animation = False #make animation of ring evolution !! slows down the code
        self.continuous_power_calculation = False #calculationg power at every time step/ false: only at last time step
        self.plot_rings = True
        self.compare_les = False

        # LES variables
        self.file_name = 'Uniform20LES.csv'

        # Print statement variables
        self.print_stuff = model_params['print_stuff']

        # Run model instantly without need to set conditions, layout and turbines
        self.run_directly = model_params['run_directly']

        if self.run_directly:
            self.set_wind_conditions()
            self.set_wind_farm_layout()
            self.set_turbine_properties()
            self.run_model()
        

    def set_wind_conditions(
        self,
        direction: float = 270.,
        speed: float = 10.,
        ABL_params: dict = None,

    ):  
        # Assumption that is stays constant over time
        
        # Get wind direction in radians
        direction_rad = math.radians(direction % 270) # [rad]
        
        # Get velocity components in x, y and z
        self.x_inflow = speed * np.cos(direction_rad) * np.cos(direction_rad) # [m/s]
        self.y_inflow = speed * np.cos(direction_rad) * np.sin(direction_rad) # [m/s]
        self.z_inflow = 0 # [m/s]
        
        # Create inflow matrix
        self.inflow = np.zeros([self.k_tot, self.n_p, 3])
        self.inflow[:, :, 0] = self.x_inflow
        self.inflow[:, :, 1] = self.y_inflow
        self.inflow[:, :, 2] = self.z_inflow

        # Unit vector in downstread direction [-]
        self.e_x = np.array([ 
            np.cos(direction_rad),
            np.sin(direction_rad),
            0,
        ]) 

    def set_wind_farm_layout(
        self, 
        turbine_positions: list = [[500, 315, 90]],
    ):
        self.position_turbines = turbine_positions #position turbine [m]
        self.n_pt = len(turbine_positions) #number of physical turbines in the system 


    def set_turbine_properties(
        self,
        yaw_angles: list = [0],
        tilt_angles: list = [0],
        axial_inductions: list = [0.27],
        rotor_diameters: list = [126],
    ):
        # TODO Make rotor diameter variable across different turbines?
        self.psi = yaw_angles # yaw angle [deg]
        self.phi = tilt_angles # tilt angle [deg]
        self.a = axial_inductions # axial induction  
        self.r = rotor_diameters[0] / 2 # blade radius [m]
        self.rotor_diamaters = rotor_diameters

        self.h = 0.3 * (2 * self.r) / self.x_inflow #simulation time step [sec]
        self.sigma = 0.32 * self.r  


    #####################################################################
    #                     TEST FUNCTIONS DEFINITIONS                    #
    #####################################################################
    def run_tests(self):
        self.validate_input_values(
            self.n_r, 
            self.n_p, 
            self.n_e, 
            self.n_pt,
            self.k_tot, 
            self.h, 
            self.position_turbines, 
            self.psi, 
            self.phi, 
            self.r, 
            self.a, 
            self.sigma,
        )

        #if no errors occur
        if self.print_stuff:
            print('All tests passed :)')

    def validate_input_values(
        self,
        n_r, 
        n_p, 
        n_e, 
        n_pt, 
        k_tot, 
        h, 
        position_turbines, 
        psi,
        phi,
        r, 
        a, 
        sigma
    ):
        int_float_types = (int, float, np.int32, np.float64)

        assert isinstance(n_r, int), "n_r should be an int."
        assert isinstance(n_p, int), "n_p should be an int."
        assert isinstance(n_e, int), "n_e should be an int."
        assert isinstance(n_pt, int), "n_pt should be an int."
        
        assert n_r > 0, "n_r should be greater than 0."
        assert n_p > 0, "n_p should be greater than 0."
        assert n_e > 0, "n_e should be greater than 0."
        assert n_pt > 0, "n_pt should be greater than 0."
        assert n_pt == len(position_turbines)==len(psi)==len(phi)==len(a), "number of turbines should be equal to number of inputs given."

        assert isinstance(k_tot, int), "k_tot should be an int."
        assert isinstance(h, int_float_types), "h should be an int or float."
        
        assert k_tot > 0, "k_tot should be greater than 0."
        
        assert all(len(position_turbines[x]) == 3 for x in range(n_pt)), "position_turbine should be a list of 3 elements."
        assert all(all(isinstance(x, int_float_types) for x in position_turbines[xi]) for xi in range(n_pt)), "All elements of position_turbine should be int or float."
        assert all(position_turbines[x][2]>0 for x in range(n_pt)), "The turbine should be above sea level"
        assert all(isinstance(x, int_float_types) for x in psi), "psi should be an int or float."
        assert all(isinstance(x, int_float_types) for x in phi), "phi should be an int or float."
        
        assert isinstance(r, int_float_types), "r should be an int or float."
        
        assert r > 0, "r should be greater than 0."

        assert all(isinstance(a[x], int_float_types) for x in range(n_pt)), "a should be an int or float."
        assert all(0 <= a[x] <= 1 for x in range(n_pt)), "a should be between 0 and 1."

        assert isinstance(sigma, int_float_types), "sigma should be an int or float."
        #assert 0 <= sigma <= 1, "sigma should be between 0 and 1."

        if self.print_stuff:
            print("Warning: it is adviced to make n_r not larger than 40" if n_r > 41 else "number of rings per turbine =", n_r)
            print("Warning: it is adviced to make n_p not larger than 16" if n_p > 17 else "number of points per ring =", n_p)
            print("Warning: it is adviced to make n_e not larger than 16" if n_e > 17 else "number of elements per ring  =", n_e)  
            print("Warning: it is adviced to make k_tot larger than 100 for numerical convergence" if k_tot < 99 else "number of simulation time steps =", k_tot)


    #####################################################################
    #                       FUNCTIONS DEFINITIONS                       #
    #####################################################################
    # ABL functions -----------------------------------------------------
    # @return u_fric    Friction velocity [1 x 3]
    def get_friction_velocity(
        self,
    ):
        K = 0.40
        z0 = 0.0002
        u_fric = K * (self.u_ref / np.log((self.H_ref + z0) / z0))  
        return u_fric

    # @param z          Height in which the velocity is calculated [n_p x 1]
    # @return inflow    Inflow velocity at z [n_p x 3]
    def get_abl(
        self,
        z,
    ):
        K = 0.40
        z0 = 0.0002

        z = np.where(z < 0, 0.001, z)
        inflow = (self.u_fric / K) * np.log((z + z0) / z0)[:, np.newaxis]
        return inflow

    # ring functions ----------------------------------------------------
    # @param phi    Yaw angle [-]
    # @param psi    Tilt angle [-]
    # @return R_zy  Rotation matrix of first tilt then yaw [3 x 3]
    def get_rotation_matrix(
        self,
        phi,
        psi,
    ):  
        Rz = np.array([
            [np.cos(np.deg2rad(psi)), np.sin(np.deg2rad(psi)), 0],
            [-np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi)), 0],
            [0, 0, 1]])
        Ry = np.array([
            [np.cos(np.deg2rad(phi)), 0, np.sin(np.deg2rad(phi))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(phi)), 0, np.cos(np.deg2rad(phi))]])
        R_zy = Ry@Rz
        return R_zy

    # @param turbine        Index turbine   [-]     
    # @return X0            Position initial ring  [n_p x 3]
    def calc_init_ring_position(
        self,
        turbine,
    ):
        R_zy = self.get_rotation_matrix(self.phi[turbine], self.psi[turbine])
        i_values = np.arange(0, self.n_p)
        X0 = np.array([R_zy @ np.array([
            0, 
            self.r * np.cos(2 * np.pi * (i / self.n_e)),
            self.r * np.sin(2 * np.pi * (i / self.n_e))
        ]) for i in i_values]) + self.position_turbines[turbine] 
                                        
        return X0



    # @param k          number of time step
    # @param Xk         State of position previous time step [X[n_t x n_r x n_p x 3]]
    # @param iturbine   Index turbine q_k [-]
    # # @param ring     Index ring [-]
    # @return U0        Inflow velocity/free stream velocity init ring at time step k [n_p x 3]
    def calc_init_ring_velocity(
        self,
        Xk,
        iturbine,
        ring,
    ):
        U0 = self.inflow[self.k]
        return U0

    # @param U0         Inflow velocity init ring [n_p x 3]
    # @param turbine    Index turbine [-]
    # @param iturbine   Index turbine in q_k [-]
    # @return G0        Vortex strenght of init ring [n_e]
    def calc_init_ring_vortex_strength(
        self,
        U0,
        turbine,
        iturbine,
        X0,
    ): 
        n = self.get_rotation_matrix(self.phi[turbine], self.psi[turbine]) @ self.e_x  #[1x3]

        if self.k == 0:
            u_r = self.calc_average_wind_speed_at_rotor(U0) # [1x3]
            c_t = self.calc_trust_coef(turbine)

        else:
            u_r = self.calc_rotor_disc_avg_velocity(turbine, iturbine)
            c_t = self.calc_local_trust_coef(turbine)

        G0_element = self.h * c_t * 0.5 * (np.dot(u_r, n))**2 #[-]
        G0 = np.tile(G0_element, (self.n_e, 1)) #[n_e]

        return G0

    # @param turbine    Index turbine [-]
    # @return c_t       Trust coefficient [-]
    def calc_trust_coef(
        self,
        turbine,
    ):
        c_t1 = 2.3
        a_t = 1 - 0.5 * np.sqrt(c_t1)

        if self.a[turbine] <= a_t:
            c_t = (4 * self.a[turbine]) * (1 - self.a[turbine])
        else: 
            c_t = (c_t1 - 4 * (np.sqrt(c_t1) - 1) * (1 - self.a[turbine]))
        
        return c_t

    # @param turbine    Index turbine [-]
    # @return c_t'      Local_rust coefficient [-]
    def calc_local_trust_coef(
        self,
        turbine,
    ):
        c_t1 = 2.3
        a_t = 1 - 0.5 * np.sqrt(c_t1)

        if self.a[turbine] <= a_t:
            c_t = (4 * self.a[turbine]) / (1 - self.a[turbine])
        else: 
            c_t = (c_t1 - 4 * (np.sqrt(c_t1) - 1) * (1 - self.a[turbine])) / (1 - self.a[turbine])**2
        
        return c_t

    # @param U0     Inflow velocity init ring [np x 3]
    # @return u_r   Average wind speed at rotor [1x3]
    def calc_average_wind_speed_at_rotor(
        self,
        U0,
    ):
        u_r = sum(U0 / (self.n_p))
        return u_r

    # @param q_k        State matrix of previous time step [[X[n_t x n_r x n_p x 3]],[U [n_t x n_r x n_p x3]],[G [n_t x n_r x n_e x 1]]]
    # @param points     Points of interest [nr_points x 3]
    # @return u_ind     Induced velocity at disc [nr_points x 3]
    def calc_induced_velocity(
        self,
        q_k,
        points,
    ):
        nr_points = len(points)
        u_ind = np.zeros([nr_points,3])

        x0 = points[:, np.newaxis, np.newaxis, np.newaxis, :]
        x1 = q_k[0][np.newaxis, :, :, :, :]
        x2 = np.roll(q_k[0], -1, axis=2)[np.newaxis, :, :, :, :]
        G = q_k[2][np.newaxis, :, :, :]

        r0 = x2 - x1
        r1 = x1 - x0
        r2 = x2 - x0

        norm_r0 = np.linalg.norm(r0, axis=-1, keepdims=True) #norm r0 for all elements
        norm_r1 = np.linalg.norm(r1, axis=-1, keepdims=True) #norm r1 for all elements
        norm_r2 = np.linalg.norm(r2, axis=-1, keepdims=True) #norm r2 for all elements

        cross_r1_r2 = np.cross(r1, r2)
        norm_cross_r1_r2 = np.linalg.norm(cross_r1_r2, axis=-1, keepdims=True)    

        #evolution of term1
        term1 = np.true_divide(cross_r1_r2, norm_cross_r1_r2**2, where=norm_cross_r1_r2 != 0)
        term1 *= (G / (4 * np.pi))

        #evolution of term2
        term2 = (r1 / norm_r1 - r2 / norm_r2)
        term2 = np.where(np.isnan(term2), 0, term2)
        term2 = np.sum(r0 * term2, axis=-1, keepdims=True)

        sigma_elements = self.get_sigma(norm_r0.squeeze(-1))
        term3 = 1 - np.exp(-(norm_cross_r1_r2.squeeze(-1)**2) / (sigma_elements**2 * norm_r0.squeeze(-1)**2))

        u_ind = np.sum(term1*term2*term3[...,np.newaxis],axis=(1,2,3))
        return u_ind

    # @param element_lenghts    Lenghts of all elements of interest
    # @return sigma_array       Gausian core size per element per ring per turbine
    def get_sigma(
        self,
        element_lenghts,
    ):
        nr_rings = len(element_lenghts[0][0])
        if self.diffusion == True:
            delta_t = (np.arange(nr_rings)) * self.h
            sigma_diffusion = np.sqrt(4 * self.alpha * self.delta * self.nu * delta_t + self.sigma**2)
            sigma_array = sigma_diffusion[np.newaxis, np.newaxis, :, np.newaxis]

            if self.stretching == True:
                init_element_lenght = element_lenghts[0][0][0][0]  #assume all elements have same initial lenght
                sigma_stretching = np.sqrt((init_element_lenght / element_lenghts)) 
                sigma_array = sigma_array * sigma_stretching
        else:
            sigma_array = np.ones((len(element_lenghts), self.n_t, nr_rings, self.n_p)) * self.sigma
        return sigma_array

    # @param q_k        States of previous time step [[X[n_t x n_r x n_p x 3]],[U [n_t x n_r x n_p x3]],[G [n_t x n_r x n_e x 1]]]
    # @param ring       Index ring [-]
    # @param iturbine   Index turbine q_k [-]
    # @return G         Vortex strenght at current time step [n_e]
    def calc_vortex_strength(
        self,
        q_k,
        ring, 
        iturbine
    ):
        G = q_k[2][iturbine][ring-1]
        return G

    # @param q_k        States of previous time step [[X[n_t x n_r x n_p x 3]],[U [n_t x n_r x n_p x3]],[G [n_t x n_r x n_e x 1]]]
    # @param ring       Index ring [-]
    # @param turbine    Index turbine q_k [-]
    # @return X         Ring position at current step [n_p x 3]
    def calc_ring_position(
        self,
        q_k,
        ring, 
        iturbine,
    ):
        u_inf_k = self.calc_ring_velocity(q_k, ring, iturbine, q_k[0][iturbine][ring-1])
        u_ind_k = self.calc_induced_velocity(q_k, q_k[0][iturbine][ring-1])
        X = q_k[0][iturbine][ring-1] + self.h * (u_inf_k + u_ind_k) 

        if self.ground_option == 2:
            X[:, 2] = np.where(X[:, 2]<=0, 0, X[:, 2])
        return X

    # @param q_k        States of previous time step [[X[n_t x n_r x n_p x 3]],[U [n_t x n_r x n_p x3]],[G [n_t x n_r x n_e x 1]]]
    # @param ring       Index ring [-]
    # @param iturbine   Index turbine q_k[-]
    # @param points     Points of interest
    # @return U         Ring velocity at current step [n_p x 3]
    def calc_ring_velocity(
        self,
        q_k,
        ring, 
        iturbine,
        points,
    ):
        U = q_k[1][iturbine][ring-1]
        return U

    # @param turbine    Index turbine [-]
    # @return P_i       Power of the turbine [-]
    def calc_power(
        self,
        turbine, 
        iturbine,
    ):
        c_p = self.calc_local_power_coef(turbine) 
        A_r = np.pi * self.r**2 
        n = self.get_rotation_matrix(self.phi[turbine],self.psi[turbine]) @ self.e_x  
        u_r = self.calc_rotor_disc_avg_velocity(turbine, iturbine)
        P_i = 0.5 * c_p * A_r * np.dot(u_r, n)**3                   
        return P_i

    # @param turbine    Index turbine [-]
    # @return c_p       Local power coefficient [-]
    def calc_local_power_coef(
        self,
        turbine,
    ):
        c_p = (4 * self.a[turbine]) / (1 - self.a[turbine])
        return c_p

    # @param turbine    Index turbine [-]
    # @return u_r       Rotor disc averaged velocity [1 x 3]
    def calc_rotor_disc_avg_velocity(
        self,
        turbine, 
        iturbine,
    ):
        n = 3                                                      #!!! OVERGENOMEN VAN VD BROEK
        points = self.get_isocell(n, turbine)                        # [n_u x n_d]    
        n_u = len(points)   

        u_inf = self.calc_disc_local_free_stream_flow(self.q_k, points, n_u, iturbine) 
        u_ind = self.calc_induced_velocity(self.q_k, points)
        sum_u = np.sum(u_inf + u_ind, axis=0)

        u_r = sum_u / n_u          
        return u_r

    # @param n          Number of concentric rings in the disc
    # @param tubrine    Index turbine in positions_turbine, phi and psi
    # @return points    x, y, and z coordinates of the points on the disc
    def get_isocell(
        self,
        n,
        turbine,
    ): 
        N1 = 3 #Number of cells in the inner ring of the disc             #!!! OVERGENOMEN VAN VD BROEK
        i = np.arange(n) + 1
        Ni = (2 * i - 1)*N1
        points = np.zeros((Ni.sum(), 3))    #[n_u x 3]
        ri = (i - 0.5) * (self.r / n)
        Thi = np.concatenate([np.linspace(0, 2 * np.pi, n, endpoint=False) for n in Ni])
        R = np.repeat(ri, Ni)
        points = np.zeros((Ni.sum(), 3))                                     
        points[:, 1] = R * np.cos(Thi)
        points[:, 2] = R * np.sin(Thi) 

        #To include tilt and yaw
        R_zy = self.get_rotation_matrix(self.phi[turbine], self.psi[turbine])
        points = points@R_zy.T + self.position_turbines[turbine]
        return points

    # @param q_k        State matrix of previous time step [[X[n_t x n_r x n_p x 3]],[U [n_t x n_r x n_p x3]],[G [n_t x n_r x n_e x 1]]]
    # @param points     Points on disc [n_u x 3]
    # @param n_u        Number of points on disc [-]
    # @param turbine    Index turbine [-]
    # @return u_inf     local free stream flow velocity [n_u x 3]
    def calc_disc_local_free_stream_flow(
        self,
        q_k,
        points,
        n_u,
        iturbine,
    ):
        u_inf = np.zeros([n_u,3])
        nr_rings = len(q_k[0][iturbine])
        for i in range(0,n_u): #for all points on disc
            x = points[i]
            u_inf_i = q_k[1][iturbine][:]
            weights = self.calc_norm_weight(q_k, x, iturbine)
            #reshape
            weights = (np.repeat(weights,3)).reshape(nr_rings, self.n_p, 3) 
            u_inf[i] = np.sum(np.sum(weights * u_inf_i, axis=0), axis=0) 

        return u_inf
    
    # @param q_k        State matrix of previous time step [[X[n_t x n_r x n_p x 3]],[U [n_t x n_r x n_p x3]],[G [n_t x n_r x n_e x 1]]]
    # @param x          Point of interest [1 x 3]
    # @param turbine    Index turbine [-]
    # @return norm_w    Normalized weight [n_r x n_p]
    def calc_norm_weight(
        self,
        q_k, 
        x, 
        iturbine,
    ):
        xi = q_k[0][iturbine][:]
        w_ib = np.exp(-10 * np.linalg.norm(x-xi, axis=-1))
        sum_wib = np.sum(w_ib)
        norm_wib = w_ib / sum_wib
        return norm_wib

    # ground effect mirror turbine functions -----------------------------------------------------------
    # @param G              Vortex strength rings physical turbine
    # @return G_mirror      Vortex strength rings mirror turbine     
    def mirror_ring_vortex_strenght(
        self,
        G,
    ):
        G_mirror = G.copy()
        G_mirror = -G_mirror
        return G_mirror

    # @param U              Velocity rings physical turbine
    # @return U_mirror      Velocity rings mirror turbine  
    def mirror_ring_velocity(
        self,
        U,
    ):
        U_mirror = U.copy()
        U_mirror[:, :, 2] = -U_mirror[:, :, 2]
        return U_mirror

    # @param X              Position rings strength physical turbine
    # @return X_mirror      Position rings mirror turbine  
    def mirror_ring_position(
        self,
        X,
    ):
        X_mirror = X.copy()
        X_mirror[:, :, 2] = -X_mirror[:, :, 2]  
        return X_mirror

    # plotting functions -------------------------------------------------------------------------------
    def make_frame(
        self,
        q_k,
        k,
    ):
        ax = self.fig.add_subplot(1, 1, 1, projection='3d', proj_type='persp')
        for turbine in range(self.n_t):
            for ring in range(min(k, self.n_r)):
                px = q_k[0][turbine][ring][:, 0]
                py = q_k[0][turbine][ring][:, 1]
                pz = q_k[0][turbine][ring][:, 2]
                
                px = np.append(px, px[0])
                py = np.append(py, py[0])
                pz = np.append(pz, pz[0])

                if ring == 0:
                    ax.plot(px / (2 * self.r), py / (2 * self.r), pz / (2 * self.r), lw=1, c='r')
                else:
                    ax.plot(px / (2 * self.r), py / (2 * self.r), pz / (2 * self.r), lw=1, c='k')

        ax.set_xlim(0, 30)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect([30, -1.1, 1.1])
        ax.view_init(elev=10, azim=-145)
        ax.set_xticks(np.arange(0, 30, 1))
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])

        ax.set_xlabel("x (x/D)")
        ax.set_ylabel("y (y/D)")
        ax.set_zlabel("z (z/D)")
        plt.savefig(f'frame_{k}.png')

        return

    # @param q_k            State matrix of last time step [n_r x [[X[n_p x 3]],[U [n_p x3]],[G [n_e x 1]]]
    # @param points         Location of the grid points
    # @param nr_points      Number of points in the grid [n_gp x 3]
    # @return velocity      Velocity of the grid
    def get_velocity_grid(
        self,
        q_k,
        points,
        nr_points,
    ):
        velocity = self.calc_induced_velocity(q_k, points) + self.inflow[self.k][0]
        return velocity

    #####################################################################
    #                            MAIN CODE                              #
    #####################################################################
    
    
    
    def run_model(
        self,
    ):
        self.run_tests()
        time_start = time.process_time()
        
        #initialiseren
        self.power = np.zeros((self.n_pt, self.k_tot)) # power array
        self.Xk0 = np.zeros((self.n_pt, self.n_p, 3)) # positions first rings
        self.delta = 0

        for k in range(0, self.k_tot): #for every discrete time step
            self.k = k
            
            if self.print_stuff:
                print('nr_time_step =', k)

            if self.ground_option == 1: #include ground effects
                self.n_t = self.n_pt * 2
                Xk = np.zeros((self.n_t, min(k + 1, self.n_r), self.n_p, 3)) #increases in size as nr of rings build to n_r   [2n_t x [nr x [n_p x 3]]
                Uk = np.zeros((self.n_t, min(k + 1, self.n_r), self.n_p, 3)) 
                Gk = np.zeros((self.n_t, min(k + 1, self.n_r), self.n_e, 1))  

                iturbine = 0 # [0, 2, 4, etc] are physical turbines , [1,3,5, etc] are mirror turbines
                for turbine in range(self.n_pt):
                    if k == 0:
                        self.Xk0[turbine] = self.calc_init_ring_position(turbine)
            
                    for ring in range(0, min(k + 1, self.n_r)):
                        if ring == 0: #first ring

                            Xk[iturbine][0] = self.Xk0[turbine] #assume angles are constant in time [np x 3]
                            Uk[iturbine][0] = self.calc_init_ring_velocity(Xk, iturbine, 0) #[np x 3]  
                            Gk[iturbine][0] = self.calc_init_ring_vortex_strength(Uk[iturbine][0], turbine, iturbine, Xk[turbine][0]) #assume a is constant in time [n_e]

                        else:
                            Xk[iturbine][ring] = self.calc_ring_position(self.q_k, ring, iturbine) #[n_p x 3]
                            Uk[iturbine][ring] = self.calc_ring_velocity(self.q_k, ring, iturbine, Xk[iturbine][ring]) #[n_p x 3]
                            Gk[iturbine][ring] = self.calc_vortex_strength(self.q_k, ring, iturbine) #[n_e]
                    iturbine += 1

                    # mirror in z plane
                    Gk[iturbine] = self.mirror_ring_vortex_strenght(Gk[iturbine-1])
                    Uk[iturbine] = self.mirror_ring_velocity(Uk[iturbine-1])
                    Xk[iturbine] = self.mirror_ring_position(Xk[iturbine-1])  

                    iturbine += 1

                self.q_k = [Xk[:], Uk[:], Gk[:]] #save states for next time step 

                if self.continuous_power_calculation == True:
                    iturbine = 0 # [0, 2, 4, etc] are physical turbines , [1,3,5, etc] are mirror turbines
                    for turbine in range(self.n_pt):
                        self.power[turbine][k] = self.calc_power(turbine, iturbine)
                        iturbine += 2

                self.delta = 1 + self.a1 * (Gk[0][0][0] / self.nu)

            else: #No ground effects

                self.n_t = self.n_pt
                Xk = np.zeros((self.n_t, min(k + 1, self.n_r), self.n_p, 3)) #increases in size as nr of rings build to n_r   [n_t x [nr x [n_p x 3]]
                Uk = np.zeros((self.n_t, min(k + 1, self.n_r), self.n_p, 3)) 
                Gk = np.zeros((self.n_t, min(k + 1, self.n_r), self.n_e, 1))

                for turbine in range(self.n_t):
                    for ring in range(0, min(k + 1, self.n_r)):
                        if ring == 0: #first ring
                            Xk[turbine][ring] = self.calc_init_ring_position(turbine) #assume angles are constant in time [np x 3]
                            Uk[turbine][ring] = self.calc_init_ring_velocity(Xk, turbine, ring) #[np x 3]
                            Gk[turbine][ring] = self.calc_init_ring_vortex_strength(Uk[turbine][ring], turbine, turbine, Xk[turbine][ring]) #assume a is constant in time [n_e]

                        else:
                            Gk[turbine][ring] = self.calc_vortex_strength(self.q_k, ring, turbine) #[n_e]
                            Xk[turbine][ring] = self.calc_ring_position(self.q_k, ring, turbine) #[n_p x 3]
                            Uk[turbine][ring] = self.calc_ring_velocity(self.q_k, ring, turbine, Xk[turbine][ring]) #[n_p x 3]
                            

                self.q_k = [Xk[:], Uk[:], Gk[:]] #save states for next time step

                if self.continuous_power_calculation == True:
                    self.power[turbine][k] = self.calc_power(turbine, turbine)

            end_time = time.process_time()

            if self.make_animation == True:
                if k == 0:
                    self.fig = plt.figure(figsize=(8, 8))
                
                self.make_frame(self.q_k, k)

        if self.print_stuff:
            print('time taken', end_time - time_start)


    def get_turbine_powers(
        self,
    ):  
        self.power = np.zeros((self.n_pt))

        if self.ground_option == 1: #include ground effects
            for turbine in range(self.n_pt):
                self.power[turbine] = self.calc_power(turbine, turbine*2)
        else:
            for turbine in range(self.n_pt):
                self.power[turbine] = self.calc_power(turbine, turbine)
        
        return self.power


    def find_nearest(
        self,
        array, 
        value
    ):
        idy = (np.abs(array-value)).argmin()
        return array[idy]
    

    def plot_shizzle(
        self,
    ):
        ################################################
        #               plotting rings                 #
        ################################################

        if self.plot_rings == True:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d', proj_type='persp')
            
            for turbine in range(self.n_pt):
                if self.ground_option == 1:
                    iturbine = turbine * 2
                else:
                    iturbine = turbine

                for ring in range(min(self.k_tot, self.n_r)):
                    px = self.q_k[0][iturbine][ring][:, 0]
                    py = self.q_k[0][iturbine][ring][:, 1]
                    pz = self.q_k[0][iturbine][ring][:, 2]
                    
                    px = np.append(px, px[0])
                    py = np.append(py, py[0])
                    pz = np.append(pz, pz[0])

                    if ring == 0:
                        ax.plot(px / (2 * self.r), py / (2 * self.r), pz / (2 * self.r), lw=1, c='r')
                    else:
                        ax.plot(px / (2 * self.r), py / (2 * self.r), pz / (2 * self.r), lw=1, c='k')
                        
            ax.set_xlim(0, 10)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(0, 1.1)
            ax.set_box_aspect([10, 1.1, 1.1])
            ax.view_init(elev=10, azim=-145)

            ax.set_xticks(np.arange(-1, 10.1, 1))
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([0, 1])
            ax.set_xlabel("x/D (-)")
            ax.set_ylabel("y/D (-)")
            ax.set_zlabel("z/D (-)")
            plt.show()

        #################################################################
        #               PLotting power over time                        #
        #################################################################
        if self.continuous_power_calculation == True:   
            if self.print_stuff:
                print(self.power[:, self.k_tot - 1])
                
            fig,ax = plt.subplots(1, 1, figsize=(8, 4))
            t = np.arange(self.k_tot) * self.h

            for turbine in range(self.n_pt):
                ax.plot(t, self.power[turbine], label=('turbine', {turbine}))

            ax.legend(loc=0)
            ax.set_xlabel("time (-)")
            ax.set_ylabel("power (-)")
            ax.set_xlim(0, t[-1])   
            plt.show()

        else:
            self.power = np.zeros((self.n_pt))
            if self.ground_option == 1: #include ground effects
                iturbine = 0
                for turbine in range(self.n_pt):
                    self.power[turbine] =  self.calc_power(turbine, iturbine)
                    iturbine += 2
            else:
                for turbine in range(self.n_pt):
                    self.power[turbine] =  self.calc_power(turbine, turbine)
            
            if self.print_stuff:
                print(self.power)


        #####################################################################
        #               PLOTTING AND FUCTIONS FOR PLOTTING                  #
        #####################################################################
        if  self.make_animation == True:
            frames = np.stack([iio.imread(f"frame_{x}.png") for x in range(self.k_tot)], axis=0)
            iio.imwrite('ring_evolution.gif', frames)

        plt.close()

        ################################################################
        #                   READ IN LES RESULTS                        #
        ################################################################
        if  self.compare_les == True:
            import pandas as pd
            colors = ['#F79123', '#014043', '#059589', '#19F7E2']

            # Read calibration cases file
            df_calibration_case = pd.read_csv(self.file_name)

            

            ##################################################################
            #               PLOTTEN DRIELUIK VELOCITY AT HUB                 #
            ##################################################################
            height = 90
            height_LES = self.find_nearest(df_calibration_case['Points:2'], height)
            df_sliced_LES = df_calibration_case.loc[
                (df_calibration_case['Points:2'] == height_LES) & 
                (df_calibration_case['Points:1'] < (500 + 2.5 * self.r)) & 
                (df_calibration_case['Points:1'] > (500 - 2.5 * self.r))
            ]

            df_sliced_LES['VelocityNorm'] = np.linalg.norm(df_sliced_LES[['UMean:0', 'UMean:1', 'UMean:2']], axis=1)
            velocityLES = df_sliced_LES.pivot_table(values='VelocityNorm', index='Points:1', columns='Points:0').values

            xcoords = np.sort(df_sliced_LES['Points:0'].unique())
            ycoords = np.sort(df_sliced_LES['Points:1'].unique())

            fig,ax = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
            xgrid, ygrid = np.meshgrid(xcoords, ycoords)
            xgrid = np.swapaxes(xgrid, 0, 1)
            ygrid = np.swapaxes(ygrid, 0, 1)
            levels = np.linspace(0, self.x_inflow, 10)


            #FVW model results
            ax[0].set_title('Free Wake Vortex Model')
            ax[0].set_ylabel("y/D (-)")
            ax[0].set_aspect("equal")
            points = np.array([xgrid,ygrid, np.ones_like(xgrid) * height_LES]).reshape(3, -1).T
            velocity = self.get_velocity_grid(self.q_k, points, len(points))
            velocity_norm = np.linalg.norm(velocity, axis=1)
            velocity_norm = velocity_norm.reshape(len(xcoords), len(ycoords))
            c = ax[0].contourf(xgrid / (2 * self.r), ygrid / (2 * self.r), velocity_norm, cmap='Blues_r', levels=levels, extend="both")
            divider = self.make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            # LES results
            ax[1].set_title('Large Eddy Simulations')
            ax[1].set_ylabel("y/D (-)")
            ax[1].set_aspect("equal")
            c = ax[1].contourf(xcoords / (2 * self.r), ycoords / (2 * self.r), velocityLES, cmap='Blues_r', levels=levels, extend="both")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            # difference
            ax[2].set_title('Difference between FVW and LES')
            velocity_difference = abs(velocityLES.T - velocity_norm)
            c = ax[2].contourf(xgrid / (2 * self.r), ygrid / (2 * self.r), velocity_difference, cmap='Blues_r', levels=levels, extend="both")
            ax[2].set_xlabel("x/D (-)")
            ax[2].set_ylabel("y/D (-)")
            ax[2].set_aspect("equal")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v_dif (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            plt.show()


            ##################################################################
            #               PLOTTEN DRIELUIK VELOCITY AT 9D                  #
            ##################################################################
            depth = 9 * (2 * self.r)
            depth_LES = self.find_nearest(df_calibration_case['Points:0'], depth)
            df_sliced_LES = df_calibration_case.loc[
                (df_calibration_case['Points:0'] == depth_LES) & 
                (df_calibration_case['Points:2'] < 1.5 * (2 * self.r)) & 
                (df_calibration_case['Points:1'] < (500 + 2.5 * self.r)) & 
                (df_calibration_case['Points:1'] > (500 - 2.5 * self.r))
            ]

            df_sliced_LES['VelocityNorm'] = np.linalg.norm(df_sliced_LES[['UMean:0','UMean:1','UMean:2']], axis=1)
            velocityLES = df_sliced_LES.pivot_table(values='VelocityNorm', index = 'Points:2', columns='Points:1').values

            zcoords = np.sort(df_sliced_LES['Points:2'].unique())
            ycoords = np.sort(df_sliced_LES['Points:1'].unique())

            fig,ax = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
            ygrid, zgrid = np.meshgrid(ycoords, zcoords)
            ygrid = np.swapaxes(ygrid, 0, 1)
            zgrid = np.swapaxes(zgrid, 0, 1)

            levels = np.linspace(0, self.x_inflow, 10)


            #FVW model results
            ax[0].set_title('Free Wake Vortex Model')
            ax[0].set_ylabel("z/D (-)")
            ax[0].set_aspect("equal")
            points = np.array([np.ones_like(zgrid) * depth_LES, ygrid, zgrid]).reshape(3, -1).T
            velocity = self.get_velocity_grid(self.q_k, points, len(points))
            velocity_norm = np.linalg.norm(velocity, axis=1)
            velocity_norm = velocity_norm.reshape(len(ycoords), len(zcoords))
            c = ax[0].contourf(ygrid / (2 * self.r), zgrid / (2 * self.r), velocity_norm, cmap='Blues_r', levels=levels, extend="both")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            # LES results
            ax[1].set_title('Large Eddy Simulations')
            ax[1].set_ylabel("z/D (-)")
            ax[1].set_aspect("equal")
            c = ax[1].contourf(ycoords / (2 * self.r), zcoords / (2 * self.r), velocityLES, cmap='Blues_r', levels=levels, extend="both")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            # difference
            ax[2].set_title('Difference between FVW and LES')
            velocity_difference = abs(velocityLES.T - velocity_norm)
            c = ax[2].contourf(ygrid / (2 * self.r), zgrid / (2 * self.r), velocity_difference, cmap='Blues_r', levels=levels, extend="both")
            ax[2].set_xlabel("y/D (-)")
            ax[2].set_ylabel("z/D (-)")
            ax[2].set_aspect("equal")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v_dif (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            plt.show()

            #############################################
            #                  Z-X plot                 #
            #############################################
            depth = self.position_turbines[0][1]
            depth_LES = self.find_nearest(df_calibration_case['Points:1'], depth)
            df_sliced_LES = df_calibration_case.loc[
                (df_calibration_case['Points:1'] == depth_LES) & 
                (df_calibration_case['Points:2'] < 1.5 * (2 * self.r))
            ]

            df_sliced_LES['VelocityNorm'] = np.linalg.norm(df_sliced_LES[['UMean:0', 'UMean:1', 'UMean:2']], axis=1)
            velocityLES = df_sliced_LES.pivot_table(values='VelocityNorm', index = 'Points:2', columns='Points:0').values

            zcoords = np.sort(df_sliced_LES['Points:2'].unique())
            xcoords = np.sort(df_sliced_LES['Points:0'].unique())

            fig,ax = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
            xgrid, zgrid = np.meshgrid(xcoords, zcoords)
            xgrid = np.swapaxes(xgrid, 0, 1)
            zgrid = np.swapaxes(zgrid, 0, 1)

            levels = np.linspace(0, self.x_inflow, 10)


            #FVW model results
            ax[0].set_title('Free Wake Vortex Model')
            ax[0].set_ylabel("z/D (-)")
            ax[0].set_aspect("equal")
            points = np.array([xgrid,np.ones_like(xgrid) * depth_LES,zgrid]).reshape(3, -1).T
            velocity = self.get_velocity_grid(self.q_k, points, len(points))
            velocity_norm = np.linalg.norm(velocity, axis=1)
            velocity_norm = velocity_norm.reshape(len(xcoords), len(zcoords))
            c = ax[0].contourf(xgrid / (2 * self.r), zgrid / (2 * self.r), velocity_norm, cmap='Blues_r', levels=levels, extend="both")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            # LES results
            ax[1].set_title('Large Eddy Simulations')
            ax[1].set_ylabel("z/D (-)")
            ax[1].set_aspect("equal")
            c = ax[1].contourf(xcoords / (2 * self.r), zcoords / (2 * self.r), velocityLES, cmap='Blues_r', levels=levels, extend="both")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            # difference
            ax[2].set_title('Difference between FVW and LES')
            velocity_difference = abs(velocityLES.T-velocity_norm)
            c = ax[2].contourf(xgrid / (2 * self.r), zgrid / (2 * self.r), velocity_difference, cmap='Blues_r', levels=levels, extend="both")
            ax[2].set_xlabel("x/D (-)")
            ax[2].set_ylabel("z/D (-)")
            ax[2].set_aspect("equal")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(c, cax=cax)
            cb.set_label("v_dif (m/s)")
            cb.set_ticks(np.arange(0, self.x_inflow + 1, self.x_inflow / 2))

            plt.show()