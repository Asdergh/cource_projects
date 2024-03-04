import numpy as np
import matplotlib.pyplot as plt



class CopterMechSistem():

    def __init__(self, mass, k_b, rotors_count, 
                 acceleration, velocity, position) -> None:
        

        self.copter_mass = mass
        self.k_b = k_b
        self.rotor_count = rotors_count

        self.copter_acceleration = acceleration
        self.copter_velocity = velocity
        self.copter_position = position
        self.g_coeff = 9.81

        self.virt_time = 0.0001
    
    def calculate_general_mech(self, anguler_vel):

        anguler_vel_sum = sum([anguler_vel ** 2 for _ in range(4)])
        self.copter_acceleration = (self.k_b * anguler_vel_sum) / self.copter_mass - self.g_coeff
    
    def integrate(self):

        self.copter_velocity += self.copter_acceleration * self.virt_time
        self.copter_position += self.copter_velocity * self.virt_time
    
    def get_acceleration(self):

        return self.copter_acceleration
    
    def get_velocity(self):

        return self.copter_velocity
    
    def get_position(self):

        return self.copter_position



class ContrallSistem():


    def __init__(self, k_p, k_i, k_d,
                 contrall_limit) -> None:
        
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.contrall_limit = contrall_limit
        self.error = 0
        self.past_error = 0
        self.integral_error = 0

        self.mech_sistem = CopterMechSistem(mass=0.006, k_b=3.986e-08, rotors_count=4,
                                             position=0, velocity=0, acceleration=0)
    

    def set_desired_position(self, desired_position):

        self.desired_position = desired_position
    
    def PID(self):

        self.error = self.desired_position - self.mech_sistem.copter_position
        self.integral_error += self.error * self.mech_sistem.virt_time
        
        self.P_regulator = self.k_p * self.error
        self.I_regulator = self.k_i * self.integral_error
        self.D_regulator = self.k_d * ((self.error - self.past_error) / self.mech_sistem.virt_time)

        self.contrall_signal = (self.P_regulator + self.I_regulator + self.D_regulator)
        self.contrall_signal = self._saturation(self.contrall_signal)

        self.past_error = self.error

    
    def _saturation(self, contrall_signal):

        if contrall_signal > self.contrall_limit:

            contrall_signal = self.contrall_limit

        elif contrall_signal < -self.contrall_limit:

            contrall_signal = -self.contrall_limit
        
        return contrall_signal



class Simulator():

    def __init__(self, T_max) -> None:
        
        self.contrall_sistem = ContrallSistem(k_p=150, k_i=0, k_d=30, contrall_limit=1000)
        self.T_max = T_max

        self.acceleration_list = []
        self.velocity_list = []
        self.position_list = []
        self.time_list = []
    
    def run_simulation(self):

        time = 0
        while time < self.T_max:

            position = self.contrall_sistem.mech_sistem.get_position()
            velocity = self.contrall_sistem.mech_sistem.get_velocity()
            acceleration = self.contrall_sistem.mech_sistem.get_acceleration()

            self.position_list.append(position)
            self.velocity_list.append(velocity)
            self.acceleration_list.append(acceleration)
            self.time_list.append(time)

            self.contrall_sistem.PID()
            self.contrall_sistem.mech_sistem.calculate_general_mech(self.contrall_sistem.contrall_signal)
            self.contrall_sistem.mech_sistem.integrate()

            time += self.contrall_sistem.mech_sistem.virt_time

        self.time_tensor = np.asarray(self.time_list)
        self.position_tensor = np.asarray(self.position_list)
        self.velocity_tensor = np.asarray(self.velocity_list)
        self.acceleration_tensor = np.asarray(self.acceleration_list)

    def show_result(self):

        # fig, axis = plt.subplots(nrows=3)

        # self.general_data = [self.position_list, self.velocity_list, self.acceleration_list]
        # self.colors = ["red", "blue", "green"]
        # self.labels = ["position", "velocity", "acceleration"]

        # for sample in range(len(self.general_data)):

        #     axis[sample].plot(self.time_tensor, self.general_data[sample], color=self.colors[sample], label=self.labels[sample])
        #     axis[sample].legend(loc="upper left")
        
        # plt.show()

        f = plt.figure(constrained_layout=True)
        gs = f.add_gridspec(3, 5)
        
        axis_1 = f.add_subplot(gs[0, :-1])
        axis_1.plot(self.position_tensor)
        axis_1.grid()
        axis_1.set_title("position")

        axis_2 = f.add_subplot(gs[1, :-1])
        axis_2.plot(self.velocity_tensor)
        axis_2.grid()
        axis_2.set_title("velocity")

        axis_3 = f.add_subplot(gs[2, :-1])
        axis_3.plot(self.acceleration_tensor, "r")
        axis_3.grid()
        axis_3.set_title("acceleration")

        plt.show()

if __name__ == "__main__":

    sim = Simulator(T_max=10)
    sim.contrall_sistem.set_desired_position(10)
    sim.run_simulation()
    sim.show_result()




         
