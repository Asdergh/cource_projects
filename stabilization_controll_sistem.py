import numpy as np
import matplotlib.pyplot as plt


class MechCalculator():


    def __init__(self, moment_of_inertia, thrust_koeff, bow_lenght, need_thrust, copter_mass) -> None:
        
        self.moment_of_inertia = moment_of_inertia
        self.thrust_koeff = thrust_koeff

        self.curent_angle = 0
        self.curent_anguler_velocity = 0
        self.curent_anguler_acceleration =0

        self.moment_of_forces = 0
        self.bow_lenght = bow_lenght
        self.copter_mass = copter_mass

        self.cmd1_anguler_vel = 0
        self.cmd2_anguler_vel = 0
        self.cmd_anguler_acc = 0

        self.need_thrust = need_thrust
        self.virt_time = 0.1

        self.desired_angle =0
    
    def calculate_mech(self):

        self.cmd1_anguler_vel = self.cmd_anguler_acc + self.need_thrust
        self.cmd2_anguler_vel = -self.cmd_anguler_acc + self.need_thrust

        self.moment_of_forces = self.thrust_koeff * (self.bow_lenght / 2) * (self.cmd1_anguler_vel ** 2 + self.cmd2_anguler_vel ** 2)
        self.curent_anguler_acceleration = self.moment_of_forces / self.moment_of_inertia
    
    def integrator(self):

        self.curent_anguler_velocity += self.curent_anguler_acceleration * self.virt_time
        self.curent_angle += self.curent_anguler_velocity * self.virt_time

    def set_cmd_anguler_acc(self, acc):

        self.cmd_anguler_acc = acc
    
    def set_desired_angle(self, angle):

        self.desired_angle = angle
    
    def get_anguler_position(self):

        return self.curent_angle

    def get_anguler_velocity(self):

        return self.curent_anguler_velocity

    def get_anguler_acceleration(self):

        return self.curent_anguler_acceleration
    

class StabilisationSistem():

    def __init__(self, ki_angle, kp_angle, kd_angle,
                 ki_ang_vel, kp_ang_vel, kd_ang_vel, 
                 saturation_level) -> None:
        
        self.saturation_level = saturation_level
        self.first_contour_kp_pid = kp_angle
        self.first_contour_ki_pid = ki_angle
        self.first_contour_kd_pid = kd_angle

        self.second_contour_kp_pid = kp_ang_vel
        self.second_contour_ki_pid = ki_ang_vel
        self.second_contour_kd_pid = kd_ang_vel

        self.fisrt_contour_contrall_sig = 0
        self.second_contour_contrall_sig = 0

        self.first_contour_error = 0
        self.second_contour_error = 0

        self.first_contour_past_error = 0
        self.second_contour_past_error = 0
        
        self.first_contour_error_integral = 0
        self.second_contour_error_integral = 0

        self.mech_sistem = MechCalculator(moment_of_inertia=7.16914 * (10 ** -5), bow_lenght=0.17, thrust_koeff=3.9865 * (10 ** -8),
                                          copter_mass=0.06, need_thrust=9.456)

    def PID_first_contour(self):

        self.first_contour_error = self.mech_sistem.desired_angle - self.mech_sistem.curent_angle
        self.first_contour_error_integral += self.first_contour_error * self.mech_sistem.virt_time
        self.first_contour_contrall_sig = self.first_contour_kp_pid * self.first_contour_error + self.first_contour_ki_pid * self.first_contour_error_integral 
        + self.first_contour_kd_pid * (self.first_contour_error - self.first_contour_past_error) / self.mech_sistem.virt_time
        self.first_contour_past_error = self.first_contour_error

        #self.first_contour_contrall_sig = self.saturation(self.first_contour_contrall_sig)


    def PID_second_contour(self):

        self.second_contour_error = self.first_contour_contrall_sig - self.mech_sistem.curent_anguler_velocity
        self.second_contour_error_integral += self.second_contour_error * self.mech_sistem.virt_time
        self.second_contour_contral_sig = self.second_contour_kp_pid * self.second_contour_error + self.second_contour_ki_pid * self.second_contour_error_integral 
        + self.second_contour_kd_pid * (self.second_contour_error - self.second_contour_past_error) / self.mech_sistem.virt_time
        self.second_contour_past_error = self.second_contour_error

        #self.second_contour_contrall_sig = self.saturation(self.second_contour_contrall_sig)
        self.mech_sistem.set_cmd_anguler_acc(self.second_contour_contrall_sig)


    def saturation(self, contrall_signal):
 
        if contrall_signal > self.saturation_level:
            contrall_signal = self.saturation_level
        
        elif contrall_signal < -self.saturation_level:
            contrall_signal = self.saturation_level
        
        return contrall_signal

        
    

    
class Simulator():

    def __init__(self, simulation_max_step=20) -> None:
        
        self.stab_sistem = StabilisationSistem(ki_angle=0.12, kd_angle=0.12, kp_angle=0.12,
                                               ki_ang_vel=0.12, kd_ang_vel=0.12, kp_ang_vel=0.12,
                                               saturation_level=1000)
        
        self.anguler_position_list = []
        self.anguler_velocity_list = []
        self.anguler_acceleration_list = []

        self.simulation_step = 0
        self.simulation_max_step = simulation_max_step

    def run_simulation(self):

        while self.simulation_step < self.simulation_max_step:
            
            anguler_position = self.stab_sistem.mech_sistem.get_anguler_position()
            anguler_velocity = self.stab_sistem.mech_sistem.get_anguler_velocity()
            anguler_acceleration = self.stab_sistem.mech_sistem.get_anguler_acceleration()

            self.anguler_position_list.append(anguler_position)
            self.anguler_velocity_list.append(anguler_velocity)
            self.anguler_acceleration_list.append(anguler_acceleration)

            self.stab_sistem.mech_sistem.calculate_mech()
            self.stab_sistem.mech_sistem.integrator()
            self.stab_sistem.PID_first_contour()
            self.stab_sistem.PID_second_contour()

            self.simulation_step += 1
        
        self.anguler_position_data = np.asarray(self.anguler_position_list)
        self.anguler_velocity_data = np.asarray(self.anguler_velocity_list)
        self.anguler_acceleration_data = np.asarray(self.anguler_acceleration_list)

        return (self.anguler_position_data, self.anguler_velocity_data, self.anguler_acceleration_data)

if __name__ == "__main__":

    sim = Simulator(simulation_max_step=3)
    colors = ["red", "blue", "green"]
    labels = ["anguler_position", "anguler_velocity", "agnuler_acceleration"]
    r, v, a = sim.run_simulation()
    samples = [r, v, a]

    fig, axis = plt.subplots(nrows=3)
    for sample in range(len(axis)):
        axis[sample].plot(range(1, len(samples[sample]) + 1), samples[sample], color=colors[sample], label=labels[sample])
        axis[sample].legend(loc="upper left")
    
    plt.show()
    

        
