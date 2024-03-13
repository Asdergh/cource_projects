#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt



class MechCalculator():


    def __init__(self, moment_of_inertia, thrust_koeff, bow_lenght, need_thrust, copter_mass) -> None:
        
        self.moment_of_inertia = moment_of_inertia
        self.thrust_koeff = thrust_koeff

        self.curent_angle = 0
        self.curent_anguler_velocity = 0
        self.curent_anguler_acceleration = 0

        self.moment_of_forces = 0
        self.bow_lenght = bow_lenght
        self.copter_mass = copter_mass

        self.cmd1_anguler_vel = 0
        self.cmd2_anguler_vel = 0
        self.cmd_anguler_acc = 0

        self.need_thrust = need_thrust
        self.virt_time = 0.001

        self.desired_angle = 0
    
    def calculate_mech(self):

        self.cmd1_anguler_vel = self.cmd_anguler_acc + self.need_thrust
        self.cmd2_anguler_vel = -self.cmd_anguler_acc + self.need_thrust

        self.moment_of_forces = self.thrust_koeff * (self.bow_lenght / 2) * (self.cmd1_anguler_vel ** 2 - self.cmd2_anguler_vel ** 2)
        self.curent_anguler_acceleration = self.moment_of_forces / self.moment_of_inertia
    
    def integrator(self):

        # print(f"\n Anguler position: [{self.curent_angle}]\n")
        # print(f"\n Anguler velocity: [{self.curent_anguler_velocity}]\n")
        # print(f"\n Anguler acceleration: [{self.curent_anguler_acceleration}]\n")
        # print(f"\n Moment of forces: [{self.moment_of_forces}]\n")

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
                 first_contour_saturation_sup, first_contour_saturation_inf,
                 second_contour_saturation_sup, second_contour_saturation_inf) -> None:
        
        self.first_contour_saturation_sup = first_contour_saturation_sup
        self.first_contour_saturation_inf = first_contour_saturation_inf
        self.second_contour_saturation_sup = second_contour_saturation_sup
        self.second_contour_saturation_inf = second_contour_saturation_inf

        self.first_contour_kp_pid = kp_angle
        self.first_contour_ki_pid = ki_angle
        self.first_contour_kd_pid = kd_angle

        self.second_contour_kp_pid = kp_ang_vel
        self.second_contour_ki_pid = ki_ang_vel
        self.second_contour_kd_pid = kd_ang_vel

        self.first_contour_contrall_sig = 0
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

        P_regulator = self.first_contour_kp_pid * self.first_contour_error
        I_regulator = self.first_contour_ki_pid * self.first_contour_error_integral
        D_regulator = self.first_contour_kd_pid * ((self.first_contour_error - self.first_contour_past_error) / self.mech_sistem.virt_time)

        self.first_contour_contrall_sig = P_regulator + I_regulator + D_regulator
        self.first_contour_past_error = self.first_contour_error
        self.first_contour_contrall_sig = self.first_contour_saturation(self.first_contour_contrall_sig)


    def PID_second_contour(self):

        self.second_contour_error = self.first_contour_contrall_sig - self.mech_sistem.curent_anguler_velocity
        self.second_contour_error_integral += self.second_contour_error * self.mech_sistem.virt_time

        P_regulator = self.second_contour_kp_pid * self.second_contour_error
        I_regulator = self.second_contour_ki_pid * self.second_contour_error_integral
        D_regulator = self.second_contour_kd_pid * ((self.second_contour_error - self.second_contour_past_error) / self.mech_sistem.virt_time)
        
        self.second_contour_contrall_sig = (P_regulator + I_regulator + D_regulator)
        self.second_contour_past_error = self.second_contour_error
        self.second_contour_contrall_sig = self.second_contour_saturation(self.second_contour_contrall_sig)

        self.mech_sistem.set_cmd_anguler_acc(self.second_contour_contrall_sig)


    def first_contour_saturation(self, contrall_signal):
 
        if contrall_signal > self.first_contour_saturation_inf:
            contrall_signal = self.first_contour_saturation_inf
        
        elif contrall_signal < -self.first_contour_saturation_sup:
            contrall_signal = -self.first_contour_saturation_sup
        
        return contrall_signal

    def second_contour_saturation(self, contrall_signal):

        if contrall_signal > self.second_contour_saturation_inf:
            contrall_signal = self.second_contour_saturation_inf
        
        elif contrall_signal < -self.second_contour_saturation_sup:
            contrall_signal = -self.second_contour_saturation_sup
        
        return contrall_signal
        
    

    
class Simulator():

    def __init__(self, simulation_max_step=20) -> None:
        
        self.stab_sistem = StabilisationSistem(ki_angle=41.0013, kd_angle=80.030, kp_angle=150.7095,
                                               ki_ang_vel=41.00096, kd_ang_vel=80.00034, kp_ang_vel=150.00020,
                                               first_contour_saturation_sup=50, first_contour_saturation_inf=300,
                                               second_contour_saturation_sup=5, second_contour_saturation_inf=15)
        
        self.anguler_position_list = []
        self.anguler_velocity_list = []
        self.anguler_acceleration_list = []
        self.error_distance = []
        self.first_contrall_sig_list = []
        self.second_contrall_sig_list = []

        self.simulation_step = 0
        self.simulation_max_step = simulation_max_step

    def run_simulation(self):

        self.time_interval = []
        while self.simulation_step < self.simulation_max_step:
            
            anguler_position = self.stab_sistem.mech_sistem.get_anguler_position()
            anguler_velocity = self.stab_sistem.mech_sistem.get_anguler_velocity()
            anguler_acceleration = self.stab_sistem.mech_sistem.get_anguler_acceleration()

            self.anguler_position_list.append(anguler_position)
            self.anguler_velocity_list.append(anguler_velocity)
            self.anguler_acceleration_list.append(anguler_acceleration)
            self.error_distance.append(self.stab_sistem.mech_sistem.desired_angle - anguler_position)
            self.first_contrall_sig_list.append(self.stab_sistem.first_contour_contrall_sig)
            self.second_contrall_sig_list.append(self.stab_sistem.second_contour_contrall_sig)
            self.time_interval.append(self.simulation_step)

            self.stab_sistem.mech_sistem.calculate_mech()
            self.stab_sistem.mech_sistem.integrator()
            self.stab_sistem.PID_first_contour()
            self.stab_sistem.PID_second_contour()
            

            self.simulation_step += self.stab_sistem.mech_sistem.virt_time
        
        self.anguler_position_data = np.asarray(self.anguler_position_list)
        self.anguler_velocity_data = np.asarray(self.anguler_velocity_list)
        self.anguler_acceleration_data = np.asarray(self.anguler_acceleration_list)
        self.error_distance = np.asarray(self.error_distance)
        self.desired_anguler_position = np.ones(self.anguler_position_data.shape) * self.stab_sistem.mech_sistem.desired_angle
        self.first_contrall_signal = np.asarray(self.first_contrall_sig_list)
        self.second_contrall_signal = np.asarray(self.second_contrall_sig_list)
        self.time_interval = np.asarray(self.time_interval)


        self.log_dict = {
            "anguler_acceleration": self.anguler_acceleration_data,
            "anguler_velocity": self.anguler_velocity_data,
            "anguler_position_data": self.anguler_position_data,
            "error_distance": self.error_distance,
            "desired_anguler_position": self.desired_anguler_position,
            "first_contrall_signal": self.first_contrall_signal,
            "second_contrall_signal": self.second_contrall_signal
        }

        data_frame = pd.DataFrame.from_dict(self.log_dict)
        print(data_frame)
        print(self.anguler_position_data)
        return (self.anguler_position_data, self.anguler_velocity_data, self.anguler_acceleration_data)

if __name__ == "__main__":

    sim = Simulator(simulation_max_step=200)
    sim.stab_sistem.mech_sistem.set_desired_angle(30.0)
    colors = ["red", "blue", "green"]
    labels = ["anguler_position", "anguler_velocity", "anguler_acceleration"]
    r, v, a = sim.run_simulation()
    samples = [r, v, a]

    plt.style.use("dark_background")
    fig, axis = plt.subplots(nrows=3)

    # axis[0].plot(sim.time_interval, sim.first_contrall_signal, color="blue", label="first signal")
    # axis[0].legend(loc="upper left")
    
    # axis[1].plot(sim.time_interval, sim.second_contrall_signal, color="y", label="second signal")
    # axis[1].legend(loc="upper left")

    axis[0].plot(sim.time_interval, sim.anguler_position_data, color="green", label="anguler position")
    axis[0].legend(loc="upper left")

    axis[1].plot(sim.time_interval, sim.anguler_velocity_data, color="red", label="anguler vel")
    axis[1].legend(loc="upper left")

    axis[2].plot(sim.time_interval, sim.anguler_acceleration_data, color="teal", label="anguler acceleration")
    axis[2].legend(loc="upper left")

    plt.show()
    

        
