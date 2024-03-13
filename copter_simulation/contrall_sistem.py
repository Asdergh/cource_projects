#!/bin/python3

import numpy as np
import pandas as pd
import site as st
import sys 

sys.path.append("/mnt/c/Users/1/Desktop/course_home_works/copter_simulation/mechanic_sistem")
from mechanic_sistem import MechSistemCalculation


class ContrallSistem():

    def __init__(self, kp_position, ki_position, kd_position,
                 kp_orientation, ki_orientation, kd_orientation,
                 kp_ang_vel, ki_ang_vel, kd_ang_vel, 
                 saturation_thred_contour_sup, saturation_thred_contour_inf, 
                 saturation_first_contour_sup=0, saturation_second_contour_sup=0, saturation_first_contour_inf=0, 
                 saturation_second_contour_inf=0):


        self.mechanic_sistem = MechSistemCalculation(copter_mass=0.0630, k_coeff=3.9865e-08, d_coeff=7.5e-7,
                                                     ray_lenght=0.17, moment_of_inertia_x=5.82857000000000e-05, moment_of_inertia_y=7.16914000000000e-05,
                                                     moment_of_inertia_z=0.001000000000000)
        self.thrust_signal = -1

        self.position_error = np.zeros(2)
        self.orientation_error = np.zeros(2)
        self.anguler_vel_error = np.zeros(3)

        self.position_past_error = np.zeros(2)
        self.orientation_past_error = np.zeros(2)
        self.anguler_vel_past_error = np.zeros(3)

        self.position_integr_error = np.zeros(2)
        self.orientation_integr_error = np.zeros(2)
        self.anguler_vel_integr_error = np.zeros(3)

        self.kp_position = kp_position
        self.ki_position = ki_position
        self.kd_position = kd_position

        self.kp_orientation = kp_orientation
        self.ki_orientation = ki_orientation
        self.kd_orientation = kd_orientation

        self.kp_ang_vel = kp_ang_vel
        self.ki_ang_vel = ki_ang_vel
        self.kd_ang_vel = kd_ang_vel

        self.first_contour_saturation_sup = saturation_first_contour_sup
        self.second_contour_saturation_sup = saturation_second_contour_sup
        self.thred_contour_saturation_sup = saturation_thred_contour_sup

        self.first_contour_saturation_inf = saturation_first_contour_inf
        self.second_contour_saturation_inf = saturation_second_contour_inf
        self.thred_contour_saturation_inf = saturation_thred_contour_inf

        self.cmd_rotors_anguler_vel = [0.0, 0.0, 0.0, 0.0]



    def _PID_first_contour(self):

        self.position_error = self.mechanic_sistem.desired_position[:-1] - self.mechanic_sistem.copter_position[:-1]
        self.thrust_signal = self.mechanic_sistem.desired_position[-1] - self.mechanic_sistem.copter_position[-1]
        self.position_integr_error += self.position_error[:-1] * self.mechanic_sistem.virt_time

        self.first_contour_P_reg = self.kp_position * self.position_error
        self.first_contour_I_reg = self.ki_position * self.position_integr_error
        self.first_contour_D_reg = self.kd_position * ((self.position_error - self.position_past_error) / self.mechanic_sistem.virt_time)

        self.first_contour_signal = (self.first_contour_P_reg + self.first_contour_I_reg + self.first_contour_D_reg)
        self.position_past_error = self.position_error

    def _PID_second_contour(self):

        self.orientation_error = self.first_contour_signal - self.mechanic_sistem.copter_anguler_orientation[:-1]
        self.orientation_integr_error = self.orientation_error * self.mechanic_sistem.virt_time

        self.second_contour_P_reg = self.kp_orientation * self.orientation_error
        self.second_contour_I_reg = self.ki_orientation * self.orientation_integr_error
        self.second_contour_D_reg = self.kd_orientation * ((self.orientation_error - self.orientation_past_error) / self.mechanic_sistem.virt_time)

        self.second_contour_signal = (self.second_contour_P_reg + self.second_contour_I_reg + self.second_contour_D_reg)
        self.orientation_past_error = self.orientation_error

    def _PID_thred_contour(self):

        self.anguler_vel_error[:-1] = self.second_contour_signal - self.mechanic_sistem.copter_anguler_velocity[:-1]
        self.anguler_vel_error[-1] = self.mechanic_sistem.copter_anguler_orientation[-1] - self.mechanic_sistem.copter_anguler_velocity[-1]
        self.anguler_vel_integr_error[:-
                                      1] = self.anguler_vel_error[:-1] * self.mechanic_sistem.virt_time
        self.anguler_vel_integr_error[-1] = self.anguler_vel_error[-1] * self.mechanic_sistem.virt_time

        self.thred_contour_P_reg = self.kp_ang_vel * self.anguler_vel_error
        self.thred_contour_I_reg = self.ki_ang_vel * self.anguler_vel_integr_error
        self.thred_contour_D_reg = self.kd_ang_vel * ((self.anguler_vel_error - self.anguler_vel_error) / self.mechanic_sistem.virt_time)

        self.thred_contour_signal = (self.thred_contour_P_reg + self.thred_contour_D_reg + self.thred_contour_I_reg)

        self.thred_contour_signal[0] = self._saturation(self.thred_contour_signal[0],
                self.thred_contour_saturation_sup,
                self.thred_contour_saturation_inf)
        self.thred_contour_signal[1] = self._saturation(self.thred_contour_signal[1],
                self.thred_contour_saturation_sup,
                self.thred_contour_saturation_inf)
        self.thred_contour_signal[2] = self._saturation(self.thred_contour_signal[2],
                self.thred_contour_saturation_sup,
                self.thred_contour_saturation_inf)
        
        self.anguler_vel_past_error = self.anguler_vel_error

    def _saturation(self, signal, sup, inf):

        if (signal > inf):

            signal = inf
        
        elif (signal < -sup):

            signal = -sup
        
        return signal

    def comand_mixer(self):

        self.cmd_rotors_anguler_vel[0] = self.thrust_signal + self.thred_contour_signal[0] + \
            self.thred_contour_signal[1] - self.thred_contour_signal[2]
        
        self.cmd_rotors_anguler_vel[1] = self.thrust_signal + self.thred_contour_signal[0] - \
            self.thred_contour_signal[1] + self.thred_contour_signal[2]
        
        self.cmd_rotors_anguler_vel[2] = self.thrust_signal - self.thred_contour_signal[0] - \
            self.thred_contour_signal[1] - self.thred_contour_signal[2]
        
        self.cmd_rotors_anguler_vel[3] = self.thrust_signal + self.thred_contour_signal[0] - \
            self.thred_contour_signal[1] - self.thred_contour_signal[2]
