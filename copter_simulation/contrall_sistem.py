#!/bin/python3

import numpy as np
import pandas as pd
import site as st
import sys 

sys.path.append("/mnt/c/Users/1/Desktop/course_home_works/copter_simulation/mechanic_sistem")
from mechanic_sistem import MechSistemCalculation


class ContrallSistem():

    def __init__(self, kp_pos_z, ki_pos_z, kd_pos_z,
                 kp_pos_y, ki_pos_y, kd_pos_y,
                 kp_pos_x, ki_pos_x, kd_pos_x,
                 kp_theta, ki_theta, kd_theta,
                 kp_gamma, ki_gamma, kd_gamma,
                 kp_psi, ki_psi, kd_psi,
                 kp_ang_vel_x, ki_ang_vel_x, kd_ang_vel_x,
                 kp_ang_vel_y, ki_ang_vel_y, kd_ang_vel_y,
                 kp_ang_vel_z, ki_ang_vel_z, kd_ang_vel_z, 
                 saturation_thred_contour_sup, saturation_thred_contour_inf, 
                 saturation_first_contour_sup=0, saturation_second_contour_sup=0, saturation_first_contour_inf=0, 
                 saturation_second_contour_inf=0):


        self.mechanic_sistem = MechSistemCalculation(copter_mass=0.0630, k_coeff=3.9865e-08, d_coeff=7.5e-7,
                                                     ray_lenght=0.17, moment_of_inertia_x=5.82857000000000e-05, moment_of_inertia_y=7.16914000000000e-05,
                                                     moment_of_inertia_z=0.001000000000000)
        self.thrust_signal = -1

        self.kp_pos_z = kp_pos_z
        self.ki_pos_z = ki_pos_z
        self.kd_pos_z = kd_pos_z
        
        self.kp_pos_y = kp_pos_y
        self.ki_pos_y = ki_pos_y
        self.kd_pos_y = kd_pos_y

        self.kp_pos_x = kp_pos_x
        self.ki_pos_x = ki_pos_x
        self.kd_pos_x = kd_pos_x

        self.kp_theta = kp_theta
        self.ki_theta = ki_theta
        self.kd_theta = kd_theta

        self.kp_gamma = kp_gamma
        self.ki_gamma = ki_gamma
        self.kd_gamma = kd_gamma

        self.kp_psi = kp_psi
        self.ki_psi = ki_psi
        self.kd_psi = kd_psi

        self.kp_ang_vel_x = kp_ang_vel_x
        self.ki_ang_vel_x = ki_ang_vel_x
        self.kd_ang_vel_x = kd_ang_vel_x

        self.kp_ang_vel_y = kp_ang_vel_y
        self.ki_ang_vel_y = ki_ang_vel_y
        self.kd_ang_vel_y = kd_ang_vel_y

        self.kp_ang_vel_z = kp_ang_vel_z
        self.ki_ang_vel_z = ki_ang_vel_z
        self.kd_ang_vel_z = kd_ang_vel_z

        self.position_z_error = 0
        self.position_z_error_past = 0
        self.position_z_integr_error = 0

        self.position_y_error = 0
        self.position_y_error_past = 0
        self.position_y_integr_error = 0

        self.position_x_error = 0
        self.position_x_error_past = 0
        self.position_x_integr_error = 0

        self.theta_error = 0
        self.theta_error_past = 0
        self.theta_integr_error = 0

        self.gamma_error = 0
        self.gamma_error_past = 0
        self.gamma_error_integr_error = 0

        self.psi_error = 0
        self.psi_error_past = 0
        self.psi_error_integr_error = 0

        self.ang_vel_x_error = 0
        self.ang_vel_x_error_past = 0
        self.ang_vel_x_integr_error = 0

        self.ang_vel_y_error = 0
        self.ang_vel_y_error_past = 0
        self.ang_vel_y_integr_error = 0
        
        self.ang_vel_z_error = 0
        self.ang_vel_z_error_past = 0
        self.ang_vel_z_error_integr_error = 0

        
        self.saturatoin_thred_contour_inf = saturation_thred_contour_inf
        self.saturatoin_thred_contour_sup = saturation_thred_contour_sup



        
        self.kp_pos_z = 0


    def _PID_position(self):

        



    

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
