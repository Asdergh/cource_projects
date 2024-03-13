#!/bin/python3


import numpy as np
import os



class MechSistemCalculation():

    def __init__(self, copter_mass, k_coeff, ray_lenght, d_coeff,
                 moment_of_inertia_x, moment_of_inertia_y, moment_of_inertia_z):

        self.copter_position = np.zeros(3)
        self.copter_velocity = np.zeros(3)
        self.copter_acceleration = np.zeros(3)
        self.copter_anguler_velocity = np.zeros(3)
        self.copter_anguler_acceleration = np.zeros(3)
        self.copter_anguler_orientation = np.zeros(3)

        self.normalization_vector = np.array([0.0, 0.0, 1])
        self.copter_mass = copter_mass
        self.thrust_coeff = k_coeff
        self.g_coeff = 9.81
        self.ray_lenght = ray_lenght
        self.d_coeff = d_coeff

        self.thrust_force = 0
        self.moment_of_inertia_x = moment_of_inertia_x
        self.moment_of_inertia_y = moment_of_inertia_y
        self.moment_of_inertia_z = moment_of_inertia_z

        self.virt_time = 0.001

    def _set_desired_position(self, desired_position):

        self.desired_position = desired_position

    def _calculate_cmd_state(self, desired_position, desired_velocity, desired_acceleration, curent_time):

        T_matrix = np.array([
            [0, 0, 0, 0, 0, 1],
            [curent_time ** 5, curent_time ** 4, curent_time **
                3, curent_time ** 2, curent_time, 1],
            [0, 0, 0, 0, 1, 0],
            [5 * curent_time ** 4, 4 * curent_time ** 3,
                3 * curent_time ** 2, 2 * curent_time, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20 * curent_time ** 3, 12 * curent_time ** 2, 6 * curent_time, 2, 0, 0],
        ])

        state_vector_x = np.array([self.copter_position[0], desired_position[0],
                                   self.copter_velocity[0], desired_velocity[0],
                                   self.copter_acceleration[0], desired_acceleration[0]])

        state_vector_y = np.array([self.copter_position[1], desired_position[1],
                                   self.copter_velocity[1], desired_velocity[1],
                                   self.copter_acceleration[1], desired_acceleration[1]])

        state_vector_z = np.array([self.copter_position[2], desired_position[2],
                                   self.copter_velocity[2], desired_position[2],
                                   self.copter_acceleration[2], desired_acceleration[2]])

        x_polinom_coeff = T_matrix.dot(state_vector_x)
        y_polinom_coeff = T_matrix.dot(state_vector_y)
        z_polinom_coeff = T_matrix.dot(state_vector_z)

        cmd_position_x = T_matrix[1, :].dot(x_polinom_coeff)
        cmd_position_y = T_matrix[1, :].dot(y_polinom_coeff)
        cmd_position_z = T_matrix[1, :].dot(z_polinom_coeff)

        self.cmd_position = np.array(
            [cmd_position_x, cmd_position_y, cmd_position_z])

    def _calculate_general_mech(self, cmd_anguler_vel):


        self.rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(self.copter_anguler_orientation[0]),
             np.sin(self.copter_anguler_orientation[0])],
            [0, -np.sin(self.copter_anguler_orientation[0]), np.cos(self.copter_anguler_orientation[0])]])

        self.rotation_matrix_y = np.array([
            [np.cos(self.copter_anguler_orientation[1]), 0,
             np.sin(self.copter_anguler_orientation[1])],
            [0, 1, 0],
            [-np.sin(self.copter_anguler_orientation[1]), 0, np.cos(self.copter_anguler_orientation[1])]])

        self.rotation_matrix_z = np.array([
            [np.cos(self.copter_anguler_orientation[2]), np.sin(
                self.copter_anguler_orientation[2]), 0],
            [-np.sin(self.copter_anguler_orientation[2]),
             np.cos(self.copter_anguler_orientation[2]), 0],
            [0, 0, 1]])

        self.rotation_matrix = (self.rotation_matrix_x.dot(
            self.rotation_matrix_y)).dot(self.rotation_matrix_z)

        self.trust_force = sum(
            [ang_vel ** 2 for ang_vel in cmd_anguler_vel]) * self.thrust_coeff

        self.copter_acceleration = ((1 / self.copter_mass) *
                                    np.dot(self.rotation_matrix, (self.normalization_vector * self.thrust_force)) + self.normalization_vector * self.g_coeff)

        self.thrust_moment = np.array([
            (self.thrust_coeff * self.ray_lenght *
            (cmd_anguler_vel[0] ** 2 - cmd_anguler_vel[2] ** 2)),
            (self.thrust_coeff * self.ray_lenght *
            (cmd_anguler_vel[3] ** 2 - cmd_anguler_vel[1] ** 2)),
            self.d_coeff * (cmd_anguler_vel[3] ** 2 + cmd_anguler_vel[1] ** 2 - cmd_anguler_vel[0] ** 2 - cmd_anguler_vel[2] ** 2)])

        self.tensor_of_inertia = np.array([
            [self.moment_of_inertia_x, 0, 0],
            [0, self.moment_of_inertia_y, 0],
            [0, 0, self.moment_of_inertia_z]
        ])

        self.copter_anguler_acceleration = ((np.linalg.inv(self.tensor_of_inertia))
                                            .dot((self.thrust_moment - np.cross(self.copter_anguler_velocity, 
                                                                                np.dot(self.tensor_of_inertia, self.copter_anguler_velocity)))))

    def _integrator(self):

        self.copter_velocity += self.copter_acceleration * self.virt_time
        self.copter_position += self.copter_velocity * self.virt_time
        self.copter_anguler_velocity += self.copter_anguler_acceleration * self.virt_time
        self.copter_anguler_orientation += self.copter_anguler_velocity * self.virt_time
