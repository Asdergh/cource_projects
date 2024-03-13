#!/bin/python3

from contrall_sistem import ContrallSistem
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(
    "/mnt/c/Users/1/Desktop/course_home_works/copter_simulation/contrall_sistem.py")


class DataHandler():

    def __init__(self, list_of_desired_positions, steps_per_pos):

        self.points_list = list_of_desired_positions
        self.steps_per_pos = steps_per_pos

        self.position_list = []
        self.orientation_list = []
        self.anguler_vel_list = []
        self.anguler_acc_list = []

        self.contrall_sistem = ContrallSistem(kp_position=np.random.normal(0, 123), ki_position=np.random.normal(0, 123), kd_position=np.random.normal(0, 123),
                                              kp_orientation=np.random.normal(0, 123), ki_orientation=np.random.normal(0, 123), kd_orientation=np.random.normal(0, 123),
                                              kp_ang_vel=np.random.normal(0, 123), ki_ang_vel=np.random.normal(0, 123), kd_ang_vel=np.random.normal(0, 123),
                                              saturation_thred_contour_inf=2631, saturation_thred_contour_sup=0)

    def run_simulation(self):

        with open("log_file.txt", "w") as file:

            for (point_number, point) in enumerate(self.points_list):

                simulation_step = 0
                log_string = f"number : [desired_point number: {point_number}], with cores: [x: {point[0]}, y: {point[1]}, z: {point[2]}]"
                while simulation_step <= self.steps_per_pos:

                    log_string += f"\t|copter_position: [x: {self.contrall_sistem.mechanic_sistem.copter_position[0]}, y: {self.contrall_sistem.mechanic_sistem.copter_position[1]}, z: {self.contrall_sistem.mechanic_sistem.copter_position[2]}]\t|copter_anguler_vel: [x_p: {self.contrall_sistem.mechanic_sistem.copter_anguler_velocity[0]}, y_p: {self.contrall_sistem.mechanic_sistem.copter_anguler_velocity[1]}, z_p: {self.contrall_sistem.mechanic_sistem.copter_anguler_velocity[2]}]\t|copter_anguler_orientation: [theta: {self.contrall_sistem.mechanic_sistem.copter_anguler_orientation[0]}, gamma: {self.contrall_sistem.mechanic_sistem.copter_anguler_orientation[0]}, psi: {self.contrall_sistem.mechanic_sistem.copter_anguler_orientation[2]}]\n" 

                    self.position_list.append(
                        self.contrall_sistem.mechanic_sistem.copter_position)
                    self.orientation_list.append(
                        self.contrall_sistem.mechanic_sistem.copter_anguler_orientation)
                    self.anguler_vel_list.append(
                        self.contrall_sistem.mechanic_sistem.copter_anguler_velocity)
                    self.anguler_acc_list.append(
                        self.contrall_sistem.mechanic_sistem.copter_anguler_acceleration)

                    self.contrall_sistem.mechanic_sistem._set_desired_position(
                        point)
                    self.contrall_sistem._PID_first_contour()
                    self.contrall_sistem._PID_second_contour()
                    self.contrall_sistem._PID_thred_contour()

                    self.contrall_sistem.mechanic_sistem._calculate_general_mech(
                        self.contrall_sistem.cmd_rotors_anguler_vel)
                    self.contrall_sistem.mechanic_sistem._integrator()

                    simulation_step += self.contrall_sistem.mechanic_sistem.virt_time

            file.write(log_string)

    def simple_animation(self):

        self.position_tensor = np.asarray(self.position_list)
        fig = plt.figure()
        axis_3d = fig.add_subplot(projection="3d")

        def anim(simulation_step):

            axis_3d.clear()
            #axis_3d.scatter(self.points_list[:, 0], self.points_list[:, 1], self.points_list[:, 2], color="red", s=0.34)
            axis_3d.scatter(self.position_tensor[simulation_step, 0], self.position_tensor[simulation_step, 1],
                            self.position_tensor[simulation_step, 2], s=12.34, color="green")

        demo = FuncAnimation(fig, anim, interval=100)
        plt.show()


if __name__ == "__main__":

    desired_positions = np.random.normal(0, 123, (10, 3))

    sim_object = DataHandler(
        list_of_desired_positions=desired_positions, steps_per_pos=20)

    sim_object.run_simulation()
    sim_object.simple_animation()
