import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_iris
plt.style.use("dark_background")



class NormalDistribution():


    def __init__(self, var=None, math_mean=None) -> None:

        self.var = var
        self.math_mean = math_mean

        self.case_nn = False
        self.case_ln = False
        self.case_np = False
        self.case_lp = False
    
    def calculate_distribution_params(self, data):

        # cheaking if var is None (None exist)
        if self.var is None:
            
            #cheaking lenght of data matrix
            if len(data.shape) == 1:

                self.var = np.asarray([[np.mean(np.dot(data[feature_i] - np.mean(data), data[feature_j] - np.mean(data))) 
                                        for feature_i in range(data.shape[0])] 
                                        for feature_j in range(data.shape[0])])
            
            elif len(data.shape) == 2:

                self.var = np.eye(data.shape[0])
                for  feature_vector in ((data.T)):

                    curent_vector_var = np.asarray([[np.mean(np.dot(feature_vector[feature_i] - np.mean(feature_vector), 
                                                                    feature_vector[feature_j] - np.mean(feature_vector)))
                                                                    for feature_i in range(feature_vector.shape[0])]
                                                                    for feature_j in range(feature_vector.shape[0])])
                    self.var = np.dot(self.var, curent_vector_var)
                
                print(self.var.shape)
                #self.var = self.var.T

            else:

                raise ValueError("must be array with size: (n, n) or less, not (n, n, n)")
        
        else:
            print("[var is already exist] !!")
        
        plt.imshow(self.var, cmap="magma")
        if self.math_mean is None:

            if len(data.shape) == 1:

                self.math_mean = np.asarray([data[feature_i] * np.mean(data) 
                                             for feature_i in range(data.shape[0])])
            
            else:

                self.math_mean = np.asarray([[data[feature_i, feature_j] * np.mean(data[feature_i])
                                             for feature_i in range(data.shape[0])]
                                             for feature_j in range(data.shape[1])])
                self.math_mean = self.math_mean.T
                
    
    def calculate_distribution(self, data, true_labels=None, param=None):

        self.data = data
        self.true_labels = true_labels
        self.param = param

        if (true_labels is None) and (param is None):

            self.case_nn = True
            self.distribution = np.exp(-(1/2) * 
                                       (((data - self.math_mean).T)
                                       .dot((self.var).T))
                                       .dot(data - self.math_mean))
        
        elif (true_labels is None) and (param is not None):

            self.case_np = True
            self.distribution = np.exp(-(1/2) * 
                                       (((data - np.dot(data, param)).T)
                                       .dot((self.var).T))
                                       .dot(data - np.dot(data, param)))
        
        elif (true_labels is not None) and (param is None):

            self.case_ln = True
            self.distribution = np.exp(-(1/2) * 
                                       (((true_labels - self.math_mean).T)
                                       .dot((self.var).T))
                                       .dot(true_labels - self.math_mean))
        
        elif (true_labels is not None) and (param is not None):

            print("RIGHT")
            self.case_lp = True
            self.distribution = np.exp(-(1/2) * 
                                       (((true_labels - np.dot(data, param)).T)
                                       .dot((self.var).T))
                                       .dot(true_labels - np.dot(data, param)))
            print(self.distribution.shape)
            print(param.T)
            print((true_labels - np.dot(data, param.T)).shape)
        
        return self.distribution

    def show_data(self):

        self.figure = plt.figure()
        self.axis_2d = self.figure.add_subplot(2, 1, 1)
        self.axis_3d = self.figure.add_subplot(2, 1, 2, projection="3d")

        def demo(simulation_step):
            
            self.axis_2d.clear()
            self.axis_3d.clear()

            if self.case_lp:


                if (len(self.data.shape) == 2) and (len(self.distribution.shape) == 0):

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    sub_axis_x, sub_axis_y = np.meshgrid(sub_axis_x, sub_axis_y)

                    distribution = np.exp(-(1/2) * 
                                       (((self.true_labels - np.dot(self.data, self.param)).T)
                                       .dot((self.var).T))
                                       .dot(self.true_labels - np.dot(self.data, self.param)))
                    distribution = np.exp(np.radians(np.random.normal(distribution, 12.34, size=sub_axis_x.shape)))
                    
                    self.axis_3d.plot_surface(sub_axis_x, sub_axis_y, distribution, cmap="coolwarm", alpha=0.45)
                    self.axis_2d.contourf(sub_axis_x, sub_axis_y, distribution, cmap="magma")
                
                elif len(self.data.shape) == 1:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    distribution = np.exp(-(1/2) * 
                                       (((self.true_labels * simulation_step - np.dot(self.data, self.param) * simulation_step).T)
                                       .dot((self.var).T))
                                       .dot(self.true_labels * simulation_step - np.dot(self.data, self.param) * simulation_step))
                    
                    self.axis_3d.plot(sub_axis_x, sub_axis_y, distribution, color="red", alpha=0.45, linestyle="--")


                    
                elif len(self.data.shape) == 2:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    sub_axis_x, sub_axis_y = np.meshgrid(sub_axis_x, sub_axis_y)

                    distribution = np.exp(-(1/2) * 
                                       (((self.true_labels * simulation_step - np.dot(self.data, self.param) * simulation_step).T)
                                       .dot((self.var).T))
                                       .dot(self.true_labels * simulation_step - np.dot(self.data, self.param) * simulation_step))

                    self.axis_3d.plot_surface(sub_axis_x, sub_axis_y, distribution, cmap="coolwarm", alpha=0.45)
                    self.axis_2d.contourf(sub_axis_x, sub_axis_y, distribution, cmap="magma")

                else:

                    raise ValueError("must be array with size: (n, n) or less, not (n, n, n)")
                
            if self.case_ln:

                if len(self.data.shape) == 1:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    distribution = np.exp(-(1/2) * 
                                       (((self.true_labels - self.math_mean).T)
                                       .dot((self.var).T))
                                       .dot(self.true_labels - self.math_mean))
                    
                    self.axis_3d.plot(sub_axis_x, sub_axis_y, distribution, color="red", alpha=0.45, linestyle="--")

                elif len(self.data.shape) == 2:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    sub_axis_x, sub_axis_y = np.meshgrid(sub_axis_x, sub_axis_y)

                    distribution = np.exp(-(1/2) * 
                                       (((self.true_labels - self.math_mean).T)
                                       .dot((self.var).T))
                                       .dot(self.true_labels - self.math_mean))

                    self.axis_3d.plot_surface(sub_axis_x, sub_axis_y, distribution, cmap="coolwarm", alpha=0.45)
                    self.axis_2d.contourf(sub_axis_x, sub_axis_y, distribution, cmap="magma")

                else:

                    raise ValueError("must be array with size: (n, n) or less, not (n, n, n)")
                
            if self.case_np:

                if len(self.data.shape) == 1:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    distribution = np.exp(-(1/2) * 
                                       (((self.data - np.dot(self.data, self.param)).T)
                                       .dot((self.var).T))
                                       .dot(self.data - np.dot(self.data, self.param)))
                    
                    self.axis_3d.plot(sub_axis_x, sub_axis_y, distribution, color="red", alpha=0.45, linestyle="--")

                elif len(self.data.shape) == 2:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    sub_axis_x, sub_axis_y = np.meshgrid(sub_axis_x, sub_axis_y)

                    distribution = np.exp(-(1/2) * 
                                       (((self.data - np.dot(self.data, self.param)).T)
                                       .dot((self.var).T))
                                       .dot(self.data - np.dot(self.data, self.param)))

                    self.axis_3d.plot_surface(sub_axis_x, sub_axis_y, distribution, cmap="coolwarm", alpha=0.45)
                    self.axis_2d.contourf(sub_axis_x, sub_axis_y, distribution, cmap="magma")

                else:

                    raise ValueError("must be array with size: (n, n) or less, not (n, n, n)")
                
            if self.case_nn:

                if len(self.data.shape) == 1:

                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[0])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[0])

                    distribution = np.exp(-(1/2) * 
                                       (((self.data - self.math_mean).T)
                                       .dot((self.var).T))
                                       .dot(self.data - self.math_mean))
                    
                    self.axis_3d.plot(sub_axis_x, sub_axis_y, distribution, color="red", alpha=0.45, linestyle="--")


                elif (len(self.data.shape) == 2):
                    
                    sub_axis_x = np.linspace(0, np.max(self.data), self.data.shape[1])
                    sub_axis_y = np.linspace(0, np.max(self.data), self.data.shape[1])

                    sub_axis_x, sub_axis_y = np.meshgrid(sub_axis_x, sub_axis_y)

                    distribution = np.exp(-(1/2) * 
                                       (((self.data - self.math_mean).T)
                                       .dot((self.var).T))
                                       .dot(self.data - self.math_mean))
                    
                    self.axis_3d.plot_surface(sub_axis_x, sub_axis_y, distribution, cmap="coolwarm", alpha=0.45)
                    self.axis_2d.contourf(sub_axis_x, sub_axis_y, distribution, cmap="magma")

                else:

                    raise ValueError("must be array with size: (n, n) or less, not (n, n, n)")

        
        animation_function = FuncAnimation(self.figure, demo, interval=10)
        plt.show()

    def calculate_optimal_param(self, data, true_label):

        self.optimal_param = ((((data.T).dot(self.var)).dot(data)).dot((1/2) 
                            * np.dot(data, self.var) - (1/2) * np.dot(true_label, self.var)))
        
# test part
if __name__ == "__main__":

    data = load_iris()
    train_data = data["data"][: data["data"].shape[0] // 2]
    train_labels = data["target"][: data["target"].shape[0] // 2]
    params = np.zeros(shape=train_data[0].shape)

    apo_dist = NormalDistribution()
    print(np.random.normal(0, 23.45) * np.eye(75))
    data_dist = NormalDistribution(var=(np.random.normal(0, 23.45) * np.eye(75)))

    apo_dist.calculate_distribution_params(train_data)
    apo_dist_result = apo_dist.calculate_distribution(data=train_data)
    data_dist_result = data_dist.calculate_distribution(data=train_data,
                                                        true_labels=train_labels,
                                                        param=params)
    
    print(data_dist_result, ": [data dist]")
    print(apo_dist_result, ": [apo dist ]")

    #apo_dist.show_data()
    data_dist.show_data()










            




        
