import numpy as np
import matplotlib.pyplot as plt
from Distributions import NormalDistribution
from sklearn.datasets import load_iris



class BayesEstimationMethod():

    def __init__(self, iteration_steps, distributions={}, random_generation=False) -> None:
        

        self.distributions = distributions
        self.random_generation = random_generation
        self.iteration_steps = iteration_steps
    
    def set_distributions(self, input_samples, true_labels, params):

        if self.distributions["apo"]["type"] == "normal":

            self.apo_dist = NormalDistribution(input_samples=params,
                                               math_mean=self.distributions["apo"]["math_mean"],
                                               covar=self.distributions["apo"]["covar"])
            print(self.apo_dist.math_mean)
        if self.distributions["data"]["type"] == "normal":

            self.data_dist = NormalDistribution(input_samples=input_samples[0],
                                                param_sample=params,
                                                true_labels=true_labels[0],
                                                math_mean=self.distributions["data"]["math_mean"],
                                                covar=self.distributions["data"]["covar"])
        
        self.apo_dist.calculate_distribution()
        self.data_dist.calculate_distribution()
        self.apos_dist = self.data_dist * self.apo_dist

    def fit(self, input_samples, true_labels):

        if self.random_generation:
            self.weights_vector = np.random.normal(0, 1.23, size=input_samples.shape[1])
        
        else:
            self.weights_vector = np.zeros(shape=input_samples.shape[1])
        
        self.set_distributions(input_samples, true_labels, self.weights_vector)
        for iteration in range(self.iteration_steps):
            
            self.apos_dist.move_distribution(param=self.weights_vector)
            self.weights_vector = np.dot(((np.dot(input_samples[iteration].T, input_samples[iteration]) + 
                                                        (self.apo_dist.covar).T)).T, 
                                                        (np.dot(input_samples[iteration].T, true_labels[iteration]) + 
                                                         np.dot((self.apo_dist.covar).T, self.apo_dist.math_mean)))
    
    def prediction(self, test_input_samples, test_true_labels):

        self.prediction_result = self.apos_dist.calculate_distribution(test_input_samples=test_input_samples,
                                              test_true_labels=test_true_labels)
        
        return self.prediction_result


if __name__ == "__main__":

    data = load_iris()
    input_samples = data["data"]
    input_true_labels = data["target"]

    train_data = input_samples[: input_samples.shape[0] // 2]
    train_labels = input_true_labels[: input_true_labels.shape[0] // 2]
    test_data = input_samples[input_samples.shape[0] // 2: ]
    test_labels = input_samples[input_true_labels.shape[0] // 2: ]

    distributions = {
        "apo": 
        {
            "type": "normal",
            "math_mean": None,
            "covar": None
        },
        "data":
        {
            "type": "normal",
            "math_mean": None,
            "covar": None
        }
    }

    model = BayesEstimationMethod(distributions=distributions,
                                  random_generation=False,
                                  iteration_steps=75)
    
    model.fit(input_samples=train_data,
              true_labels=train_labels)
    
    prediction = model.prediction(test_input_samples=test_data,
                     test_true_labels=test_labels)

    print(prediction)
    
    

    

    
            

            






        
        



        