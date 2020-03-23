from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

class StatisticsPlotter:
    """
    Plots statistics about the simulation.
    """

    def plot_infected_per_generation(self, infected_per_generation_vector, recovered_per_generation,
                                                           dead_per_generation, sick_per_generation):
        # plot parameters
        plt.title('injection to time')
        plt.xlabel('day', color='#1C2833')
        plt.ylabel('people', color='#1C2833')

        # data
        p1 = plt.plot(infected_per_generation_vector)
        p2 = plt.plot(recovered_per_generation)
        p3 = plt.plot(dead_per_generation)
        p4 = plt.plot(sick_per_generation)
        plt.legend((p1[0], p2[0], p3[0], p4[0]), ("infected", "recovered", "dead", "sick"))
        plt.show()

    def plot_log_with_linear_regression(self, infected_per_generation_vector, recovered_per_generation,
                                                           dead_per_generation):
        # fit only to the start of the model.
        # We will always start with exponent, then die down one way or another.
        diff_vector = np.diff(infected_per_generation_vector)
        x_start_search = np.argmax(diff_vector > 0)
        x_max = np.argmax(diff_vector[x_start_search:] <= 0) # fit only until curvature starts going down.
        
        # get log vector
        infected_log_vector = np.log(infected_per_generation_vector)
        recovered_log_vector = np.log(recovered_per_generation)
        dead_log_vector = np.log(dead_per_generation)
        
        # put into variables for the fitting
        orig_x = np.array(range(infected_log_vector.size))
        x = orig_x[:x_max].reshape((-1, 1)) # used for the regression
        y = infected_log_vector[:x_max]
        
        # fit to linear
        model = LinearRegression()
        model.fit(x, y)
        
        # get fitting constants, y = b0 + b1*X
        b_0 = model.intercept_
        b_1 = model.coef_[0]
        r_sqr = model.score(x, y) # R^2
        
        # plot the regression line
        regression_x = np.linspace(0,x_max,100)
        regression_y = b_1 * regression_x + b_0
        plt.plot(regression_x, regression_y, '-r',
            label="{:.2f} + {:.2f}*x; R^2 = {:.2f}".format(b_0, b_1, r_sqr))
        
        # plot the data itself
        p1 = plt.plot(orig_x, infected_log_vector)
        p2 = plt.plot(orig_x, recovered_log_vector),
        p3 = plt.plot(orig_x, dead_log_vector)

        
        # plot parameters
        plt.title('Log graph with regression')
        plt.xlabel('steps', color='#1C2833')
        plt.ylabel('log(people)', color='#1C2833')
        plt.legend(loc='upper left')

        plt.grid()
        
        plt.show()
        


class StatePlotter:
    """
    Plots a current state of the simulation.
    """
    pass

