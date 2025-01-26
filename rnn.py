import numpy as np

from lstm import LSTM


class rnn:

    # Purpose: creates an RNN with hidden layers equal to the number defined
    # Inputs:
    #   num_hidden_layers - number of hidden layers
    def __init__(self, num_hidden_layers, num_of_features, learning_rate):

        self.learning_rate = learning_rate

        self.hidden_layers = []

        for i in range(num_hidden_layers):
            self.hidden_layers.append(
                LSTM(i, num_of_features))

    # Purpose: This functions performs the prediction.
    # Inputs:
    #   inputs - an array of values for each day
    def perform_prediction(self, inputs, num_of_features):
        short_term_mem = 0
        long_term_mem = 0

        for i in range(len(inputs)):
            # This works as a many-to-one config for the RNN since we only care about the closing price.
            # Here you can choose which features you want
            long_term_mem, short_term_mem = self.hidden_layers[i].perform_calculation(inputs[i, :num_of_features],
                                                                                      short_term_mem,
                                                                                      long_term_mem)

        return long_term_mem, short_term_mem

    # Purpose: This function calculates the mean squared error
    # inputs:
    #   actual_value: this is the value we want
    #   predicted_value: this is the value the model predicted
    def calculate_cost(self, actual_value, predicted_value):
        y_diff = (actual_value - predicted_value)
        cost = np.power(2, y_diff)

        return cost

    # Purpose: Performs backpropagation through time to update weights and biases of the LSTM network based on
    # target values.
    # inputs:
    #       target_vals: A list of target values for each time step
    #       num_of_days: The number of time steps
    def back_propagation(self, target_vals, num_of_days):
        back_prop = 0
        internal_state = 0
        forget_next = 0

        # Iterate through hidden layers in reverse order to perform backpropagation
        for i in range(len(self.hidden_layers)):
            # Update weights and biases for each hidden layer
            back_prop, internal_state, forget_next = self.hidden_layers[
                (len(self.hidden_layers) - 1) - i].update_weights(target_vals[num_of_days - 1 - i], back_prop,
                                                                  internal_state, forget_next)

        # Calculate the stochastic gradient descent
        sum_weights = 0
        sum_mem = 0
        sum_b = 0

        # Compute the sum of weight gradients, memory gradients, and bias gradients for each hidden layer
        for i in range(len(self.hidden_layers)):
            sum_weights += np.outer(np.transpose(self.hidden_layers[i].partials), self.hidden_layers[i].input)

        for j in range(len(self.hidden_layers) - 1):
            sum_mem = np.outer(self.hidden_layers[j + 1].partials, self.hidden_layers[j].ht_curr)

        for j in range(len(self.hidden_layers)):
            sum_b += self.hidden_layers[j].partials

        # Update weights, memory, and bias using stochastic gradient descent for each hidden layer
        for k in range(len(self.hidden_layers)):
            new_weights = self.hidden_layers[k].stochastic_descent_inputs(self.learning_rate * sum_weights)
            new_mem = self.hidden_layers[k].stochastic_descent_mem(self.learning_rate * sum_mem)
            new_bias = self.hidden_layers[k].stochastic_descent_bias(self.learning_rate * sum_b)

            self.hidden_layers[k].update_gates(new_weights, new_mem, new_bias)
