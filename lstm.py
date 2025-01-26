from gates import forget_gate
from gates import input_gate
from gates import cell_gate
from gates import output_gate
import numpy as np
import random
from activation_funcs import sigmoid
from activation_funcs import tanh


class LSTM:
    # Purpose: Creates an LSTM with the weights defined by the RNN.
    # Inputs:
    #   - label: number in the chain of hidden layers
    #   - w_forget/input/output: These are all the weights defined by the rnn
    #   - b_forget/input/output: These are all the biases defined by the rnn

    def __init__(self, label, num_of_features):
        self.label = label
        self.partials = 0

        # Weights and biases
        self.w_forget_i = np.random.uniform(-2, 2, (num_of_features, 1))
        self.w_input_i = np.random.uniform(-2, 2, (num_of_features, 1))
        self.w_output_i = np.random.uniform(-2, 2, (num_of_features, 1))
        self.w_mem_cell_i = np.random.uniform(-2, 2, (num_of_features, 1))

        self.w_forget_m = random.uniform(-2, 2)
        self.w_input_m = random.uniform(-2, 2)
        self.w_output_m = random.uniform(-2, 2)
        self.w_mem_cell_m = random.uniform(-2, 2)

        self.b_forget = random.uniform(-2, 2)
        self.b_input = random.uniform(-2, 2)
        self.b_output = random.uniform(-2, 2)
        self.b_mem_cell = random.uniform(-2, 2)

        # Values that each gate outputs
        self.input = np.zeros([num_of_features])
        self.out = 0
        self.zout = 0
        self.i = 0
        self.f = 0
        self.zf = 0
        self.zi = 0
        self.zo = 0
        self.g = 0
        self.zg = 0
        self.ht_prev = 0
        self.ht_curr = 0
        self.ct_prev = 0
        self.ct_curr = 0

        self.forget_gate = forget_gate(self.w_forget_i, self.w_forget_m, self.b_forget)
        self.input_gate = input_gate(self.w_input_i, self.w_input_m, self.b_input)
        self.cell_gate = cell_gate(self.w_mem_cell_i, self.w_mem_cell_m, self.b_mem_cell)
        self.output_gate = output_gate(self.w_output_i, self.w_output_m, self.b_output)

    # Purpose: Performs the calculation done by the LSTM. forget_gate->input->output
    # Inputs:
    #   - input: value for that day
    #   - short_mem: short term memory passed down by other LSTMs
    #   - long_mem: long term memory passed down by other LSTMs

    def perform_calculation(self, input, short_mem, long_mem):
        self.input = input
        self.ct_prev = long_mem
        self.ht_prev = short_mem

        self.zf, self.f = self.forget_gate.perform_calculation(self.input, self.ht_prev)
        self.zi, self.i = self.input_gate.perform_calculation(self.input, self.ht_prev)
        self.zg, self.g = self.cell_gate.perform_calculation(self.input, self.ht_prev)

        self.ct_curr = (self.ct_prev * float(self.f)) + (float(self.i) * float(self.g))
        self.zo, self.out, self.ht_curr = self.output_gate.perform_calculation(self.input, self.ht_prev, self.ct_curr)

        # Returns the current cell state and hidden state
        return self.ct_curr, float(self.ht_curr)

    def update_weights(self, actual, back_prop, ct_next, forget_next):
        # Difference between the current hidden state and the actual output
        diff = self.ht_curr - actual + back_prop

        # Internal state of the LSTM cell
        internal_state = diff * self.out * (1 - np.power(tanh(self.ct_curr), 2)) + ct_next * forget_next

        # Partial Derivatives
        mem_partial = internal_state * self.i * (1 - np.power(self.g, 2))
        input_partial = internal_state * self.g * self.i * (1 - self.i)
        forget_partial = internal_state * self.ct_prev * self.f * (1 - self.f)
        output_partial = diff * tanh(self.ct_curr) * self.out * (1 - self.out)

        self.partials = np.array \
            ([[float(mem_partial), float(input_partial), float(forget_partial), float(output_partial)]])

        concatenated_mem = np.array([[self.w_mem_cell_m, self.w_input_m, self.w_forget_m, self.w_output_m]])
        diff_mem = np.dot(concatenated_mem, np.transpose(self.partials))

        # Returns the change in memory weights, the internal state, and the forget gate value
        return float(diff_mem), float(internal_state), float(self.f)

    def stochastic_descent_inputs(self, sum_weights):
        concatenated_input = np.concatenate \
            ([np.transpose(self.w_mem_cell_i), np.transpose(self.w_input_i), np.transpose(self.w_forget_i),
              np.transpose(self.w_output_i)], axis=0)
        new_weights = np.subtract(concatenated_input, sum_weights)

        return new_weights

    def stochastic_descent_mem(self, sum_mem):
        concatenated_mem = np.array([[self.w_mem_cell_m, self.w_input_m, self.w_forget_m, self.w_output_m]])
        new_mem = np.subtract(np.transpose(concatenated_mem), sum_mem)

        return new_mem

    def stochastic_descent_bias(self, sum_b):
        concatenated_mem = np.array([self.b_mem_cell, self.b_input, self.b_forget, self.b_output])
        new_bias = np.subtract(np.transpose(concatenated_mem), sum_b)

        return new_bias

    def update_gates(self, new_weights, new_mem, new_bias):
        for i in range(len(new_weights[0, :])):
            self.w_mem_cell_i[i, 0] = new_weights[0, i]
            self.w_input_i[i, 0] = new_weights[0, i]
            self.w_forget_i[i, 0] = new_weights[0, i]
            self.w_output_i[i, 0] = new_weights[0, i]
        pass

        self.w_mem_cell_m = new_mem[0, 0]
        self.w_input_m = new_mem[1, 0]
        self.w_forget_m = new_mem[2, 0]
        self.w_output_m = new_mem[3, 0]

        self.b_mem_cell = new_bias[0, 0]
        self.b_input = new_bias[0, 1]
        self.b_forget = new_bias[0, 2]
        self.b_output = new_bias[0, 3]

        self.cell_gate.update_weights(self.b_mem_cell, self.w_mem_cell_i, self.w_mem_cell_m)
        self.input_gate.update_weights(self.b_input, self.w_input_i, self.w_input_m)
        self.forget_gate.update_weights(self.b_forget, self.w_forget_i, self.w_forget_m)
        self.output_gate.update_weights(self.b_output, self.w_output_i, self.w_output_m)
