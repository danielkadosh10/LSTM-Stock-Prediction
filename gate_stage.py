import activation_funcs as act_func
import numpy as np


class gate_stage:
    def __init__(self, w_input, w_mem, b_input):
        self.bias = b_input
        self.short_term_w = w_mem
        self.input_w = w_input

    def perform_calculation(self, input, short_mem, activation_func):
        percent_to_remember = 0

        # Aggregated input to the LSTM gate
        num_to_remember = self.bias + np.dot(input, self.input_w) + np.dot(self.short_term_w, short_mem)

        match activation_func:
            case act_func.func_type.SIGMOID:
                percent_to_remember = act_func.sigmoid(num_to_remember)

            case act_func.func_type.TANH:
                percent_to_remember = act_func.tanh(num_to_remember)

            case _:
                print("Not a supported function for the processing stage")

        return num_to_remember, percent_to_remember

    def init_weights(self, short_term_w, input_w, bias):
        self.short_term_w = short_term_w
        self.input_w = input_w
        self.bias = bias

    def update_weights(self, bias, i_weight, s_weight):
        self.bias = bias
        self.input_w = i_weight
        self.short_term_w = s_weight
