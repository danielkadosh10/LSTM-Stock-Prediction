from gate_stage import gate_stage
import activation_funcs as act_func


class forget_gate:
    def __init__(self, w_input, w_mem, b_forget):
        self.perc_long_term = gate_stage(w_input, w_mem, b_forget)

    def perform_calculation(self, input, short_mem):
        pre_sig_val, perc_long_mem = self.perc_long_term.perform_calculation(input, short_mem,
                                                                             act_func.func_type.SIGMOID)

        return pre_sig_val, perc_long_mem

    def update_weights(self, bias, i_weight, s_weight):
        self.perc_long_term.update_weights(bias, i_weight, s_weight)


class input_gate:
    def __init__(self, w_input, w_mem, b_input):
        self.perc_potential_mem = gate_stage(w_input, w_mem, b_input)

    def perform_calculation(self, input, short_term):
        pre_sig_val, perc_potential_mem = self.perc_potential_mem.perform_calculation(input, short_term,
                                                                                      act_func.func_type.SIGMOID)

        return pre_sig_val, perc_potential_mem

    def update_weights(self, bias, i_weight, s_weight):
        self.perc_potential_mem.update_weights(bias, i_weight, s_weight)


class cell_gate:
    def __init__(self, w_input, w_mem, b_cell):
        self.potential_long_mem = gate_stage(w_input, w_mem, b_cell)

    def perform_calculation(self, input, short_term):
        pre_tanh_val, potential_long_mem = self.potential_long_mem.perform_calculation(input, short_term,
                                                                                       act_func.func_type.TANH)

        return pre_tanh_val, potential_long_mem

    def update_weights(self, bias, i_weight, s_weight):
        self.potential_long_mem.update_weights(bias, i_weight, s_weight)


class output_gate:
    def __init__(self, w_input, w_mem, b_output):
        self.short_term_mem = gate_stage(w_input, w_mem, b_output)

    def perform_calculation(self, input, short_term, long_term):
        pre_tanh_val, short_term_mem = self.short_term_mem.perform_calculation(input, short_term,
                                                                               act_func.func_type.SIGMOID)
        normalize_long_term = act_func.tanh(long_term)
        potential_short_term = short_term_mem * normalize_long_term

        return pre_tanh_val, short_term_mem, potential_short_term

    def update_weights(self, bias, i_weight, s_weight):
        self.short_term_mem.update_weights(bias, i_weight, s_weight)
