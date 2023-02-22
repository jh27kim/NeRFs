import torch

class PositionalEncoder():
    def __init__(self, input_dim, freq_level, include_input):
        self._input_dim = input_dim
        self._freq_level = freq_level
        self._out_dim = 2*input_dim*freq_level
        self._include_input = include_input

        if include_input:
            self._out_dim += input_dim

        self._embed_fns = self.encoder_fn()

    def encoder_fn(self):
        embed_fn = []

        if self._include_input:
            embed_fn.append(lambda x: x)

        for freq in range(self._freq_level):
            embed_fn.append(lambda x : torch.sin((2**freq) * x))
            embed_fn.append(lambda x : torch.cos((2**freq) * x))

    def encode(self, x):
        return torch.cat([fn(x) for fn in self._embed_fns], -1)

    def get_out_dim(self):
        return self._out_dim

