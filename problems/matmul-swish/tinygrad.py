from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_matrix, weight_matrix, bias, scaling_factor, output, batch_size, in_features, out_features):
    z = input_matrix @ weight_matrix.transpose() + bias
    output.assign(scaling_factor * z * z.sigmoid())
    output.realize() 