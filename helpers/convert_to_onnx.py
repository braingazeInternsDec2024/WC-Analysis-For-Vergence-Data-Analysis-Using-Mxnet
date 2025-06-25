# import mxnet as mx
# import torch

# def convert_mxnet_params_to_torch(loaded_params):
#     """
#     Converts loaded MXNet params (flat dict) to PyTorch state dict.
#     """
#     state_dict = {}

#     for key, mx_nd in loaded_params.items():
#         np_array = mx_nd.asnumpy()
#         torch_tensor = torch.from_numpy(np_array)
#         state_dict[key] = torch_tensor

#     return state_dict

# if __name__ == "__main__":
#     # Load MXNet params (flat dict)
#     loaded_params = mx.nd.load('weights/16and32-0000.params')

#     # Convert to PyTorch state dict
#     torch_state_dict = convert_mxnet_params_to_torch(loaded_params)

#     # Save PyTorch state dict
#     torch.save(torch_state_dict, 'converted_16and32.pt')

#     print("Converted MXNet params saved as converted_16and32.pt")

sym = "weights/16and32-patched-symbol.json"  
params = "weights/16and32-0000.params"       
input_shape = (1, 3, 224, 224)
onnx_file = "16and32.onnx"

from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
