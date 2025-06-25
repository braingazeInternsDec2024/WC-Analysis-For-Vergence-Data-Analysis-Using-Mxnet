import mxnet as mx
import json

def print_symbol_info(symbol_json_path):
    # Load symbol JSON file as dict to inspect raw contents
    with open(symbol_json_path, 'r') as f:
        symbol_json = json.load(f)
    print("=== Raw Symbol JSON keys ===")
    print(symbol_json.keys())
    print()

    # Load symbol from JSON file
    symbol = mx.sym.load(symbol_json_path)
    
    # Print all argument names (inputs + weights + biases)
    arg_names = symbol.list_arguments()
    print("=== Arguments (inputs + params) ===")
    for name in arg_names:
        print(f"  {name}")
    print(f"Total args: {len(arg_names)}\n")

    # Print auxiliary states (like running_mean/var for batchnorm)
    aux_names = symbol.list_auxiliary_states()
    print("=== Auxiliary states ===")
    for aux in aux_names:
        print(f"  {aux}")
    print()

    # Print outputs
    outputs = symbol.list_outputs()
    print("=== Outputs ===")
    for out in outputs:
        print(f"  {out}")
    print()

    # Inspect the internal JSON graph nodes (layers)
    nodes = symbol_json.get('nodes', [])
    print(f"=== Number of nodes in symbol graph: {len(nodes)} ===")
    # Print first 5 nodes as sample
    for i, node in enumerate(nodes[:5]):
        print(f"Node {i}:")
        print(f"  name: {node.get('name')}")
        print(f"  op: {node.get('op')}")
        print(f"  inputs: {node.get('inputs')}")
        print()

    # Print the input shapes to check
    # Example input shape - modify to your expected input
    input_shapes = {'data': (1, 3, 224, 224)}

    try:
        arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(**input_shapes)
        print("=== Inferred shapes ===")
        print("Arguments shapes:")
        for name, shape in zip(arg_names, arg_shapes):
            print(f"  {name}: {shape}")
        print()
        print("Outputs shapes:")
        for name, shape in zip(outputs, out_shapes):
            print(f"  {name}: {shape}")
        print()
        print("Auxiliary states shapes:")
        for name, shape in zip(aux_names, aux_shapes):
            print(f"  {name}: {shape}")
        print()
    except Exception as e:
        print(f"Error inferring shapes: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python print_mxnet_model_info.py <symbol_json_path>")
        sys.exit(1)
    symbol_json_path = sys.argv[1]
    print_symbol_info(symbol_json_path)
