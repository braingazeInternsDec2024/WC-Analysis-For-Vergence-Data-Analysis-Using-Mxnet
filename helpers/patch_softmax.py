import json

input_file = "weights/16and32-symbol.json"
output_file = "weights/16and32-patched-symbol.json"

with open(input_file, "r") as f:
    symbol = json.load(f)

count = 0
for node in symbol["nodes"]:
    if node["op"] == "SoftmaxActivation":
        print(f"Patching op: {node['name']} from SoftmaxActivation to softmax")
        node["op"] = "softmax"
        if "attrs" in node:
            if "mode" in node["attrs"]:
                print(f"Removing mode from: {node['name']}")
                del node["attrs"]["mode"]
            node["attrs"]["axis"] = "1"
        else:
            node["attrs"] = {"axis": "1"}
        count += 1

print(f"✅ Patched {count} softmax ops.")

with open(output_file, "w") as f:
    json.dump(symbol, f)
