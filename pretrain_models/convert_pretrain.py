import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        print(old_k, "->", k)
        newmodel[k] = v

    state = {"model": newmodel, "__author__": "MOCO"}

    torch.save(state, output)
