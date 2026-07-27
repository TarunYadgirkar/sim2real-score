"""Export example policies in each supported format.

Also documents the raw-PyTorch interface sim2real-score expects: a TorchScript
module mapping float32 [N, obs_dim] -> [N, action_dim], deterministic.

    python examples/export_example_policies.py --out /tmp/policies
"""
import argparse
import os

import torch

# Gains matching the ground-truth fixtures on the built-in `linear` env.
FRICTION_OVERFIT = [[-2.0, 1.1]]    # cancels nominal friction -> breaks when friction drops
LATENCY_FRAGILE = [[-10.0, -18.0]]  # strong derivative gain -> breaks under delay


class LinearPolicyNet(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.lin = torch.nn.Linear(len(weight[0]), len(weight), bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def export(weight, out_dir, stem):
    net = LinearPolicyNet(weight).eval()
    pt_path = os.path.join(out_dir, f"{stem}.pt")
    torch.jit.save(torch.jit.script(net), pt_path)

    onnx_path = os.path.join(out_dir, f"{stem}.onnx")
    torch.onnx.export(net, torch.zeros(1, len(weight[0])), onnx_path,
                      input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "n"}, "action": {0: "n"}})
    return pt_path, onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="example_policies")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for stem, w in [("friction_overfit", FRICTION_OVERFIT),
                    ("latency_fragile", LATENCY_FRAGILE)]:
        for path in export(w, args.out, stem):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
