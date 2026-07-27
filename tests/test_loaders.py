"""Acceptance 6: a saved linear policy round-trips through the TorchScript (.pt),
ONNX, and SB3 loaders to matching predict outputs. Skipped where a backend lib
is absent."""
import numpy as np
import pytest
from sim2real_score import LinearPolicy, load_policy

W = np.array([[-5.0, -0.2]], dtype=np.float64)
B = np.array([0.3], dtype=np.float64)
OBS = np.array([[0.7, -0.4], [0.1, 0.2], [-0.9, 0.5]], dtype=np.float64)


def _expected():
    return LinearPolicy(weight=W, bias=B).predict(OBS)


def test_torchscript_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 1)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
                self.lin.bias.copy_(torch.tensor(B, dtype=torch.float32))

        def forward(self, x):
            return self.lin(x)

    path = tmp_path / "policy.pt"
    torch.jit.save(torch.jit.script(Net()), str(path))
    pol = load_policy(str(path), kind="torch", obs_dim=2, action_dim=1)
    out = pol.predict(OBS)
    assert np.allclose(out, _expected(), atol=1e-4)


def test_onnx_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 1)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
                self.lin.bias.copy_(torch.tensor(B, dtype=torch.float32))

        def forward(self, x):
            return self.lin(x)

    path = tmp_path / "policy.onnx"
    torch.onnx.export(Net(), torch.zeros(1, 2), str(path),
                      input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "n"}, "action": {0: "n"}})
    pol = load_policy(str(path), kind="onnx", obs_dim=2, action_dim=1)
    out = pol.predict(OBS)
    assert np.allclose(out, _expected(), atol=1e-4)


def test_sb3_roundtrip(tmp_path):
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", "Pendulum-v1", seed=0, n_steps=64, device="cpu")
    path = tmp_path / "sb3_model.zip"
    model.save(str(path))
    pol = load_policy(str(path), kind="sb3")
    obs = np.zeros((1, 3), dtype=np.float32)
    out = pol.predict(obs)
    ref, _ = model.predict(obs, deterministic=True)
    assert np.allclose(out, ref, atol=1e-4)
