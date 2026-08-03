"""Command line entry point: sim2real-score run ..."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import dump_dr_config, run_analysis
from .envs.registry import default_space
from .policies.loader import load_policy
from .randomization.space import RandomizationSpace
from .report import build_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim2real-score",
        description="Find where a control policy breaks under domain shift.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse a policy and write a report")
    run.add_argument("--policy", required=True,
                     help="path to an SB3 .zip, TorchScript .pt, or .onnx policy")
    run.add_argument("--policy-kind", default="auto",
                     choices=["auto", "sb3", "torch", "onnx"])
    run.add_argument("--vecnormalize",
                     help="path to a saved SB3 VecNormalize .pkl; its observation "
                          "statistics are applied before every prediction "
                          "(required for policies trained under VecNormalize)")
    run.add_argument("--env", required=True,
                     help="env id (e.g. linear, Hopper-v5, Walker2d-v5, Reacher-v5)")
    run.add_argument("--config", help="randomization space YAML (defaults per env)")
    run.add_argument("--out", default="sim2real_report", help="output directory")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                     help="parallel rollout workers")
    run.add_argument("--serial", action="store_true",
                     help="force serial execution (identical results, easier to debug)")
    run.add_argument("--sobol-base", type=int, default=32,
                     help="Saltelli base sample count; powers of two are best")
    run.add_argument("--grid-res", type=int, default=3,
                     help="points per axis in the coarse grid")

    sub.add_parser("envs", help="list built-in env ids with shipped defaults")
    return p


def _cmd_run(args) -> int:
    space = (RandomizationSpace.from_yaml(args.config) if args.config
             else default_space(args.env))
    policy = load_policy(args.policy, kind=args.policy_kind,
                         vecnormalize=args.vecnormalize)

    result = run_analysis(policy, args.env, space, seed=args.seed, jobs=args.jobs,
                          serial=args.serial, sobol_base=args.sobol_base,
                          grid_res=args.grid_res)

    os.makedirs(args.out, exist_ok=True)
    html = build_report(result, args.out)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        f.write(result.to_json())
    dr = dump_dr_config(result, os.path.join(args.out, "dr_config.yaml"))

    ranking = result.sensitivity.ranking()
    # A degenerate analysis has no variance to attribute, so the ranking is an
    # artefact of tie-breaking. Say so rather than printing an ordering.
    informative = bool(ranking) and not result.sensitivity.degenerate
    print(f"robustness score : {result.score:.1f}/100")
    print("most sensitive   : " + (", ".join(ranking) if informative
                                   else "n/a (no variance to attribute)"))
    for name in ranking:
        bp = result.breaking_points.get(name)
        if bp is None:
            continue
        edges = []
        if bp.low is not None:
            edges.append(f"below {bp.low:.3g}")
        if bp.high is not None:
            edges.append(f"above {bp.high:.3g}")
        print(f"  {name:<15} {' / '.join(edges) if edges else 'holds across range'}")
    print(f"report           : {html}")
    print(f"suggested DR     : {dr}")
    return 0


def _cmd_envs() -> int:
    for env_id in ["linear", "Reacher-v5", "Hopper-v5", "Walker2d-v5"]:
        space = default_space(env_id)
        print(f"{env_id:<14} params: {', '.join(sorted(space.params))}")
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "envs":
        return _cmd_envs()
    return 1


if __name__ == "__main__":
    sys.exit(main())
