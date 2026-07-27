"""Ground-truth checks, independent of the sensitivity machinery, so acceptance
tests validate the tool rather than a circular fixture."""

BOUNDARY_MARGIN = 0.12
SCORE_HIGH = 90.0
SCORE_LOW = 40.0


def brackets_boundary(evaluate_point, policy, env_id, space, param, boundary, side,
                      margin=BOUNDARY_MARGIN, seed=0):
    """A located breaking `boundary` for `param` is real iff stepping `margin`
    toward the failing side drops success below 0.5 while stepping toward the safe
    side keeps it above 0.5. `side`='low' fails below the boundary; 'high' fails
    above it."""
    if side == "low":
        fail_at, safe_at = boundary - margin, boundary + margin
    else:
        fail_at, safe_at = boundary + margin, boundary - margin
    fail = evaluate_point(policy, env_id, space, {param: fail_at}, seed=seed)
    safe = evaluate_point(policy, env_id, space, {param: safe_at}, seed=seed)
    return fail.success_rate < 0.5 < safe.success_rate
