"""Analyze flight recordings from `deploy.py --record`, e.g. boundary-rush recovery experiments.

Prints the metrics of each recording and plots them aligned at the moment the drone left the
safety limits (or at the end of the race phase if it never left them). Pass a run with and one
without `deploy.brake_rl` to compare both recoveries.

Usage:
    python scripts/analyze_flight.py                       # latest recording in logs/flights
    python scripts/analyze_flight.py logs/flights/a.npz logs/flights/b.npz --out plots/cmp.png
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation as R

LOG_DIR = Path(__file__).parents[1] / "logs" / "flights"
STOP_SPEED = 0.2  # m/s, horizontal speed counted as stopped
MAX_JUMP_SPEED = 15.0  # m/s, faster position jumps between samples mean invalid tracking
PHASE_COLORS = {
    "race": "0.6",
    "brake": "tab:red",
    "rth_break": "tab:blue",
    "rth_return": "tab:cyan",
    "rth_land": "tab:green",
}


def load(path: str | Path) -> dict:
    """Load a recording and derive speed, tilt, Euler angles and the exit index."""
    raw = np.load(path, allow_pickle=False)
    d = {k: raw[k] for k in raw.files if k != "meta"}
    for k in ("pos", "quat", "vel", "ang_vel"):
        d[k] = d[k].astype(np.float64)
    d["meta"] = json.loads(str(raw["meta"]))
    d["path"] = Path(path)
    # Cut the recording where the state estimate becomes invalid: NaNs, zero quaternions, or a
    # position jump faster than MAX_JUMP_SPEED (mocap lost or swapped the drone, estimator diverged)
    valid = np.isfinite(d["pos"]).all(-1) & (np.linalg.norm(d["quat"], axis=-1) > 0.5)
    jump = np.linalg.norm(np.diff(d["pos"], axis=0), axis=-1) / np.maximum(np.diff(d["t"]), 1e-3)
    valid[1:] &= ~(jump > MAX_JUMP_SPEED)
    d["tracking_lost_t"] = None
    if not valid.all():
        n = int(np.argmin(valid))
        d["tracking_lost_t"] = float(d["t"][n])
        d["tracking_lost_pos"] = d["pos"][n - 1].round(2).tolist() if n else None
        for k in ("t", "est_time", "pos", "quat", "vel", "ang_vel", "cmd", "phase"):
            d[k] = d[k][:n]
    d["speed"] = np.linalg.norm(d["vel"], axis=-1)
    d["speed_xy"] = np.linalg.norm(d["vel"][:, :2], axis=-1)
    rot = R.from_quat(d["quat"])
    d["rpy"] = rot.as_euler("xyz")
    d["tilt"] = np.degrees(np.arccos(np.clip(rot.as_matrix()[:, 2, 2], -1, 1)))
    low = np.asarray(d["meta"]["pos_limit_low"])
    high = np.asarray(d["meta"]["pos_limit_high"])
    d["outside"] = np.linalg.norm(
        np.maximum(0, low - d["pos"]) + np.maximum(0, d["pos"] - high), axis=-1
    )
    airborne = np.cumsum(d["pos"][:, 2] > 0.1) > 0
    exit_idx = np.flatnonzero(airborne & (d["outside"] > 0))
    race = np.flatnonzero(d["phase"] == "race")
    d["exit_idx"] = int(exit_idx[0]) if len(exit_idx) else int(race[-1]) if len(race) else 0
    d["left_box"] = bool(len(exit_idx))
    d["t_rel"] = d["t"] - d["t"][d["exit_idx"]]
    return d


def phase_span(d: dict, phase: str) -> tuple[int, int] | None:
    """First and last index of a phase."""
    idx = np.flatnonzero(d["phase"] == phase)
    return (int(idx[0]), int(idx[-1])) if len(idx) else None


def metrics(d: dict) -> dict:
    """Recovery metrics relative to the box exit."""
    e = d["exit_idx"]
    after = slice(e, None)
    m = {
        "file": d["path"].name,
        "brake_rl": d["meta"].get("brake_rl"),
        "left_box": d["left_box"],
        "exit_pos": d["pos"][e].round(2).tolist(),
        "exit_speed": float(d["speed"][e]),
        "peak_speed": float(d["speed"].max()),
        "peak_speed_after_exit": float(d["speed"][after].max()),
        "max_outside": float(d["outside"].max()),
        "z_min_after_exit": float(d["pos"][after, 2].min()),
        "z_max": float(d["pos"][:, 2].max()),
        "tilt_max_after_exit": float(d["tilt"][after].max()),
    }
    slow = np.flatnonzero(d["speed_xy"][after] < STOP_SPEED)
    m["t_stop"] = float(d["t_rel"][e + slow[0]]) if len(slow) else np.nan
    back = np.flatnonzero(d["outside"][after] == 0)
    m["t_back_inside"] = float(d["t_rel"][e + back[0]]) if len(back) else np.nan
    m["t_outside"] = float(np.sum(np.diff(d["t"])[d["outside"][1:] > 0]))
    if (b := phase_span(d, "brake")) is not None:
        i0, i1 = b
        m["brake_start_after_exit"] = float(d["t_rel"][i0])
        m["brake_duration"] = float(d["t"][i1] - d["t"][i0])
        m["speed_at_brake_start"] = float(d["speed"][i0])
        m["brake_end_speed"] = float(d["speed"][i1])
        m["brake_distance_xy"] = float(np.linalg.norm(d["pos"][i1, :2] - d["pos"][i0, :2]))
        m["brake_z_drop"] = float(d["pos"][i0, 2] - d["pos"][i0 : i1 + 1, 2].min())
        m["brake_outcome"] = next(
            (n for n in d["event_name"] if n in ("brake_stopped", "brake_handed_over")), "?"
        )
    if (r := phase_span(d, "rth_break")) is not None:
        m["rth_start_after_exit"] = float(d["t_rel"][r[0]])
        m["speed_at_rth_start"] = float(d["speed"][r[0]])
    if d["tracking_lost_t"] is not None:
        m["TRACKING_LOST_after_exit"] = float(d["tracking_lost_t"] - d["t"][e])
        m["last_valid_pos"] = d["tracking_lost_pos"]
        m["last_valid_speed"] = float(d["speed"][-1])
    start = np.asarray(d["meta"]["start_pos"])
    m["final_pos"] = d["pos"][-1].round(2).tolist()
    m["land_err_xy"] = float(np.linalg.norm(d["pos"][-1, :2] - start[:2]))
    dt = np.diff(d["est_time"])
    dt = dt[dt > 0]
    m["est_rate_hz"] = float(1 / np.median(dt)) if len(dt) else np.nan
    m["est_max_gap_ms"] = float(dt.max() * 1000) if len(dt) else np.nan
    return m


def plot(runs: list[dict], out: str) -> None:
    """Top view, speed, height, tilt (actual vs commanded) and distance outside the box."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2, figsize=(14, 13))
    meta = runs[0]["meta"]
    low, high = np.asarray(meta["pos_limit_low"]), np.asarray(meta["pos_limit_high"])
    a = ax[0, 0]
    a.add_patch(plt.Rectangle(low[:2], *(high[:2] - low[:2]), fill=False, ec="k", ls="--"))
    a.plot(*np.asarray(meta["start_pos"])[:2], "g*", ms=14, label="start")
    styles = ["-", "--", ":", "-."]
    for k, d in enumerate(runs):
        ls = styles[k % len(styles)]
        name = f"{d['path'].stem} ({'RL brake' if d['meta'].get('brake_rl') else 'no brake'})"
        for phase, c in PHASE_COLORS.items():
            mask = d["phase"] == phase
            if mask.any():
                a.plot(d["pos"][mask, 0], d["pos"][mask, 1], ls, c=c, lw=1.5)
        a.plot(*d["pos"][d["exit_idx"], :2], "kx", ms=10)
        t = d["t_rel"]
        ax[0, 1].plot(t, d["speed"], ls, label=name)
        ax[1, 0].plot(t, d["pos"][:, 2], ls, label=name)
        ax[1, 1].plot(t, d["outside"], ls, label=name)
        ax[2, 0].plot(t, d["tilt"], ls, label=f"{name} tilt")
        cmd_tilt = np.degrees(
            np.arccos(np.clip(np.cos(d["cmd"][:, 0]) * np.cos(d["cmd"][:, 1]), -1, 1))
        )
        ax[2, 0].plot(t, cmd_tilt, ls, alpha=0.6, label=f"{name} commanded")
        ax[2, 1].plot(t, d["cmd"][:, 3], ls, label=name)
        for phase, c in PHASE_COLORS.items():
            if (s := phase_span(d, phase)) is not None and k == 0:
                for a2 in ax.flat[1:]:
                    a2.axvspan(t[s[0]], t[s[1]], color=c, alpha=0.08, lw=0)
    for phase, c in PHASE_COLORS.items():
        a.plot([], [], c=c, label=phase)
    a.set_aspect("equal")
    a.set(xlabel="x [m]", ylabel="y [m]", title="Top view (x: box exit)")
    a.legend(fontsize=8)
    titles = [
        ("speed [m/s]", "Speed"),
        ("z [m]", "Height"),
        ("outside box [m]", "Distance outside the safety box"),
        ("tilt [deg]", "Tilt: actual vs commanded"),
        ("thrust cmd [N]", "Collective thrust command (NaN = high-level commander)"),
    ]
    for a2, (ylabel, title) in zip(ax.flat[1:], titles):
        a2.axvline(0, c="k", ls=":", lw=1)
        a2.set(xlabel="time since box exit [s] (shading: phases of the first run)", ylabel=ylabel)
        a2.set_title(title)
        a2.grid(alpha=0.3)
        a2.legend(fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Saved plot to {out}")


def main(*paths: str, out: str = "plots/flight_analysis.png") -> None:
    """Analyze one or more recordings (default: the latest one in logs/flights)."""
    if not paths:
        files = sorted(LOG_DIR.glob("flight_*.npz"))
        assert files, f"No recordings in {LOG_DIR}"
        paths = (str(files[-1]),)
    runs = [load(p) for p in paths]
    for d in runs:
        m = metrics(d)
        print(f"\n== {m.pop('file')}")
        t_exit = d["t"][d["exit_idx"]]
        events = ", ".join(f"{n}@{t - t_exit:+.2f}s" for t, n in zip(d["event_t"], d["event_name"]))
        for k, v in m.items():
            print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")
        print(f"  events (rel. to exit)    {events}")
    plot(runs, out)


if __name__ == "__main__":
    fire.Fire(main)
