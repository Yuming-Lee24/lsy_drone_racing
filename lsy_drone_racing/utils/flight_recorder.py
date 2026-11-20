"""Record the estimated drone state, the commands and the flight phase during deployment.

A background thread samples the `ROSConnector` estimate at a fixed rate, so the recording also
covers the parts of the flight that run outside the env's step loop (RL brake, return_to_start).
Analyze recordings with `scripts/analyze_flight.py`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from drone_estimators.ros import ROSConnector
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FlightRecorder:
    """Samples the estimated state of one drone in a background thread."""

    def __init__(
        self, connector: ROSConnector, drone_name: str, out_dir: str | Path, freq: float = 100.0
    ):
        """Create the recorder. Call `start` to begin sampling.

        Args:
            connector: ROS connector with the drone in its `estimator_names`.
            drone_name: Name of the drone in the connector.
            out_dir: Directory for the `.npz` recording.
            freq: Sampling frequency in Hz.
        """
        self.connector, self.drone_name, self.freq = connector, drone_name, freq
        self.out_dir = Path(out_dir)
        self.meta: dict[str, Any] = {}
        self.phase = "idle"
        self.cmd = np.full(4, np.nan)  # [roll, pitch, yaw] in rad, collective thrust in N
        self._rows: list[tuple] = []
        self._events: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start sampling."""
        if self._thread is not None:
            return
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True, name="flight_recorder")
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        """Label the following samples with `phase` and log a phase event."""
        with self._lock:
            self.phase = phase
            self.cmd = np.full(4, np.nan)
        self.event(f"phase:{phase}")

    def set_cmd(self, rpy: NDArray, thrust: float) -> None:
        """Store the latest low-level attitude command."""
        with self._lock:
            self.cmd = np.array([*np.asarray(rpy, dtype=np.float64)[:3], float(thrust)])

    def event(self, name: str) -> None:
        """Log a timestamped event."""
        self._events.append((time.perf_counter() - self._t0, name))
        logger.debug(f"Recorder event {name}")

    def _run(self) -> None:
        c, n = self.connector, self.drone_name
        t_next = time.perf_counter()
        while not self._stop.is_set():
            try:
                pos, quat = np.asarray(c.est_pos[n]), np.asarray(c.est_quat[n])
                vel, ang_vel = np.asarray(c.est_vel[n]), np.asarray(c.est_ang_vel[n])
                est_time = float(c.est_time[n])
            except Exception:  # Connector closed or not ready yet
                pos = quat = vel = ang_vel = None
            if pos is not None:
                with self._lock:
                    phase, cmd = self.phase, self.cmd.copy()
                t = time.perf_counter() - self._t0
                self._rows.append((t, est_time, pos, quat, vel, ang_vel, cmd, phase))
            t_next = max(t_next + 1.0 / self.freq, time.perf_counter())
            self._stop.wait(max(0.0, t_next - time.perf_counter()))

    def stop_and_save(self) -> Path | None:
        """Stop sampling and write the recording. Returns its path (None if nothing recorded)."""
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        if not self._rows:
            logger.warning("Flight recorder captured no samples")
            return None
        t, est_time, pos, quat, vel, ang_vel, cmd, phase = zip(*self._rows)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"flight_{datetime.now():%Y%m%d_%H%M%S}.npz"
        np.savez(
            path,
            t=np.array(t),
            est_time=np.array(est_time),
            pos=np.array(pos),
            quat=np.array(quat),
            vel=np.array(vel),
            ang_vel=np.array(ang_vel),
            cmd=np.array(cmd),
            phase=np.array(phase),
            event_t=np.array([e[0] for e in self._events]),
            event_name=np.array([e[1] for e in self._events]),
            meta=json.dumps(self.meta, default=_to_json),
        )
        logger.info(f"Saved flight recording ({len(t)} samples) to {path}")
        return path


def _to_json(x: Any) -> Any:
    """JSON fallback for numpy values and config objects."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return str(x)
