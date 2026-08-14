"""The sole canonical first-party boundary for SO follower motor side effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .safety import SafetyError, communication_fault, require_raw_goal_ticks


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_MOTORS = ARM_JOINTS + ("gripper",)
PID_REGISTERS = {"p": "P_Coefficient", "i": "I_Coefficient", "d": "D_Coefficient"}


@dataclass(frozen=True)
class MotorState:
    timestamp: float
    positions: Mapping[str, float]
    normalized: bool


@dataclass(frozen=True)
class WriteAudit:
    """Write accounting separated by intent and confirmed bus calls."""

    goal_position_attempts: int
    pid_attempts: int
    torque_attempts: int
    other_register_attempts: int
    goal_position_physical_writes: int
    pid_physical_writes: int
    torque_physical_writes: int
    other_register_physical_writes: int


def _write_category(register: str) -> str:
    if register == "Goal_Position":
        return "goal_position"
    if register in PID_REGISTERS.values():
        return "pid"
    if register == "Torque_Enable":
        return "torque"
    return "other_register"


class _ReadOnlyBusGuard:
    """Narrow proxy that makes motor-register writes unreachable at runtime."""

    def __init__(self, raw_bus: Any, record_attempt: Callable[[str], None]):
        self.__raw_bus = raw_bus
        self.__record_attempt = record_attempt

    def connect(self, handshake: bool = False) -> None:
        # Read-only mode intentionally skips LeRobot's optional handshake. It
        # opens the serial port only; the first explicit Present_Position read
        # validates communication without invoking robot.configure().
        self.__raw_bus.connect(handshake=False)

    def sync_read(self, register: str, motors=None, normalize: bool = True):
        return self.__raw_bus.sync_read(register, motors, normalize=normalize)

    def read(self, register: str, motor: str, normalize: bool = True):
        return self.__raw_bus.read(register, motor, normalize=normalize)

    def disconnect(self, disable_torque: bool = False) -> None:
        if disable_torque:
            self.__record_attempt("torque")
            raise SafetyError("read-only disconnect cannot change torque state")
        self.__raw_bus.disconnect(False)

    def write(self, register: str, *_args, **_kwargs) -> None:
        self.__record_attempt(_write_category(register))
        raise SafetyError(f"read-only motor bus blocked write to {register}")

    def sync_write(self, register: str, *_args, **_kwargs) -> None:
        self.__record_attempt(_write_category(register))
        raise SafetyError(f"read-only motor bus blocked sync_write to {register}")

    def enable_torque(self, *_args, **_kwargs) -> None:
        self.__record_attempt("torque")
        raise SafetyError("read-only motor bus blocked torque enable")

    def disable_torque(self, *_args, **_kwargs) -> None:
        self.__record_attempt("torque")
        raise SafetyError("read-only motor bus blocked torque disable")

    def configure_motors(self, *_args, **_kwargs) -> None:
        self.__record_attempt("other_register")
        raise SafetyError("read-only motor bus blocked motor configuration")

    def write_calibration(self, *_args, **_kwargs) -> None:
        self.__record_attempt("other_register")
        raise SafetyError("read-only motor bus blocked calibration write")

    def setup_motor(self, *_args, **_kwargs) -> None:
        self.__record_attempt("other_register")
        raise SafetyError("read-only motor bus blocked motor setup")


class MotorIO:
    """Explicit lifecycle wrapper; constructing/importing it never opens a bus."""

    def __init__(
        self,
        port: str,
        calibration_path: Path | str,
        pid: Mapping[str, int],
        *,
        disable_torque_on_disconnect: bool = False,
        bus: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.port = str(port)
        self.calibration_path = Path(calibration_path).resolve()
        self.pid = {key: int(pid[key]) for key in ("p", "i", "d")}
        if self.pid != {"p": 64, "i": 0, "d": 32}:
            raise ValueError("canonical PID must be P64/I0/D32")
        if not self.calibration_path.is_file():
            raise FileNotFoundError(self.calibration_path)
        self.disable_torque_on_disconnect = bool(disable_torque_on_disconnect)
        self._bus = bus
        self._robot = None
        self._clock = clock
        self._sleeper = sleeper
        self._connected = False
        self._motion_permitted = False
        self._read_only = False
        self.present_position_reads = 0
        self.present_position_read_failures = 0
        self._write_attempts = {
            "goal_position": 0,
            "pid": 0,
            "torque": 0,
            "other_register": 0,
        }
        self._physical_writes = {
            "goal_position": 0,
            "pid": 0,
            "torque": 0,
            "other_register": 0,
        }
        self.last_safe_goal: dict[str, int] | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def motion_permitted(self) -> bool:
        return self._motion_permitted

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def write_audit(self) -> WriteAudit:
        return WriteAudit(
            self._write_attempts["goal_position"],
            self._write_attempts["pid"],
            self._write_attempts["torque"],
            self._write_attempts["other_register"],
            self._physical_writes["goal_position"],
            self._physical_writes["pid"],
            self._physical_writes["torque"],
            self._physical_writes["other_register"],
        )

    def _record_write_attempt(self, category: str) -> None:
        self._write_attempts[category] += 1

    @property
    def bus(self) -> Any:
        if self._bus is None:
            raise RuntimeError("motor bus is not connected")
        return self._bus

    def _build_real_bus(self) -> Any:
        project_root = Path(__file__).resolve().parents[2]
        vendor_src = project_root / "third_party" / "lerobot" / "src"
        if str(vendor_src) not in sys.path:
            sys.path.insert(0, str(vendor_src))
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        robot = SO100Follower(
            SO100FollowerConfig(
                port=self.port,
                use_degrees=True,
                id=self.calibration_path.stem,
                calibration_dir=self.calibration_path.parent,
                disable_torque_on_disconnect=self.disable_torque_on_disconnect,
            )
        )
        self._robot = robot
        return robot.bus

    def connect(self, *, permit_motion: bool = False, read_only: bool | None = None) -> None:
        if self._connected:
            raise RuntimeError("motor bus is already connected")
        read_only = not permit_motion if read_only is None else bool(read_only)
        if permit_motion and read_only:
            raise ValueError("read-only mode cannot permit motion")
        if self._bus is None:
            self._bus = self._build_real_bus()
        if read_only and not isinstance(self._bus, _ReadOnlyBusGuard):
            self._bus = _ReadOnlyBusGuard(self._bus, self._record_write_attempt)
        try:
            self._bus.connect(handshake=not read_only)
            self._connected = True
            self._read_only = read_only
            if permit_motion:
                self.configure_and_verify_pid()
                self._motion_permitted = True
            else:
                self._motion_permitted = False
        except Exception as exc:
            self._motion_permitted = False
            self._read_only = False
            self._connected = False
            try:
                self._bus.disconnect(False)
            except Exception:
                pass
            raise communication_fault(exc) from exc

    def configure_and_verify_pid(self) -> dict[str, dict[str, int]]:
        if not self._connected:
            raise RuntimeError("motor bus is not connected")
        if self._read_only:
            self._record_write_attempt("pid")
            raise SafetyError("read-only motor bus blocked PID configuration")
        try:
            for key, register in PID_REGISTERS.items():
                for name in ARM_JOINTS:
                    self._record_write_attempt("pid")
                    self.bus.write(register, name, self.pid[key], normalize=False)
                    self._physical_writes["pid"] += 1
            readback: dict[str, dict[str, int]] = {}
            for key, register in PID_REGISTERS.items():
                values = self.bus.sync_read(register, list(ARM_JOINTS), normalize=False)
                readback[key] = {name: int(values[name]) for name in ARM_JOINTS}
                bad = [name for name in ARM_JOINTS if readback[key][name] != self.pid[key]]
                if bad:
                    raise SafetyError(f"PID {key.upper()} readback mismatch: {', '.join(bad)}")
            return readback
        except Exception as exc:
            self._motion_permitted = False
            if isinstance(exc, SafetyError):
                raise
            raise communication_fault(exc) from exc

    def read_present_positions(self, *, normalized: bool) -> MotorState:
        if not self._connected:
            raise RuntimeError("motor bus is not connected")
        try:
            positions = self.bus.sync_read("Present_Position", normalize=normalized)
            timestamp = float(self._clock())
            self.present_position_reads += 1
            return MotorState(timestamp, dict(positions), normalized)
        except Exception as exc:
            self.present_position_read_failures += 1
            self._motion_permitted = False
            raise communication_fault(exc) from exc

    def write_goal_ticks(self, goals: Mapping[str, int | float]) -> dict[str, int]:
        """Perform one atomic raw-tick write after every canonical gate has passed."""

        self._record_write_attempt("goal_position")
        if not self._connected or not self._motion_permitted:
            raise SafetyError("motion is not permitted")
        checked = require_raw_goal_ticks(goals, ALL_MOTORS)
        try:
            self.bus.sync_write("Goal_Position", checked, normalize=False)
            self._physical_writes["goal_position"] += 1
        except Exception as exc:
            self._motion_permitted = False
            raise communication_fault(exc) from exc
        self.last_safe_goal = checked.copy()
        return checked

    def move_to_start_ticks(
        self, start_goal_ticks: Mapping[str, int | float], steps: int, duration_s: float
    ) -> None:
        """Controlled interpolation primitive; explicit invocation is required."""

        self._record_write_attempt("goal_position")
        if steps <= 0 or duration_s <= 0.0:
            raise ValueError("steps and duration_s must be positive")
        target = require_raw_goal_ticks(start_goal_ticks, ARM_JOINTS)
        current = self.read_present_positions(normalized=False).positions
        for index in range(1, steps + 1):
            ratio = index / steps
            partial = {
                name: int(round(float(current[name]) + ratio * (target[name] - float(current[name]))))
                for name in ARM_JOINTS
            }
            if not self._connected or not self._motion_permitted:
                raise SafetyError("motion is not permitted")
            try:
                self.bus.sync_write("Goal_Position", partial, normalize=False)
                self._physical_writes["goal_position"] += 1
            except Exception as exc:
                self._motion_permitted = False
                raise communication_fault(exc) from exc
            self._sleeper(duration_s / steps)

    def disconnect(self) -> None:
        if self._bus is None or not self._connected:
            return
        try:
            self._bus.disconnect(self.disable_torque_on_disconnect)
        finally:
            self._motion_permitted = False
            self._read_only = False
            self._connected = False


__all__ = ["ALL_MOTORS", "ARM_JOINTS", "MotorIO", "MotorState", "WriteAudit"]
