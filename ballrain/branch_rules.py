"""Closed schema for tunable diversion policy (stored in the pinned description)."""
from dataclasses import dataclass, fields
import sys

from .path import RouteError


@dataclass(frozen=True)
class BranchRules:
    stall_pause_speed: float = 1.0
    stall_speed_source: str = "speed+speed_change"
    stall_odometer: str = "local_s"
    rest_stop_speed: float | None = None  # inherit EnvConfig.rest_speed
    rest_max_drive_frames: int | None = None  # inherit EnvConfig.rest_max_frames
    rest_max_brake_frames: int | None = None
    rest_require_support: bool = True
    rest_position_tolerance: float = 2.0
    rest_fallback: str = "same_tas_frame"
    drive_arrival_radius: float = 1.0
    drive_gain: float = 0.8
    drive_speed_cap: float = 6.0
    drive_deadband: float = 0.3
    rest_control_repeat: int = 2

    @classmethod
    def from_desc(cls, desc):
        raw = desc.get("rules", {})
        if not isinstance(raw, dict):
            raise RouteError("branch rules must be an object")
        schema = {f.name: f for f in fields(cls)}
        enums = {"stall_speed_source": ("speed", "speed+speed_change"),
                 "stall_odometer": ("local_s", "s"), "rest_fallback": ("same_tas_frame", "error")}
        for k, v in raw.items():
            if k not in schema:
                raise RouteError(f"unknown branch rule: {k}")
            ty = schema[k].type
            if ty == str:
                valid = v in enums[k]
            elif ty == bool:
                valid = type(v) is bool
            elif v is None:
                valid = schema[k].default is None
            elif ty in (int, int | None):
                valid = type(v) is int and 1 <= v <= 2147483647
            else:
                valid = type(v) in (float, int) and 0 <= v <= sys.float_info.max
                if k == "rest_stop_speed" and v == 0:
                    valid = False
            if not valid:
                raise RouteError(f"invalid branch rule {k}: {v!r}")
        return cls(**raw)


DEFAULT_RULES = BranchRules()
