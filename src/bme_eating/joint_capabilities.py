from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointCapabilities:
    version: str
    schema: str
    protocol: str
    architecture: str
    grouped: bool = False
    consistent_inputs: bool = False
    stabilized: bool = False
    feature_width: int = 137
    roi_width: int = 258
    full_data_workflow: bool = False

    @property
    def score_formula(self):
        return (f"calibrated_event_probability_{self.schema}" if self.grouped
                else "calibrated_event_probability_times_quality")


PROFILES = {
    "statsfusion_joint_event": JointCapabilities("v4.10", "v410", "statsfusion-joint-event-v1", "statsfusion_joint_event"),
    "statsfusion_joint_event_v411": JointCapabilities("v4.11", "v411", "statsfusion-joint-event-v2", "statsfusion_joint_event_v411", True),
    "statsfusion_joint_event_v412": JointCapabilities("v4.12", "v412", "statsfusion-joint-event-v3", "statsfusion_joint_event_v412", True, True),
    "statsfusion_joint_event_v413": JointCapabilities("v4.13", "v413", "statsfusion-joint-event-v4", "statsfusion_joint_event_v413", True, True, True, 201, 259),
    "statsfusion_joint_event_v414": JointCapabilities("v4.14", "v414", "statsfusion-joint-event-v5", "statsfusion_joint_event_v414", True, True, True, 201, 259, True),
}

EVENT_PROBABILITY_FORMULAS = frozenset(profile.score_formula for profile in PROFILES.values() if profile.grouped)


def capabilities(value):
    if isinstance(value, dict):
        architecture = value["model"]["architecture"]
        return PROFILES[architecture]
    profile = getattr(value, "capabilities", None)
    if profile is not None:
        return profile
    if getattr(value, "is_v412", False):
        return PROFILES["statsfusion_joint_event_v412"]
    if getattr(value, "is_v411", False):
        return PROFILES["statsfusion_joint_event_v411"]
    return PROFILES["statsfusion_joint_event"]


def supported(value, name):
    return getattr(capabilities(value), name)


def calibration_discriminates(config, payload):
    if not supported(config, "stabilized"):
        return True
    return (payload.get("discriminative") is True
            and payload.get("slope", 0) > config["calibration"]["minimum_slope"]
            and (payload.get("probability_range") or 0) > config["calibration"]["minimum_probability_range"])
