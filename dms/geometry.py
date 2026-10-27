"""ADDW gaze areas (Annex I Part 1 3.3.1) on an ILLUSTRATIVE left-hand-drive cabin (design assumption DA-18).

Angles are seen from the ocular reference point (3.3.1: UN R46 eye point, 635 mm above the R-point):
yaw > 0 to the right, pitch > 0 upwards.  Glass areas are approximated by angular rectangles; Area 2 adds the
10 deg margin of 3.3.1.2 (b).  This is NOT a real vehicle: it only makes the area logic and the ADDW
fixation-zone applicability ("when possible, within Area 3", Part 2 1.4.2) explicit and testable.

Interpretation note: 3.3.1.1 calls Area 1 the "overlap" of the roof and the zone outside the +/-55 deg planes,
while 3.3.1.2 calls Area 2 a "union".  We read Area 1 as roof OR outside +/-55 deg (union) - unverified.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import labels as L
from .regulations import ADDW_AREA1_YAW_DEG, ADDW_AREA2_MARGIN_DEG, ADDW_AREA3_PITCH_DEG

# (yaw_min, yaw_max, pitch_min, pitch_max) in degrees - DA-18
WINDSCREEN = (-30.0, 55.0, -10.0, 18.0)
DRIVER_WINDOW = (-100.0, -35.0, -15.0, 15.0)
PASSENGER_WINDOW = (58.0, 100.0, -20.0, 10.0)
ROOF_PITCH_DEG = 25.0


def _inside(yaw: float, pitch: float, rect: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    y0, y1, p0, p1 = rect
    return (y0 - margin) <= yaw <= (y1 + margin) and (p0 - margin) <= pitch <= (p1 + margin)


def in_area1(yaw: float, pitch: float) -> bool:
    """3.3.1.1: roof, or outside the vertical planes at +/-55 deg (read as a union, see module note)."""
    return pitch > ROOF_PITCH_DEG or abs(yaw) > ADDW_AREA1_YAW_DEG


def in_area2(yaw: float, pitch: float) -> bool:
    """3.3.1.2: windscreen and windows plus 10 deg around them."""
    m = ADDW_AREA2_MARGIN_DEG
    return any(_inside(yaw, pitch, r, m) for r in (WINDSCREEN, DRIVER_WINDOW, PASSENGER_WINDOW))


def in_area3(yaw: float, pitch: float) -> bool:
    """3.3.1.3: below the plane 30 deg downward, excluding Area 1 and Area 2 by default (3.3.1.3 (a)(b))."""
    return pitch < ADDW_AREA3_PITCH_DEG and not in_area1(yaw, pitch) and not in_area2(yaw, pitch)


def addw_area(yaw: float, pitch: float) -> str:
    if in_area3(yaw, pitch):
        return "3"
    if in_area2(yaw, pitch):
        return "2"
    if in_area1(yaw, pitch):
        return "1"
    return "-"          # none of the three (e.g. a centre display between the windscreen margin and -30 deg)


@dataclass(frozen=True)
class Zone:
    key: str            # ADDW zone letter or Euro NCAP location id
    name: str
    yaw: float
    pitch: float
    label: str          # gaze label the perception layer is assumed to output (DA-18)

    @property
    def area(self) -> str:
        return addw_area(self.yaw, self.pitch)


# ADDW Part 2 1.4.2 zones (a)-(n) placed in the illustrative cabin
ADDW_ZONES: dict[str, Zone] = {z.key: z for z in [
    Zone("a", "driver's left knee", -12, -55, L.OFF_ROAD),
    Zone("b", "driver's right knee", 8, -55, L.OFF_ROAD),
    Zone("c", "driver's lap", -2, -62, L.OFF_ROAD),
    Zone("d", "passenger's footwell", 38, -50, L.OFF_ROAD),
    Zone("e", "passenger's seat surface", 45, -40, L.OFF_ROAD),
    Zone("f", "glove box", 50, -33, L.OFF_ROAD),
    Zone("g", "air vents left of driver", -38, -12, L.OFF_ROAD),
    Zone("h", "air vents right of driver", 14, -18, L.OFF_ROAD),
    Zone("i", "instrument cluster", -3, -16, L.INSTRUMENT),
    Zone("j", "steering wheel buttons", -3, -27, L.INSTRUMENT),
    Zone("k", "gear shifter", 25, -48, L.OFF_ROAD),
    Zone("l", "HVAC controls", 22, -36, L.INFOTAINMENT),
    Zone("m", "infotainment display", 24, -23, L.INFOTAINMENT),
    Zone("n", "centre console", 18, -50, L.OFF_ROAD),
]}

# Euro NCAP DE v1.2 1.2.1-1.2.3 gaze locations -> label and task type (DA-18).
# task: "non_driving" | "driving" | "phone_basic" | "phone_advanced"
ENCAP_LOCATIONS: dict[str, tuple[str, str, float, float]] = {
    # name: (label, task, yaw, pitch)
    "driver side window": (L.OFF_ROAD, "non_driving", -70, 0),
    "passenger side window": (L.OFF_ROAD, "non_driving", 75, -5),
    "passenger footwell": (L.OFF_ROAD, "non_driving", 38, -50),
    "passenger face": (L.OFF_ROAD, "non_driving", 62, 5),
    "in-vehicle infotainment system": (L.INFOTAINMENT, "non_driving", 24, -23),
    "glovebox": (L.OFF_ROAD, "non_driving", 50, -33),
    "rear passenger": (L.OFF_ROAD, "non_driving", 150, 0),
    "rear view mirror": (L.MIRROR, "driving", 20, 12),
    "driver side mirror": (L.MIRROR, "driving", -50, -8),
    "passenger side mirror": (L.MIRROR, "driving", 68, -10),
    "instrument cluster": (L.INSTRUMENT, "driving", -3, -16),
    "driver knee outboard": (L.PHONE, "phone_basic", -15, -55),
    "driver knee inboard": (L.PHONE, "phone_basic", 5, -55),
    "driver lap": (L.PHONE, "phone_basic", -2, -62),
    "phone mounted on dashboard outboard": (L.PHONE, "phone_basic", -32, -12),
    "phone in charge port": (L.PHONE, "phone_basic", 20, -45),
    "phone held centre of steering wheel": (L.PHONE, "phone_basic", -2, -30),
    "phone held 9-11 / 13-15 o'clock on wheel": (L.PHONE, "phone_advanced", -14, -20),
    "phone held in view of windscreen": (L.PHONE, "phone_advanced", -15, 2),
    "phone held in view of instrument cluster": (L.PHONE, "phone_advanced", -3, -14),
}
