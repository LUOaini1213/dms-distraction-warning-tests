"""Gaze-region labels (the input taxonomy) and how the logic groups them.

The perception layer (camera, face/eye tracking, gaze classifier) is OUT of scope: the DMS logic only sees
one label per frame.  Label meaning used throughout the project (design assumption DA-18):

  road          forward road view through the windscreen
  mirror        interior / exterior mirrors (driving task)
  instrument    instrument cluster and steering-wheel area (driving task)
  infotainment  centre display and centre-stack controls (non-driving task)
  phone         a mobile phone wherever it is held or mounted (non-driving task)
  off_road      any other non-driving location: side windows, passenger, footwell, glovebox, lap, rear seats
  unknown       gaze not determinable (blink, eyes closed, tracking loss, occlusion)
"""
ROAD = "road"
MIRROR = "mirror"
INSTRUMENT = "instrument"
INFOTAINMENT = "infotainment"
PHONE = "phone"
OFF_ROAD = "off_road"
UNKNOWN = "unknown"

REGIONS = (ROAD, MIRROR, INSTRUMENT, INFOTAINMENT, PHONE, OFF_ROAD, UNKNOWN)
DRIVING_TASK = frozenset({MIRROR, INSTRUMENT})          # Euro NCAP DE v1.2 1.2.1/1.2.2 "Driving task" locations
NON_DRIVING = frozenset({INFOTAINMENT, PHONE, OFF_ROAD})  # counted for distraction warnings (design choice)
EYES_OFF_ROAD = frozenset({MIRROR, INSTRUMENT, INFOTAINMENT, PHONE, OFF_ROAD})
