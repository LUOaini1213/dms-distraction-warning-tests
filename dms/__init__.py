"""P7 cockpit DMS & distraction-warning logic layer.

Input : per-frame gaze-region label, eyelid closure, vehicle speed and sensor-availability flags.
Output: per-frame HMI requests (visual / acoustic / haptic), ADAS sensitivity request, EF request.

All scenario data in this project are SYNTHETIC (see dms/scenarios.py).  Personal project, AI-assisted.
Regulatory numbers live in dms/regulations.py with their clause; design assumptions in dms/config.py.
"""
__version__ = "0.1.0"
DATA_ORIGIN = "SYNTHETIC"
