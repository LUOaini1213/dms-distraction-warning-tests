"""Regulatory / protocol numbers - every value carries its source.  Full clause list: regulation_requirements.csv.

Documents actually read for this project (2026-10-27):

GSR   Regulation (EU) 2019/2144, final adopted text PE-CONS 82/19 (Council register).  In PE-CONS the dates in
      Annex II are still placeholders ("[24 months after the date of application]").
ADDW  Commission Delegated Regulation (EU) 2023/2590, read as the adopted Commission text C(2023) 4523 final,
      transmitted to the Council as ST 11895/23 INIT (articles) and ST 11895/23 ADD 1 (Annexes I-II).
      EUR-Lex returned empty pages to the fetch tool, so the OJ publication (OJ L, 2023/2590, 22.11.2023)
      itself was not compared word by word.
DDAW  Commission Delegated Regulation (EU) 2021/1341, read in the official EEA Norwegian translation
      (EEA Supplement No 87, 28.11.2024, pp. 745-760).  The English OJ text (OJ L 292, 16.8.2021, p. 4) was not
      reachable; numbers are unambiguous, wording below is our translation.
ENCAP Euro NCAP "Safe Driving - Driver Engagement Protocol" Version 1.2, July 2026 (implementation January 2026)
      and Technical Bulletin SD-202 "Driver Monitoring Test Procedure" Version 1.2, July 2026.
      Values printed in square brackets in SD-202 (e.g. "[0.2]s") are provisional and flagged as such here.
NHTSA "Visual-Manual NHTSA Driver Distraction Guidelines for In-Vehicle Electronic Devices",
      Federal Register Vol. 78 No. 81, pp. 24818-24890, 26 April 2013, Docket NHTSA-2010-0053.
IVASA ASEAN NCAP IVASA "Test Protocol for Intelligent Cabin Interaction" (2026) - a separate beyond-rating
      programme, NOT part of the ASEAN NCAP 2026-2030 star rating.
ASEAN ASEAN NCAP Assessment Protocol - Safety Assist v3.2 (2026-2030 protocols): driver state monitoring is
      only an "Advanced SAT" functional definition (7.2.4) scored on fitment (7.3-7.5); no thresholds exist.

Anything not in these documents is a DESIGN ASSUMPTION and lives in dms/config.py with a DA-xx tag.
"""
from __future__ import annotations

# ======================================================================= ADDW (EU) 2023/2590, Annex I
ADDW_SRC = "EU 2023/2590 ADDW"
ADDW_ACTIVATION_KPH = 20.0        # Part 1 3.1.1 "automatically activated above the speed of 20 km/h"
ADDW_CALIBRATION_MAX_S = 60.0     # 3.1.1 "cumulative period of up to 1 minute of driving at speeds >=20 km/h"
ADDW_HIGH_SPEED_KPH = 50.0        # 3.3.2.1 (a) "vehicle's speed at 50 km/h or above"
ADDW_T_MAX_HIGH_S = 3.5           # 3.3.2.1 (b) gaze in Area 3 "for a maximum time of 3,5 seconds"
ADDW_LOW_SPEED_KPH = 20.0         # 3.3.2.2 (a) "vehicle's speed at 20 km/h or above"
ADDW_T_MAX_LOW_S = 6.0            # 3.3.2.2 (b) "maximum time of 6 seconds in the nominal situation"
ADDW_NON_NOMINAL_EXTRA_S = 1.5    # 3.3.2.1 / 3.3.2.2 non-nominal situations "may extend ... by additional 1,5 s"
ADDW_SACCADE_TOL_MIN_S = 0.050    # 3.3.2.4 "minimum time tolerance of 50 milliseconds (eye saccades)"
ADDW_EYE_ABOVE_R_POINT_MM = 635.0  # 3.3.1 ocular reference point, UN R46 eye point above R-point
ADDW_AREA1_YAW_DEG = 55.0         # 3.3.1.1 (b) vertical planes rotated +55 deg / -55 deg
ADDW_AREA2_MARGIN_DEG = 10.0      # 3.3.1.2 (b) "10 deg around the area of the windscreen and windows"
ADDW_AREA3_PITCH_DEG = -30.0      # 3.3.1.3 "any area below a plane, 30 deg downward"
ADDW_ACOUSTIC_HZ = (200.0, 8000.0)  # 3.4.3.2 majority of the acoustic warning within 200-8000 Hz
ADDW_ACOUSTIC_DB = (50.0, 90.0)     # 3.4.3.2 ... and 50-90 dB
# Part 2 - spot-check testing by the technical service
ADDW_TEST_SPEED_TOL_KPH = 1.0     # 1.2.2 displayed speed measured to +/-1 km/h
ADDW_TEST_SPEED_BANDS_KPH = ((20.0, 35.0), (50.0, 65.0))  # 1.5.1 each fixation point once in each band
ADDW_TEST_PRE_ATTENTIVE_S = 60.0  # 2.3.1 (b) system assesses driver "not distracted for at least 60 seconds"
ADDW_TEST_BETWEEN_POINTS_S = 15.0  # 2.3.5 / 2.3.9 not distracted >= 15 s before / after each fixation point
ADDW_TEST_HOLD_EXTRA_S = 3.0      # 2.3.8 hold gaze until warning or "expected time ... passed by, at least, 3 s"
ADDW_TEST_FN_HIGH_S = 4.0         # 3.1 false negative: no warning within 4 s (incl. 0,5 s buffer), 3.3.2.1 case
ADDW_TEST_FN_LOW_S = 6.5          # 3.2 false negative: no warning within 6,5 s (incl. 0,5 s buffer), 3.3.2.2 case
ADDW_TEST_RETEST_MAX = 2          # 4.1 re-test a false-negative fixation point at most two times
# 1.4.2 fixation-point zones (a)-(n); tested "if present ... and, when possible, within Area 3"
ADDW_FIXATION_ZONES = {
    "a": "driver's left knee", "b": "driver's right knee", "c": "driver's lap",
    "d": "passenger's footwell", "e": "passenger's seat surface", "f": "glove box",
    "g": "air vents left of driver", "h": "air vents right of driver", "i": "instrument cluster",
    "j": "steering wheel buttons", "k": "gear shifter", "l": "HVAC controls",
    "m": "infotainment display", "n": "centre console",
}

# ======================================================================= DDAW (EU) 2021/1341, Annex I
DDAW_SRC = "EU 2021/1341 DDAW"
DDAW_SCOPE_VMAX_KPH = 70.0        # Art. 1: M and N vehicles with maximum design speed > 70 km/h
DDAW_ACTIVATION_KPH = 70.0        # Part 1 3.1.4 automatically activated above 70 km/h
DDAW_RANGE_KPH = (65.0, 130.0)    # 3.1.5 once active, normally functions within 65-130 km/h (or vmax)
DDAW_MONITOR_START_MAX_S = 300.0  # 3.1.6 less than 5 minutes until drowsiness is actively monitored
DDAW_KSS_WARN = 8                 # 3.3.1 shall warn at KSS >= 8 (Karolinska Sleepiness Scale, Appendix)
DDAW_KSS_MAY_WARN = 7             # 3.3.1 may warn at KSS 7
# Part 2 - manufacturer validation
DDAW_MIN_PARTICIPANTS = 10        # Part 2 3.1 at least 10 participants
DDAW_MIN_EVENTS = 10              # 3.1 sum of true positives + false negatives at least 10
DDAW_KSS_INTERVAL_S = 300.0       # 5.1.2 KSS self-rating about every 5 minutes
DDAW_TP_KSS = 7                   # 5.1.4 warning is a true positive if previous or next rating is KSS >= 7
DDAW_MEAN_SENS_MIN = 0.40         # 8.1 (a) mean sensitivity above 40 %
DDAW_LB_SENS_MIN = 0.20           # 8.1 (b) lower bound of the 90 % confidence interval above 20 %
DDAW_LB_Z = 1.645                 # 8.1 (b) "mean (sensitivity) - 1,645 x standard deviation (sensitivity)"
DDAW_ADJ_LONG_INTERVAL = (0.05, 0.025)  # 8.1 (c) +5 / +2,5 points if rating interval > 15 min
DDAW_ADJ_OPEN_ROAD = (-0.05, -0.025)    # 8.1 (d) -5 / -2,5 points if tested on an open road
DDAW_LEARNING_EXCL_MAX_S = 1800.0  # 8.2 exclude learning phase or first 30 min after activation (shorter)

# ======================================================================= Euro NCAP 2026
ENCAP_DE_SRC = "Euro NCAP DE v1.2"
ENCAP_SD202_SRC = "Euro NCAP SD-202 v1.2"
ENCAP_DSM_MIN_SPEED_KPH = 20.0    # DE 1.1: distraction/microsleep/sleep/unresponsive warn at >= 20 km/h
ENCAP_IMPAIR_MIN_SPEED_KPH = 50.0  # DE 1.1 and 1.3.2: impairment (drowsiness) functional from 50 km/h
ENCAP_IMPAIR_LEARNING_MAX_S = 600.0  # DE 1.1: learning period of up to 10 minutes at >= 50 km/h
ENCAP_START_MEASURE_MAX_S = 60.0  # DE 1.1: total of 1 minute of driving at >= 10 km/h to start measuring
ENCAP_START_MEASURE_KPH = 10.0
ENCAP_LGA_RANGE_S = (3.0, 4.0)    # DE 1.2.1 long distraction "between 3 and 4 seconds"; SD-202 1.1 TLGA
ENCAP_LGA_PRE_ON_ROAD_S = 4.0     # DE 1.2.1 preceded by a four second on-road gaze; SD-202 T0 = Taway - 4.0 s
ENCAP_VATS_CUM_S = 10.0           # DE 1.2.2 "cumulative 10 seconds within a 30 second time period"
ENCAP_VATS_WINDOW_S = 30.0
ENCAP_KSS_DROWSY = 7              # DE 1.3.2 detect a driver "reaching a KSS level >7 at the latest"
ENCAP_MICROSLEEP_S = (1.0, 2.0)   # DE 1.3.3 short eye closure 1-2 s; SD-202 1.1 TMicrosleep 1-2 s
ENCAP_SLEEP_S = 3.0               # DE 1.3.4 continued eye closure >= 3 s; SD-202 1.1 fixed to 3 s
ENCAP_UNRESP_EYES_S = 6.0         # DE 1.3.5 eyes closed for >= 6 s
ENCAP_UNRESP_GAZE_S = 3.0         # DE 1.3.5 gaze not back on the road within 3 s after a distraction warning
ENCAP_INFORM_S = 10.0             # DE 1.1.1.2 "Inform if not functional": notify within ten seconds
ENCAP_FSS_LATENCY_S = 1.0         # DE 1.4.3.1 FSS activated <= 1 s of any continuous gaze away from the road
ENCAP_FSS_DELTA_TTC_S = 0.2       # DE 1.4.3.1 FCW/AEB trigger time differs by at least 200 ms
ENCAP_FSS_OFF_ROAD_S = 1.0        # DE 1.4.3 table: short off-road glances -> >= 1 s after continuous road gaze
ENCAP_FSS_OFF_STATE_S = 2.0       # DE 1.4.3 table: all transient states -> >= 2 s after end of state
ENCAP_EF_AFTER_WARN_S = 5.0       # DE 1.4.3.3 EF no later than 5 s after the distinct warning phase started
# SD-202 test execution
ENCAP_VIDEO_MIN_HZ = 25.0         # SD-202 1: record video at >= 25 Hz
ENCAP_TEST_SPEED_KPH = (20.0, 80.0)  # SD-202 4.1 warning-timing tests at 20-80 km/h
ENCAP_LD_HOLD_MAX_S = 5.0         # SD-202 4.1.1.1 maintain gaze up to 5 s
ENCAP_LD_TOL_S = 0.2              # SD-202 4.1.1.1 tolerance "[0.2]s" (bracketed = provisional)
ENCAP_VATS_TOL_S = 2.0            # SD-202 4.1.1.2 Twarn expected at dT_VATS +/- "[2.0]" s (provisional)
ENCAP_MS_HOLD_MAX_S = 4.0         # SD-202 4.1.2.2 close eyes up to 4 s, tolerance [0.2] s
ENCAP_SLEEP_HOLD_MAX_S = 7.0      # SD-202 4.1.2.3 close eyes up to 7 s, tolerance [0.2] s
ENCAP_UNRESP_SLEEP_HOLD_S = 13.0  # SD-202 4.2.3.3.1 eyes closed for a minimum of 13 s
ENCAP_UNRESP_LAP_HOLD_S = 14.0    # SD-202 4.2.3.3.2 lap gaze for a minimum of 14 s
ENCAP_NOT_FUNCTIONAL_HOLD_S = 10.0  # SD-202 4.3 hand occluding a cluster-mounted DMS for >= 10 s
ENCAP_TRANSITION_MAX_S = {"owl": 0.48, "lizard": 0.16, "body_lean": 1.0}  # SD-202 3.2.3.2

# ======================================================================= NHTSA 2013 (EGDS acceptance)
NHTSA_SRC = "NHTSA VM-DDG 2013"
NHTSA_PARTICIPANTS = 24           # VI.A.4 six per age group 18-24, 25-39, 40-54, 55+; VI.A.5 12 men / 12 women
NHTSA_PASS_MIN = 21               # VI.E.14 "for at least 21 of the 24 test participants"
NHTSA_GLANCE_S = 2.0              # VI.E.14 (a) glances > 2.0 s; (b) mean glance <= 2.0 s
NHTSA_LONG_GLANCE_SHARE = 0.15    # VI.E.14 (a) "no more than 15 percent (rounded up)"
NHTSA_TEORT_S = 12.0              # VI.E.14 (c) sum of glance durations <= 12.0 s
NHTSA_TSOT_S = 12.0               # VI.G.17 occlusion: mean total shutter open time <= 12.0 s
NHTSA_AGE_GROUPS = ("18-24", "25-39", "40-54", "55+")

# ======================================================================= ASEAN NCAP IVASA (beyond rating)
IVASA_SRC = "ASEAN NCAP IVASA ICI-TP"
IVASA_DISTRACTION_HOLD_S = 5.0    # 4.1.1 look at the distraction point for 5 s; success = warning within it
IVASA_FATIGUE_HOLD_S = 5.0        # 4.1.2 keep both eyes closed for 5 s
IVASA_REPEATS = 3                 # 4.1.1 (d) / 4.1.2 (c) each scenario three times
IVASA_EXTRA_HOLD_S = 2.0          # 4.1.2 Note 1: keep the action 2 s after the warning threshold
IVASA_DISTRACTION_POINTS = ("infotainment display", "passenger footwell", "phone charging area")
