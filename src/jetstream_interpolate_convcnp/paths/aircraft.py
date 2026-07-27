"""
aircraft.py
-----------
Simple aircraft performance profiles used by the loss function. These are
deliberately lightweight (constant cruise TAS + regime-based fuel flow)
rather than a full BADA-style performance model, but the dataclass is easy
to extend or replace with something more detailed (e.g. Mach/altitude
dependent TAS, fuel flow polynomials, etc).
"""
from dataclasses import dataclass


@dataclass
class AircraftProfile:
    name: str
    cruise_tas_mps: float          # true airspeed at cruise, m/s
    fuel_flow_cruise_kgps: float   # kg/s fuel burn in level flight
    fuel_flow_climb_kgps: float    # kg/s fuel burn while climbing
    fuel_flow_descent_kgps: float  # kg/s fuel burn while descending
    max_climb_rate_mps: float = 12.0     # m/s, used to normalise altitude penalty
    max_descent_rate_mps: float = 12.0   # m/s
    max_turn_rate_dps: float = 3.0       # deg/s, "standard rate" turn ~ airliner
    service_ceiling_m: float = 12500.0
    min_alt_m: float = 3000.0
    vertical_rate_threshold_mps: float = 0.5  # |dz/dt| below this counts as "cruise"


# A handful of representative presets. TAS and fuel-flow numbers are rough
# publicly-known ballpark figures for cruise conditions -- replace with real
# performance-model outputs (e.g. BADA, OEM data) for anything beyond
# illustrative use.
PROFILES = {
    "regional_turboprop": AircraftProfile(
        name="regional_turboprop",
        cruise_tas_mps=140.0,          # ~272 kt, e.g. Dash 8 / ATR
        fuel_flow_cruise_kgps=0.30,
        fuel_flow_climb_kgps=0.45,
        fuel_flow_descent_kgps=0.18,
        max_climb_rate_mps=10.0,
        max_descent_rate_mps=10.0,
        max_turn_rate_dps=3.0,
        service_ceiling_m=7600.0,
        min_alt_m=1500.0,
    ),
    "narrowbody_jet": AircraftProfile(
        name="narrowbody_jet",
        cruise_tas_mps=230.0,          # ~447 kt, e.g. A320/B737 cruise
        fuel_flow_cruise_kgps=0.72,
        fuel_flow_climb_kgps=1.05,
        fuel_flow_descent_kgps=0.35,
        max_climb_rate_mps=12.0,
        max_descent_rate_mps=12.0,
        max_turn_rate_dps=3.0,
        service_ceiling_m=12500.0,
        min_alt_m=3000.0,
    ),
    "widebody_jet": AircraftProfile(
        name="widebody_jet",
        cruise_tas_mps=250.0,          # ~486 kt, e.g. B777/A350 cruise
        fuel_flow_cruise_kgps=2.6,
        fuel_flow_climb_kgps=3.6,
        fuel_flow_descent_kgps=1.1,
        max_climb_rate_mps=11.0,
        max_descent_rate_mps=11.0,
        max_turn_rate_dps=2.0,
        service_ceiling_m=13100.0,
        min_alt_m=4000.0,
    ),
    "business_jet": AircraftProfile(
        name="business_jet",
        cruise_tas_mps=245.0,
        fuel_flow_cruise_kgps=0.35,
        fuel_flow_climb_kgps=0.55,
        fuel_flow_descent_kgps=0.15,
        max_climb_rate_mps=15.0,
        max_descent_rate_mps=15.0,
        max_turn_rate_dps=4.0,
        service_ceiling_m=15500.0,
        min_alt_m=3000.0,
    ),
}
