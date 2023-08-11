"""A library for interfacing with the DJI Tello drone."""
from tellox.pilot import Pilot
from tellox.utils import aggregate_sensor_readings

__all__ = ["Pilot", "aggregate_sensor_readings"]
