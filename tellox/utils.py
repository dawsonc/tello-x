"""Utility functions for TelloX."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SensorReading:
    flight_time: float
    acceleration: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    attitude: npt.NDArray[np.float64]
    height: float
    tof_distance: float
    baro: float
    battery: float


def aggregate_sensor_readings(
    readings: list[SensorReading],
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Aggregate a list of readings into a set of numpy arrays for each field.

    Args:
        readings: A list of readings to aggregate.

    Returns:
        A dictionary containing numpy arrays for each field of SensorReading,
        with the measurements for each reading concatenated along the first
        axis.
    """
    # Accumulate everything into a list
    flight_time_readings = []
    acceleration_readings = []
    velocity_readings = []
    attitude_readings = []
    height_readings = []
    tof_distance_readings = []
    baro_readings = []
    battery_readings = []

    for reading in readings:
        flight_time_readings.append(reading.flight_time)
        acceleration_readings.append(reading.acceleration)
        velocity_readings.append(reading.velocity)
        attitude_readings.append(reading.attitude)
        height_readings.append(reading.height)
        tof_distance_readings.append(reading.tof_distance)
        baro_readings.append(reading.baro)
        battery_readings.append(reading.battery)

    # Convert everything to numpy arrays
    flight_time_readings = np.array(flight_time_readings)
    acceleration_readings = np.array(acceleration_readings)
    velocity_readings = np.array(velocity_readings)
    attitude_readings = np.array(attitude_readings)
    height_readings = np.array(height_readings)
    tof_distance_readings = np.array(tof_distance_readings)
    baro_readings = np.array(baro_readings)
    battery_readings = np.array(battery_readings)

    return {
        "flight_time": flight_time_readings,
        "acceleration": acceleration_readings,
        "velocity": velocity_readings,
        "attitude": attitude_readings,
        "height": height_readings,
        "tof_distance": tof_distance_readings,
        "baro": baro_readings,
        "battery": battery_readings,
    }
