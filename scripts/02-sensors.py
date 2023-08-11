"""An example of using tellox to get sensor readings from the drone."""
from time import sleep

import matplotlib.pyplot as plt
import tellox as tx

# Make a pilot object.
pilot = tx.Pilot()

# Sensor readings are only provided when requested, so we need to request them
readings = []
flight_time = 5.0  # seconds
framerate = 10.0  # Hz
print("Starting measurements!")
for _ in range(int(flight_time * framerate)):
    readings.append(pilot.get_sensor_readings())
    sleep(1 / framerate)
print("Done with measurements!")

# Convert the readings to numpy arrays and plot them
readings_dict = tx.aggregate_sensor_readings(readings)
# Since we didn't actually fly, readings["flight_time"] will be zero (that
# measures the time since the motors turned on), so we can just plot with
# a generic time axis
plt.plot(readings_dict["acceleration"][:, 0], label="x")
plt.plot(readings_dict["acceleration"][:, 1], label="y")
plt.plot(readings_dict["acceleration"][:, 2], label="z")
plt.xlabel("Measurement #")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.show()
