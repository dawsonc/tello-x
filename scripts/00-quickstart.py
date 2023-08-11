"""A minimal example of using the tellox library."""
import time

import tellox as tx

# Make a pilot object, take off, and land
pilot = tx.Pilot()
pilot.takeoff()
time.sleep(2)
pilot.land()
