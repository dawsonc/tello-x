"""A minimal example of using the tellox library."""
from time import sleep

import cv2
from tellox import Pilot

# Make a pilot object
pilot = Pilot(visualize=True)

# Take off, wait a bit, then land
pilot.takeoff()
sleep(5)
pilot.land()
