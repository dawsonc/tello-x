"""Define the main interface for tellox"""
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from djitellopy import BackgroundFrameRead, Tello
from dt_apriltags import Detection, Detector


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


class Pilot:
    """Python interface for the Tello drone.

    Wraps the [djitellopy](https://djitellopy.readthedocs.io/en/latest/)
    library with additional functionality for controlling the drone, logging
    sensor readings, and using the camera.
    """

    # Attributes for interfacing with the tello
    _tello_interface: Tello
    _frame_reader: BackgroundFrameRead

    # Camera intrinsics sourced from
    # https://tellopilots.com/threads/camera-intrinsic-parameter.2620/
    _camera_parameters = np.array(
        [921.170702, 919.018377, 459.904354, 351.238301]
    )

    # Apriltag detector
    _apriltag_detector: Detector
    _tag_size: float

    # Visualization
    _visualize: bool
    _window_name: str = "tellox"

    def __init__(
        self,
        apriltag_size: float = 0.1,
        apriltag_family: str = "tag36h11",
        visualize: bool = False,
    ):
        """
        Initialize the Tello drone interface and connect to the drone.

        Args:
            apriltag_size: The size of the apriltag in meters. Defined as the
                length of the edge of the black and white border around the
                outside of the tag.
            apriltag_family: The family of apriltags to detect. Can be
                multiple families separated by a space.
            visualize: Whether to visualize the camera feed and apriltag
                detections
        """
        # Drone initialization
        self._tello_interface = Tello()
        self._tello_interface.connect()
        self._frame_reader = self._tello_interface.get_frame_read()

        # AprilTag initialization
        self._apriltag_detector = Detector(families="tag36h11")
        self._tag_size = apriltag_size

        # Visualization if requested
        self._visualize = visualize
        if self._visualize:
            cv2.namedWindow(self._window_name)

    def takeoff(self):
        """Take off."""
        self._tello_interface.takeoff()

    def land(self):
        """Land."""
        self._tello_interface.land()

    def send_command(
        self, xyz_velocity: npt.NDArray[np.float64], yaw_velocity: float
    ):
        """Send a command to the drone sending the desired velocity.

        Cannot run faster than 1 kHz.

        Args:
            xyz_velocity: The desired velocity in the body frame (x, y, z),
                with x forward, y left, and z up. Values should be between
                +/- 1 m/s
            yaw_velocity: The desired yaw velocity in degrees per second.
                Values should be between -100 and 100.
        """
        # Convert to cm/s, which the tello expects
        self._tello_interface.send_rc_control(
            int(xyz_velocity[0] * 100),
            int(xyz_velocity[1] * 100),
            int(xyz_velocity[2] * 100),
            int(yaw_velocity),
        )

    def get_sensor_readings(self) -> dict:
        """Get the current sensor readings from the drone.

        Returns:
            A dictionary of sensor readings with the following keys
                - acceleration: acceleration in the body frame (x, y, z); m/s^2
                - velocity: velocity in the body frame (x, y, z); m/s
                - attitude: attitude in the body frame (roll, pitch, yaw); deg

        """
        # Units are specified in https://dl-cdn.ryzerobotics.com/downloads/tello/20180910/Tello%20SDK%20Documentation%20EN_1.3.pdf  # noqa: E501
        s = self._tello_interface.get_current_state()

        # Convert acceleration and velocities from cm to m basis
        accel = np.array([s["agx"], s["agy"], s["agz"]], dtype=np.float64)
        velocity = np.array([s["vgx"], s["vgy"], s["vgz"]], dtype=np.float64)
        accel /= 100.0
        velocity /= 100.0

        # Height, time of flight, and baro sensors need to convert from cm to m
        height = s["h"] / 100.0
        time_of_flight_distance = s["tof"] / 100.0
        baro = s["baro"] / 100.0

        # Other quantities are already in the correct units
        attitude = np.array(
            [s["pitch"], s["roll"], s["yaw"]], dtype=np.float64
        )
        battery = s["bat"]
        flight_time = s["time"]

        return SensorReading(
            flight_time=flight_time,
            acceleration=accel,
            velocity=velocity,
            attitude=attitude,
            height=height,
            tof_distance=time_of_flight_distance,
            baro=baro,
            battery=battery,
        )

    def set_camera_direction(self, direction: str):
        """
        Set the direction of the camera.

        Args:
            direction: The direction of the camera. Can be "forward" or
                "downward".

        Raises:
            ValueError: If the direction is not "forward" or "downward".
        """
        if direction not in ["forward", "downward"]:
            raise ValueError(
                f"Direction must be 'forward' or 'downward', not {direction}"
            )

        if direction == "forward":
            self._tello_interface.set_video_direction(Tello.CAMERA_FORWARD)
        else:
            self._tello_interface.set_video_direction(Tello.CAMERA_DOWNWARD)

    def get_camera_frame(self) -> npt.NDArray[np.uint8]:
        """Get the current frame from the camera.

        Returns:
            The current frame from the camera.
        """
        img = self._frame_reader.frame

        if self._visualize:
            cv2.imshow(self._window_name, img)
            cv2.waitKey(1)

        return img

    def detect_tags(self, img: npt.NDArray[np.uint8]) -> list[Detection]:
        """
        Detect apriltags in an image.

        See https://github.com/duckietown/lib-dt-apriltags for documentation
        on the AprilTag detector. The pose of the tag is returned in the
        camera frame, with tag.pose_R containing the rotation matrix from the
        camera frame to the tag frame, and tag.pose_t containing the position
        of the tag in the camera frame.

        The camera's frame has x to the right of the drone, y down, and z
        forward. IMPORTANT: this is different from the drone body frame, which
        has x forward, y left, and z up.

        Args:
            img: The image to detect apriltags in.

        Returns:
            A list of dt_apriltags.Detection objects (empty if no tags were
            detected).
        """
        tags = self._apriltag_detector.detect(
            img,
            estimate_tag_pose=True,
            camera_params=self._camera_parameters,
            tag_size=self._tag_size,
        )

        # Visualize if requested
        if self._visualize:
            # Copy the image so we can draw on it
            img = img.copy()

            # Draw the april tag centers and tag IDs
            for tag in tags:
                cv2.circle(
                    img,
                    (int(tag.center[0]), int(tag.center[1])),
                    3,
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    img,
                    str(tag.tag_id),
                    (int(tag.center[0]), int(tag.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        return tags

    @property
    def R_drone_camera(self) -> npt.NDArray[np.float64]:
        """Rotation matrix from the drone body frame to the camera frame."""
        return np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ],
            dtype=np.float64,
        )

    def convert_to_drone_frame(
        self, xyz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Convert a vector from the camera frame to the drone body frame.

        Args:
            xyz: A vector in the camera frame.

        Returns:
            The vector in the drone body frame.
        """
        return self.R_drone_camera @ xyz
