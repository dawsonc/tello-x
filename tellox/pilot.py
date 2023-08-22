"""Define the main interface for tellox"""
import logging
import time
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detection, Detector
from scipy.spatial.transform import Rotation as R

from .utils import SensorReading

# Funky stuff alert! Due to a conflict between opencv and av (a dependency in
# dji-tellopy), we need to import cv2 and create a window before importing
# dji-tellopy. See https://github.com/PyAV-Org/PyAV/issues/1050

cv2.namedWindow("Starting tellox...")
cv2.waitKey(1)
cv2.destroyAllWindows()

from djitellopy import BackgroundFrameRead, Tello  # noqa: E402


class Pilot:
    """Python interface for the Tello drone.

    Wraps the [djitellopy](https://djitellopy.readthedocs.io/en/latest/)
    library with additional functionality for controlling the drone, logging
    sensor readings, and using the camera.
    """

    # Track the time since the last takeoff for logging purposes
    last_takeoff_time: float

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
        log_level: int = logging.WARN,
        apriltag_quad_decimate: int = 4,
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
            log_level: set the desired logging level for the Tello driver.
                e.g. logging.INFO, logging.WARN
            apriltag_quad_decimate: Decimate the image by this factor before
                detecting apriltags. This can speed up detection at the cost
                of detection range.
        """
        # Drone initialization
        Tello.LOGGER.setLevel(log_level)
        self._tello_interface = Tello()
        self._tello_interface.connect()
        self._tello_interface.streamon()
        self._frame_reader = self._tello_interface.get_frame_read()

        # AprilTag initialization
        self._apriltag_detector = Detector(
            families="tag36h11", quad_decimate=apriltag_quad_decimate
        )
        self._tag_size = apriltag_size

        # Visualization if requested
        self._visualize = visualize
        if self._visualize:
            cv2.namedWindow(self._window_name)
            cv2.waitKey(1)

    def takeoff(self):
        """Take off."""
        self._tello_interface.takeoff()
        self.last_takeoff_time = time.time()

    def land(self):
        """Land."""
        self._tello_interface.land()

    def send_control(
        self, xyz_velocity: npt.NDArray[np.float64], yaw_velocity: float
    ):
        """Send a control command to the drone sending the desired velocity.

        Cannot run faster than 1 kHz.

        Args:
            xyz_velocity: The desired velocity in the body frame (x, y, z),
                with x forward, y left, and z up. Values should be between
                +/- 1 m/s
            yaw_velocity: The desired yaw velocity in degrees per second.
                Values should be between -100 and 100.
        """
        # Convert to cm/s, which the tello expects
        # Tello expects commands in order left/right (y), forward/backward (x),
        # up/down (z).
        self._tello_interface.send_rc_control(
            int(xyz_velocity[1] * 100),
            int(xyz_velocity[0] * 100),
            int(xyz_velocity[2] * 100),
            int(yaw_velocity),
        )

    def get_sensor_readings(self) -> SensorReading:
        """Get the current sensor readings from the drone.

        Returns:
            A SensorReading object with the current sensor readings.

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
        # We could get the flight time from the state, but it's only logged
        # as an INTEGER?!?!?!?! So instead we track it ourselves.
        flight_time = time.time() - self.last_takeoff_time

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

    def get_camera_frame(
        self, visualize: Optional[bool] = None
    ) -> npt.NDArray[np.uint8]:
        """Get the current frame from the camera.

        Args:
            visualize: if None, default to self._visualize. Otherwise, override
                the pilot-level visualization setting.

        Returns:
            The current frame from the camera.
        """
        img = self._frame_reader.frame

        if visualize if visualize is not None else self._visualize:
            # Convert from RGB to BGR for visualization with OpenCV
            img_to_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(self._window_name, img_to_show)
            cv2.waitKey(1)

        return img

    def detect_tags(
        self, img: npt.NDArray[np.uint8], visualize: Optional[bool] = None
    ) -> list[Detection]:
        """
        Detect apriltags in an image.

        See https://github.com/pupil-labs/apriltags for documentation
        on the AprilTag detector. The pose of the tag is returned in the
        camera frame, with tag.pose_R containing the rotation matrix from the
        camera frame to the tag frame, and tag.pose_t containing the position
        of the tag in the camera frame.

        The camera's frame has x to the right of the drone, y down, and z
        forward. IMPORTANT: this is different from the drone body frame, which
        has x forward, y left, and z up.

        Args:
            img: The image to detect apriltags in.
            visualize: if None, default to self._visualize. Otherwise, override
                the pilot-level visualization setting.

        Returns:
            A list of dt_apriltags.Detection objects (empty if no tags were
            detected).
        """
        # If the image is in RGB, convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        tags = self._apriltag_detector.detect(
            img_gray,
            estimate_tag_pose=True,
            camera_params=self._camera_parameters,
            tag_size=self._tag_size,
        )

        # Visualize if requested
        if visualize if visualize is not None else self._visualize:
            # Copy the image so we can draw on it
            img_to_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw the april tag centers and tag IDs
            for tag in tags:
                cv2.circle(
                    img_to_show,
                    (int(tag.center[0]), int(tag.center[1])),
                    3,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    img_to_show,
                    (f"ID {tag.tag_id}"),
                    (int(tag.center[0]) + 5, int(tag.center[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(self._window_name, img_to_show)
            cv2.waitKey(1)

        return tags

    def get_drone_pose(
        self, tag: Detection
    ) -> tuple[npt.NDArray[np.float64], R, npt.NDArray[np.float64]]:
        """
        Get the pose (position and rotation) of the drone in the global frame.

        These are defined in the global frame, which is north-east-down.

        Args:
            The detection object for the tag

        Returns:
            The (x, y, z) position of the drone in the global frame
            The rotation of the drone camera measured in the global frame
            The roll, pitch, and yaw of the drone in radians.
        """
        # Do some spatial algebra
        # rotation of tag measured in camera frame
        R_cam_tag = R.from_matrix(tag.pose_R)
        # Rotation of camera measured in tag frame
        R_tag_cam = R_cam_tag.inv()
        # Vector from camera to tag in camera frame
        p_cam_tag = tag.pose_t.reshape(-1)
        # Vector from tag to camera in camera frame
        p_tag_cam_cam = -p_cam_tag
        # Vector from tag to camera in tag frame
        p_tag_cam_tag = R_tag_cam.apply(p_tag_cam_cam)

        # We want to express the position and orientation of the drone in the
        # global frame, which is defined as north-east-down (NED).
        R_drone_cam = R.from_matrix(self.R_drone_camera)
        R_global_tag = R_drone_cam  # NED->tag is same as drone->cam
        R_cam_drone = R_drone_cam.inv()
        R_global_drone = R_global_tag * R_tag_cam * R_cam_drone
        p_global_cam_global = R_global_tag.apply(p_tag_cam_tag)
        # no translation from from camera to drone
        p_global_drone_global = p_global_cam_global

        # Convert to euler angles
        euler_angles = R_global_drone.as_euler("XYZ")

        return p_global_drone_global, R_global_drone, euler_angles

    @property
    def R_drone_camera(self) -> npt.NDArray[np.float64]:
        """
        Rotation matrix of the camera frame measured in the drone body frame.

        Drone body frame is North-East-Down.
        Camera frame is standard, with x to the right, y down, and z forward.
        """
        return np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )

    def convert_to_drone_frame(
        self, xyz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Convert a vector from the camera frame to the drone body frame.

        Args:
            xyz: A vector in the camera frame.

        Returns:
            The vector in the drone body frame.
        """
        return self.R_drone_camera @ xyz
