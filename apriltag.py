import os
import cv2
from pyapriltags import Detector
import numpy as np
import time
from ntcore import NetworkTableInstance
import socket
import struct

try:
    from picamera2 import Picamera2
except ImportError:
    # mock for testing on my mac or any non pi systems
    class Picamera2:
        def configure(self, config):
            pass
        def start(self):
            print("[Mock] Camera started")
        def capture_array(self):
            import numpy as np
            return np.zeros((480, 640, 3), dtype=np.uint8)  # blank frame

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_params.npz')
CAMERA_TILT_DEG = 40.0  # Camera tilts 40° downward from horizontal
CAMERA_TILT_RAD = np.radians(CAMERA_TILT_DEG)

# Camera offset from robot center in robot frame (metres)
# +x = robot forward (toward front/intake), +y = robot left
# From measurements: camera is 3.25in forward, 2.5in left of center
#CAMERA_OFFSET_X = -0.08255   # 3.25 inches behind robot center (-x in robot frame)
#CAMERA_OFFSET_Y = -0.0635    # 2.5 inches right of robot center (-y in robot frame)

#Camera extrinsics, camera pose relative to robot frame
#Robot frame +x = forward, +y = left, +z = up
CAMERA_EXT_X = -0.1125 #METERS BEHIND ROBOT CENTER
CAMERA_EXT_Y = -0.0910 #METERS RIGHT OF ROBOT CENTER
CAMERA_EXT_Z = 0.2230 #METERS ABOVE ROBOT ORIGIN

CAMERA_EXT_YAW = 180.0 #CAMERA FACES BACKWARD
CAMERA_EXT_PITCH = -40.0 #CAMERA TILTED DOWNWARD
CAMERA_EXT_ROLL = 0.0

display = False

# ── Known AprilTag field positions ────────────────────────────────────────────
# Format: tag_id → (field_x, field_y, facing_angle_rad)
# facing_angle = direction the tag's front normal points in field coordinates
# Field origin: bottom-right corner, +x toward cave, +y toward left wall
TAG_POSITIONS = {
    # West wall (back wall, x≈0) — IDs 0-4 indicate bucket deposit zone
    # All at same position, tag faces +x into the field
    0: (0.0090932, 0.5716016, 0.0),
    1: (0.0090932, 0.5716016, 0.0),
    2: (0.0090932, 0.5716016, 0.0),
    3: (0.0090932, 0.5716016, 0.0),
    4: (0.0090932, 0.5716016, 0.0),
    # North wall (left wall, y≈1.22) — tag faces -y into the field
    5: (0.8120126, 1.1405108, -np.pi / 2),
    # South wall (right wall, y=0) — tag faces +y into the field
    6: (1.116711, 0.0024892, np.pi / 2),
    # East wall (cave side, x≈2.44) — tag faces -x into the field
    7: (2.3597108, 0.5716016, np.pi),
}


def calibrate(picam2, board_size=(8, 6), square_size=0.025, min_frames=15, capture_interval=4.0):
    """
    Calibrate camera from live Picamera2 feed with a printed chessboard.

    Hold the chessboard in front of the camera at different angles/distances.
    The function captures a frame every `capture_interval` seconds, checks for
    the chessboard, and collects until `min_frames` good detections are gathered.
    Press 'q' to stop early (if at least 5 frames collected).

    Args:
        picam2:           Already-started Picamera2 instance.
        board_size:       Inner corners of the chessboard (cols, rows).
        square_size:      Size of one square in your chosen unit (e.g. mm or cm).
        min_frames:       Number of good chessboard frames to collect.
        capture_interval: Seconds between capture attempts (gives you time to move the board).

    Returns:
        (camMatrix, distCoeff) on success, or None on failure.
    """
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    world_pts = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    world_pts[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    world_pts *= square_size

    world_pts_list = []
    img_pts_list = []
    used_count = 0

    print(f"[calibrate] Hold chessboard ({board_size[0]}x{board_size[1]}) in front of camera.")
    print(f"[calibrate] Capturing every {capture_interval}s. Need {min_frames} good frames. Press 'q' to finish early.")

    last_capture = 0.0

    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        now = time.time()
        status_text = f"Collected: {used_count}/{min_frames}"

        if now - last_capture >= capture_interval:
            last_capture = now
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save first attempt for debugging
            if used_count == 0 and not hasattr(calibrate, '_saved_debug'):
                cv2.imwrite('calibration_debug.png', frame_gray)
                print(f"[calibrate] Saved debug frame to calibration_debug.png ({frame_gray.shape})")
                calibrate._saved_debug = True

            found, corners = cv2.findChessboardCorners(
                frame_gray, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )

            if found:
                corners_refined = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), term_criteria)
                world_pts_list.append(world_pts)
                img_pts_list.append(corners_refined)
                used_count += 1
                print(f"  chessboard found ({used_count}/{min_frames})")
                cv2.drawChessboardCorners(frame, board_size, corners_refined, found)
                status_text = f"CAPTURED {used_count}/{min_frames}"
            else:
                status_text = f"No board detected ({used_count}/{min_frames})"

            if used_count >= min_frames:
                break

        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Calibration")

    if used_count < 5:
        print(f"[calibrate] ERROR: only collected {used_count} frames (need at least 5)")
        return None

    print(f"[calibrate] Running calibration with {used_count} frames...")
    rep_error, cam_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        world_pts_list, img_pts_list, frame_gray.shape[::-1], None, None
    )

    print(f"  Camera matrix:\n{cam_matrix}")
    print(f"  Reprojection error: {rep_error:.4f} pixels")

    np.savez(CALIBRATION_FILE,
             repError=rep_error, camMatrix=cam_matrix, distCoeff=dist_coeff,
             rvecs=rvecs, tvecs=tvecs)
    print(f"  Saved to {CALIBRATION_FILE}")

    return cam_matrix, dist_coeff


def load_calibration():
    """Load calibration from file. Returns (camMatrix, distCoeff) or None."""
    if not os.path.exists(CALIBRATION_FILE):
        return None
    data = np.load(CALIBRATION_FILE)
    print(f"[calibration] Loaded from {CALIBRATION_FILE} (reprojection error: {data['repError']:.4f}px)")
    return data['camMatrix'], data['distCoeff']


def undistort_frame(frame, cam_matrix, dist_coeff):
    """Remove lens distortion from a frame and return the matching camera matrix."""
    h, w = frame.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cam_matrix, dist_coeff, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(frame, cam_matrix, dist_coeff, None, new_matrix)
    return undistorted, new_matrix
    
def listen():
    """
    Debug/diagnostic tool only - run this manually during testing.
    Listens on port 1150 for roboRIO reply packets to confirm
    packets are reaching the roboRIO.
    """
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("0.0.0.0", 1150))
    recv_sock.settimeout(1.0)
    while True:
        try:
            data, addr = recv_sock.recvfrom(1024)
            print(f" << roboRIO reply: {data.hex()}")
        except socket.timeout:
            pass

def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [1,  0,   0],
        [0, ca, -sa],
        [0, sa,  ca]
    ], dtype=float)

def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ ca, 0, sa],
        [  0, 1,  0],
        [-sa, 0, ca]
    ], dtype=float)

def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [ 0,   0, 1]
    ], dtype=float)

def make_transform(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

def get_robot_T_camera():
    #returns T_robot_camera which tranforms points from camera frame into robot frame
    yaw = np.radians(CAMERA_EXT_YAW)
    pitch = np.radians(CAMERA_EXT_PITCH)
    roll = np.radians(CAMERA_EXT_ROLL)
    #Rotation from camera frame to robot frame, zyx order, yaw, pitch then roll
    R_robot_camera = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    t_robot_camera = np.array([CAMERA_EXT_X, CAMERA_EXT_Y, CAMERA_EXT_Z], dtype=float)
    return make_transform(R_robot_camera, t_robot_camera)

class DSPacketSender:
    """
    FRC Driver Station UDP packet sender.
    Sends enable/disable commands to the roboRIO.
    """

    def __init__(self, roborio_ip):
        self.roborio_ip = roborio_ip
        self.roborio_port = 1110
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence = 0

        self.AUTONOMOUS_DISABLED = 0x02
        self.AUTONOMOUS_ENABLED  = 0x06
        self.ALLIANCE_STATION    = 0x00  # Red 1

        print("DS Packet Sender initialized")

    def create_ds_packet(self, enabled):
        control = self.AUTONOMOUS_ENABLED if enabled else self.AUTONOMOUS_DISABLED
        packet = bytearray(6)
        struct.pack_into('>H', packet, 0, self.sequence & 0xFFFF)
        packet[2] = 0x01
        packet[3] = control
        packet[4] = 0x00
        packet[5] = self.ALLIANCE_STATION
        return packet

    def enable_robot(self, autonomous=True):
        packet = self.create_ds_packet(enabled=True)
        self.sock.sendto(packet, (self.roborio_ip, self.roborio_port))
        self.sequence = (self.sequence + 1) & 0xFFFF
        print(f"Sent ENABLE command (autonomous={autonomous})")

    def disable_robot(self):
        packet = self.create_ds_packet(enabled=False)
        self.sock.sendto(packet, (self.roborio_ip, self.roborio_port))
        self.sequence = (self.sequence + 1) & 0xFFFF
        print("Sent DISABLE command")

    def send_keepalive(self, enabled=False):
        """Send keepalive packet - call at ~50Hz from main loop."""
        packet = self.create_ds_packet(enabled=enabled)
        self.sock.sendto(packet, (self.roborio_ip, self.roborio_port))
        self.sequence = (self.sequence + 1) & 0xFFFF

    def close(self):
        self.sock.close()


class AprilTagDetector:
    def __init__(self, tag_family='tag36h11', camera_params=None, roborio_ip='10.0.67.2'):
        """
        Initialize AprilTag detector for Raspberry Pi with IMX296 camera.

        Args:
            tag_family:    AprilTag family (default: 'tag36h11')
            camera_params: Camera calibration parameters [fx, fy, cx, cy]
            roborio_ip:    IP address of the roboRIO
            display:       True to configure for display mode, False for headless max-fps
        """
        self.task_done_sub = None
        self.display = display

        self.detector = Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # Camera parameters for IMX296 after 90° CCW rotation.
        #
        # The physical sensor captures 1480 (W) × 1110 (H).
        # After cv2.ROTATE_90_COUNTERCLOCKWISE the frame becomes 1110 (W) × 1480 (H).
        # The principal point axes swap accordingly:
        #   cx (horizontal centre) = 1110 / 2 = 555
        #   cy (vertical centre)   = 1480 / 2 = 740
        # fx and fy are swapped to match the new axis assignment.
        # These are uncalibrated estimates — proper camera calibration will
        # improve pose accuracy significantly.
        #
        # pose_t from pyapriltags is in the rotated camera's coordinate frame:
        # Raw pose_t from pyapriltags is in the tilted camera's frame.
        # We apply a pitch correction of +40° (CAMERA_TILT_DEG) before publishing
        # so the NT values are in the robot's horizontal frame:
        #   pose_tx = lateral offset (+ = right in image, unchanged by tilt)
        #   pose_ty = vertical offset (+ = below robot horizontal plane)
        #   pose_tz = horizontal depth (+ = away from camera, along ground plane)
        # The camera is rear-mounted, so the RIO must combine these with the
        # robot's current heading (theta) to convert into field-relative coords.
        if camera_params is None:
            # Focal length calibrated from known-distance measurement:
            # tag 0 at 0.715m actual → 0.383m reported with fx=500 → scale 1.867
            self.fx = 933  # Focal length along new X axis (was Y on raw sensor)
            self.fy = 933  # Focal length along new Y axis (was X on raw sensor)
            self.cx = 555  # Principal point x: half of rotated frame width  (1110/2)
            self.cy = 740  # Principal point y: half of rotated frame height (1480/2)
        else:
            self.fx, self.fy, self.cx, self.cy = camera_params

        self.setup_networktables(roborio_ip)

        self.ds_sender = DSPacketSender(roborio_ip)
        self.robot_enabled = False
        self.last_keepalive_time = time.time()

        self.picam2 = Picamera2()

        if display:
            # Preview config - includes ISP pipeline for display quality
            config = self.picam2.create_preview_configuration(
                main={"size": (1480, 1110), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
        else:
            # Still config - lower ISP overhead, better throughput for headless
            config = self.picam2.create_still_configuration(
                main={"size": (1480, 1110), "format": "RGB888"}
            )

        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warm-up


    def setup_networktables(self, roborio_ip):
        self.nt_inst = NetworkTableInstance.getDefault()
        self.nt_inst.startClient4("apriltag_detector")

        if isinstance(roborio_ip, int):
            self.nt_inst.setServerTeam(roborio_ip)
        else:
            self.nt_inst.setServer(roborio_ip)

        self.vision_table = self.nt_inst.getTable("Vision")
        self.task_done_sub = self.vision_table.getBooleanTopic("task_done").subscribe(False)

        # Single tag (primary)
        self.tag_detected_entry  = self.vision_table.getBooleanTopic("tag_detected").publish()
        self.tag_id_entry        = self.vision_table.getIntegerTopic("tag_id").publish()
        self.tag_x_entry         = self.vision_table.getDoubleTopic("tag_x").publish()
        self.tag_y_entry         = self.vision_table.getDoubleTopic("tag_y").publish()
        self.tag_distance_entry  = self.vision_table.getDoubleTopic("tag_distance").publish()
        self.tag_count_entry     = self.vision_table.getIntegerTopic("tag_count").publish()

        # Primary tag pose (camera-relative, metres)
        self.tag_pose_tx_entry       = self.vision_table.getDoubleTopic("tag_pose_tx").publish()
        self.tag_pose_ty_entry       = self.vision_table.getDoubleTopic("tag_pose_ty").publish()
        self.tag_pose_tz_entry       = self.vision_table.getDoubleTopic("tag_pose_tz").publish()
        self.tag_pose_err_entry      = self.vision_table.getDoubleTopic("tag_pose_err").publish()
        self.tag_decision_margin_entry = self.vision_table.getDoubleTopic("tag_decision_margin").publish()

        # All tags as arrays
        self.tags_ids_entry       = self.vision_table.getIntegerArrayTopic("tags_ids").publish()
        self.tags_x_entry         = self.vision_table.getDoubleArrayTopic("tags_x").publish()
        self.tags_y_entry         = self.vision_table.getDoubleArrayTopic("tags_y").publish()
        self.tags_distances_entry = self.vision_table.getDoubleArrayTopic("tags_distances").publish()
        self.tags_pose_tx_entry   = self.vision_table.getDoubleArrayTopic("tags_pose_tx").publish()
        self.tags_pose_ty_entry   = self.vision_table.getDoubleArrayTopic("tags_pose_ty").publish()
        self.tags_pose_tz_entry   = self.vision_table.getDoubleArrayTopic("tags_pose_tz").publish()

        # Capture timestamp in RIO FPGA time (microseconds), converted via NT4 clock sync
        self.tag_timestamp_entry = self.vision_table.getIntegerTopic("tag_timestamp_us").publish()

        # Field-relative robot pose (computed from AprilTag known positions + pose)
        self.robot_field_x_entry     = self.vision_table.getDoubleTopic("robot_field_x").publish()
        self.robot_field_y_entry     = self.vision_table.getDoubleTopic("robot_field_y").publish()
        self.robot_field_theta_entry = self.vision_table.getDoubleTopic("robot_field_theta").publish()

        self.heartbeat_entry   = self.vision_table.getIntegerTopic("heartbeat").publish()
        self.heartbeat_counter = 0
        self.start_light_entry = self.vision_table.getBooleanTopic("start_light_detected").publish()

        print(f"NetworkTables initialized, connecting to roboRIO at {roborio_ip}")

    def capture_rio_timestamp(self):
        """
        Record a timestamp at frame capture and convert to RIO FPGA time
        using NT4's built-in clock synchronization.

        Returns RIO-relative timestamp in microseconds, or -1 if not synced yet.
        """
        local_time_us = time.monotonic_ns() // 1000
        offset = self.nt_inst.getServerTimeOffset()
        if offset is None:
            return -1
        return local_time_us + offset

    @staticmethod
    def correct_pose_tilt(pose_t):
        """
        Rotate pose_t from the tilted camera frame to the robot's horizontal frame.
        Applies a pitch correction of +CAMERA_TILT_DEG around the camera X axis.

        Returns corrected (tx, ty, tz) tuple in metres.
        """
        tx = float(pose_t[0])  # lateral — unaffected by pitch
        ty = float(pose_t[1])
        tz = float(pose_t[2])
        cos_a = np.cos(CAMERA_TILT_RAD)
        sin_a = np.sin(CAMERA_TILT_RAD)
        ty_corrected = ty * cos_a - tz * sin_a
        tz_corrected = ty * sin_a + tz * cos_a
        return tx, ty_corrected, tz_corrected

    @staticmethod
    def compute_field_pose(tag, debug=False):
        """
        Compute the robot's field-relative position and heading from a single
        AprilTag detection using the full pose_R rotation matrix.

        pyapriltags tag coordinate convention:
          Tag +X = right (when looking at the tag face)
          Tag +Y = down
          Tag +Z = into the tag / into the wall (away from camera)

        For a tag on a wall with facing_angle φ (direction tag front faces,
        i.e. the outward normal = -Z_tag in field coords):
          Tag -Z in field = (cos φ, sin φ, 0)  → outward normal into the field
          Tag +Z in field = (-cos φ, -sin φ, 0) → into the wall
          Tag +Y in field = (0, 0, -1)          → down (gravity)
          Tag +X in field = (-sin φ, cos φ, 0)  → right when facing the tag

        Returns (robot_x, robot_y, robot_theta) or None if tag ID is unknown.
        """
        tag_info = TAG_POSITIONS.get(tag.tag_id)
        if tag_info is None or tag.pose_R is None or tag.pose_t is None:
            return None

        tag_x, tag_y, facing_angle = tag_info

        # R_tag_to_field columns are tag +X, +Y, +Z expressed in field coords.
        cos_f = np.cos(facing_angle)
        sin_f = np.sin(facing_angle)
        R_field_tag = np.array([
            [-sin_f,  0, -cos_f],
            [ cos_f,  0, -sin_f],
            [ 0,     -1,  0    ]
        ], dtype=float)

        # pose_R rotates from tag frame → camera frame.
        # pose_R.T rotates from camera frame → tag frame.
        # R_tag_to_field @ pose_R.T = camera frame → field frame.
        t_field_tag = np.array([tag_x, tag_y, 0.0], dtype=float)

        T_field_tag = make_transform(R_field_tag, t_field_tag)

        # pose_t is the tag position in camera coordinates.
        # Transform to field coordinates:
        R_camera_tag = tag.pose_R
        t_camera_tag = tag.pose_t.flatten()

        T_camera_tag = make_transform(R_camera_tag, t_camera_tag)

        #invert to get camera pose in tag frame
        T_tag_camera = np.linalg.inv(T_camera_tag)

        #camera pose in field frame
        T_field_camera = T_field_tag @ T_tag_camera

        #fixed robot pose relative to camera
        T_robot_camera = get_robot_T_camera()

        #robot pose in field frame
        T_field_robot = T_field_camera @ np.linalg.inv(T_robot_camera)

        cam_x = T_field_camera[0, 3]
        cam_y = T_field_camera[1, 3]

        cam_forward = T_field_camera[:3, 2]
        camera_heading = np.arctan2(cam_forward[1], cam_forward[0])
        robot_theta = camera_heading + np.pi
        robot_theta = np.arctan2(np.sin(robot_theta), np.cos(robot_theta))

        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)

        robot_x = cam_x - (CAMERA_EXT_X * cos_t - CAMERA_EXT_Y * sin_t)
        robot_y = cam_y - (CAMERA_EXT_X * sin_t + CAMERA_EXT_Y * cos_t)

        if debug:
            print(f"  [debug] T_field_tag:\n{T_field_tag}")
            print(f"  [debug] T_camera_tag:\n{T_camera_tag}")
            print(f"  [debug] T_tag_camera:\n{T_tag_camera}")
            print(f"  [debug] T_field_camera:\n{T_field_camera}")
            print(f"  [debug] T_robot_camera:\n{T_robot_camera}")
            print(f"  [debug] T_field_robot:\n{T_field_robot}")
            print(f"  [debug] camera pos: ({cam_x:.4f}, {cam_y:.4f})")
            print(f"  [debug] robot pos: ({robot_x:.4f}, {robot_y:.4f}, {robot_theta:.4f})")

        return robot_x, robot_y, robot_theta

    def publish_detections(self, tags, capture_timestamp_us):
        self.heartbeat_counter += 1
        self.heartbeat_entry.set(self.heartbeat_counter)
        self.tag_count_entry.set(len(tags))
        self.tag_timestamp_entry.set(capture_timestamp_us)

        if tags:
            primary = tags[0]
            self.tag_detected_entry.set(True)
            self.tag_id_entry.set(int(primary.tag_id))
            self.tag_x_entry.set(float(primary.center[0]))
            self.tag_y_entry.set(float(primary.center[1]))
            dist = float(np.linalg.norm(primary.pose_t)) if primary.pose_t is not None else -1.0
            self.tag_distance_entry.set(dist)
            self.tag_decision_margin_entry.set(float(primary.decision_margin))

            if primary.pose_t is not None:
                tx, ty, tz = self.correct_pose_tilt(primary.pose_t)
                self.tag_pose_tx_entry.set(tx)
                self.tag_pose_ty_entry.set(ty)
                self.tag_pose_tz_entry.set(tz)
                self.tag_pose_err_entry.set(float(primary.pose_err))
            else:
                self.tag_pose_tx_entry.set(0.0)
                self.tag_pose_ty_entry.set(0.0)
                self.tag_pose_tz_entry.set(0.0)
                self.tag_pose_err_entry.set(-1.0)

            self.tags_ids_entry.set([int(t.tag_id) for t in tags])
            self.tags_x_entry.set([float(t.center[0]) for t in tags])
            self.tags_y_entry.set([float(t.center[1]) for t in tags])
            self.tags_distances_entry.set([
                float(np.linalg.norm(t.pose_t)) if t.pose_t is not None else -1.0
                for t in tags
            ])
            corrected = [self.correct_pose_tilt(t.pose_t) if t.pose_t is not None else (0.0, 0.0, 0.0) for t in tags]
            self.tags_pose_tx_entry.set([c[0] for c in corrected])
            self.tags_pose_ty_entry.set([c[1] for c in corrected])
            self.tags_pose_tz_entry.set([c[2] for c in corrected])

            # Field-relative pose from the primary tag
            field_pose = self.compute_field_pose(primary)
            if field_pose is not None:
                self.robot_field_x_entry.set(field_pose[0])
                self.robot_field_y_entry.set(field_pose[1])
                self.robot_field_theta_entry.set(field_pose[2])
        else:
            self.tag_detected_entry.set(False)
            self.tag_id_entry.set(-1)
            self.tag_x_entry.set(0.0)
            self.tag_y_entry.set(0.0)
            self.tag_distance_entry.set(-1.0)
            self.tag_pose_tx_entry.set(0.0)
            self.tag_pose_ty_entry.set(0.0)
            self.tag_pose_tz_entry.set(0.0)
            self.tag_pose_err_entry.set(-1.0)
            self.tag_decision_margin_entry.set(0.0)
            self.tags_ids_entry.set([])
            self.tags_x_entry.set([])
            self.tags_y_entry.set([])
            self.tags_distances_entry.set([])
            self.tags_pose_tx_entry.set([])
            self.tags_pose_ty_entry.set([])
            self.tags_pose_tz_entry.set([])

    def send_ds_keepalive(self):
        current_time = time.time()
        if current_time - self.last_keepalive_time >= 0.02:  # 50Hz
            self.ds_sender.send_keepalive(enabled=self.robot_enabled)
            self.last_keepalive_time = current_time

    def detect_tags(self, image, camera_params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=0.08  # Tag size in meters - adjust to your actual tag size
        )

    def draw_detection(self, image, tag):
        """Draw a 3D cube on the tag to visualize its pose. Only called in display mode."""
        if tag.pose_R is None or tag.pose_t is None:
            return

        half = 0.1 / 2  # half of tag_size (0.1m)
        cube_height = 0.1  # cube extends 10cm out from the tag face

        # 8 corners of a cube: bottom face sits on the tag, top face extends toward camera
        cube_pts = np.float32([
            [-half, -half, 0],  [half, -half, 0],
            [half,  half, 0],   [-half,  half, 0],
            [-half, -half, -cube_height], [half, -half, -cube_height],
            [half,  half, -cube_height],  [-half,  half, -cube_height],
        ])

        cam_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        rvec, _ = cv2.Rodrigues(tag.pose_R)
        tvec = tag.pose_t.reshape(3, 1)

        img_pts, _ = cv2.projectPoints(cube_pts, rvec, tvec, cam_matrix, None)
        pts = img_pts.reshape(-1, 2).astype(int)

        # Draw bottom face (green, on the tag)
        cv2.drawContours(image, [pts[:4]], -1, (0, 255, 0), 2)
        # Draw top face (red, floating above)
        cv2.drawContours(image, [pts[4:]], -1, (0, 0, 255), 2)
        # Draw vertical pillars (blue)
        for i in range(4):
            cv2.line(image, tuple(pts[i]), tuple(pts[i + 4]), (255, 0, 0), 2)

        # Tag ID and distance text
        center = tuple(tag.center.astype(int))
        cv2.putText(image, f"ID: {tag.tag_id}",
                    (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        distance = np.linalg.norm(tag.pose_t)
        cv2.putText(image, f"Dist: {distance:.2f}m",
                    (center[0] - 20, center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def run(self, cam_matrix=None, dist_coeff=None):
        """
        Run the detection loop.

        Args:
            cam_matrix: Camera matrix from calibration (None to skip undistortion).
            dist_coeff: Distortion coefficients from calibration (None to skip undistortion).
        """
        # FPS tracking - display mode only
        fps = 0
        fps_counter = 0
        fps_start_time = time.time()

        use_undistort = cam_matrix is not None and dist_coeff is not None
        if use_undistort:
            print("Lens distortion correction: ENABLED")
        else:
            print("Lens distortion correction: DISABLED (no calibration data)")

        print("Starting AprilTag detection...")
        print(f"NetworkTables connected: {self.nt_inst.isConnected()}")
        print(f"Mode: {'display' if self.display else 'headless'}")

        try:
            time.sleep(2)
            self.ds_sender.enable_robot(autonomous=True)
            self.robot_enabled = True

            # ── Main detection loop ───────────────────────────────────────
            while True:
                if self.task_done_sub.get():
                    print("RIO signaled task done - stopping DS keepalive")
                    self.ds_sender.disable_robot()
                    self.robot_enabled = False
                    break

                frame = self.picam2.capture_array()
                capture_timestamp_us = self.capture_rio_timestamp()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if use_undistort:
                    frame, current_matrix = undistort_frame(frame, cam_matrix, dist_coeff)
                    current_camera_params = [
                        current_matrix[0, 0],
                        current_matrix[1, 1],
                        current_matrix[0, 2],
                        current_matrix[1, 2],
                    ]
                else:
                    current_camera_params = [self.fx, self.fy, self.cx, self.cy]

                self.send_ds_keepalive()

                tags = self.detect_tags(frame, current_camera_params)
                self.publish_detections(tags, capture_timestamp_us)

                if tags:
                    for tag in tags:
                        if tag.pose_t is not None:
                            do_debug = (self.heartbeat_counter % 100 == 1)
                            field = self.compute_field_pose(tag, debug=do_debug)
                            dist = np.linalg.norm(tag.pose_t)
                            pt = tag.pose_t.flatten()
                            if field is not None:
                                fx, fy, ft = field
                                print(f"[NT] id={tag.tag_id}  "
                                      f"field=({fx:.3f}, {fy:.3f}, θ={ft:.3f}rad)  "
                                      f"dist={dist:.3f}m  "
                                      f"raw_t=({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})  "
                                      f"err={tag.pose_err:.4f}  "
                                      f"margin={tag.decision_margin:.1f}")
                            else:
                                print(f"[NT] id={tag.tag_id}  "
                                      f"field=UNKNOWN_TAG  dist={dist:.3f}m  "
                                      f"raw_t=({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
                        else:
                            print(f"[NT] id={tag.tag_id}  "
                                  f"px=({tag.center[0]:.1f}, {tag.center[1]:.1f})  "
                                  f"pose=N/A")
                elif self.heartbeat_counter % 50 == 0:
                    print(f"[NT] no tags | heartbeat={self.heartbeat_counter}")

                if self.display:
                    for tag in tags:
                        self.draw_detection(frame, tag)

                    # FPS overlay
                    fps_counter += 1
                    if fps_counter >= 30:
                        fps = fps_counter / (time.time() - fps_start_time)
                        fps_counter = 0
                        fps_start_time = time.time()

                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    nt_status = "NT: Connected" if self.nt_inst.isConnected() else "NT: Disconnected"
                    nt_color  = (0, 255, 0) if self.nt_inst.isConnected() else (0, 0, 255)
                    cv2.putText(frame, nt_status, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, nt_color, 2)

                    robot_status = "ROBOT: ENABLED" if self.robot_enabled else "ROBOT: DISABLED"
                    robot_color  = (0, 255, 0) if self.robot_enabled else (0, 0, 255)
                    cv2.putText(frame, robot_status, (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, robot_color, 2)

                    cv2.imshow('AprilTag Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nStopping detection...")

        finally:
            if self.robot_enabled:
                print("Disabling robot...")
                self.ds_sender.disable_robot()
                self.robot_enabled = False

    def cleanup(self):
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.nt_inst.stopClient()
        self.ds_sender.close()
        print("Cleanup complete")


def monitor_rio(roborio_ip="10.0.67.2", interval=0.25):
    """
    Subscribe to SmartDashboard/Test/* from the RIO and print a live table.
    Run with: python apriltag.py --monitor
    """

    inst = NetworkTableInstance.getDefault()
    inst.setServer(roborio_ip)
    inst.startClient4("pi-monitor")

    sd = inst.getTable("SmartDashboard")

    # All the Test/ keys the RIO publishes (grouped for readability)
    bool_keys = [
        "Test/Vision Connected", "Test/Has Target", "Test/Field Pose Valid",
    ]
    num_keys = [
        "Test/Tag Count",
        "Test/Primary ID", "Test/Primary Pixel X", "Test/Primary Pixel Y",
        "Test/Primary Distance (m)",
        "Test/Pose TX (m)", "Test/Pose TY (m)", "Test/Pose TZ (m)",
        "Test/Pose Error", "Test/Decision Margin", "Test/Timestamp us",
        "Test/Field X (m)", "Test/Field Y (m)",
        "Test/Field Theta (rad)", "Test/Field Theta (deg)",
        "Test/Odom X (m)", "Test/Odom Y (m)", "Test/Odom Theta (rad)",
        "Test/Delta X (m)", "Test/Delta Y (m)", "Test/Delta Theta (rad)",
        "Test/IMU Theta (rad)", "Test/IMU Theta (deg)",
        "Test/Vision Latency (ms)",
    ]
    str_keys = [
        "Auto/Phase", "Test/Visible Tags",
    ]

    bool_subs = {k: sd.getBooleanTopic(k).subscribe(False) for k in bool_keys}
    num_subs  = {k: sd.getDoubleTopic(k).subscribe(float('nan')) for k in num_keys}
    str_subs  = {k: sd.getStringTopic(k).subscribe("") for k in str_keys}

    # Pre-create per-tag subscribers so they persist and receive updates
    max_tags = 8
    tag_subs = {}
    for i in range(max_tags):
        prefix = f"Test/Tag[{i}]/"
        tag_subs[i] = {
            "ID":            sd.getDoubleTopic(prefix + "ID").subscribe(float('nan')),
            "Distance (m)":  sd.getDoubleTopic(prefix + "Distance (m)").subscribe(0),
            "Pose TX (m)":   sd.getDoubleTopic(prefix + "Pose TX (m)").subscribe(0),
            "Pose TY (m)":   sd.getDoubleTopic(prefix + "Pose TY (m)").subscribe(0),
            "Pose TZ (m)":   sd.getDoubleTopic(prefix + "Pose TZ (m)").subscribe(0),
        }

    print(f"[monitor] Connecting to RIO at {roborio_ip}  (Ctrl-C to quit)")
    print(f"[monitor] Refreshing every {interval}s\n")

    try:
        while True:
            # Let NT process incoming data before we read
            inst.flush()
            time.sleep(interval)

            connected = inst.isConnected()
            # Clear screen and print header
            print("\033[2J\033[H", end="")  # ANSI clear + home
            print(f"{'═' * 60}")
            print(f"  RIO TEST MONITOR   NT4: {'CONNECTED' if connected else 'DISCONNECTED'}")
            print(f"{'═' * 60}")

            if not connected:
                print("\n  Waiting for NetworkTables connection...")
                continue

            # Strings
            for k, sub in str_subs.items():
                label = k.replace("Test/", "").replace("Auto/", "")
                print(f"  {label:30s}  {sub.get()}")

            # Booleans
            for k, sub in bool_subs.items():
                label = k.replace("Test/", "")
                val = sub.get()
                icon = "✓" if val else "✗"
                print(f"  {label:30s}  {icon}  ({val})")

            print(f"{'─' * 60}")
            print(f"  {'KEY':30s}  {'VALUE':>12s}")
            print(f"{'─' * 60}")

            # Numbers
            for k, sub in num_subs.items():
                label = k.replace("Test/", "")
                val = sub.get()
                if val != val:  # NaN check — key not published yet
                    print(f"  {label:30s}  {'---':>12s}")
                elif "ID" in label or "Count" in label:
                    print(f"  {label:30s}  {int(val):>12d}")
                elif "Pixel" in label or "us" in label.lower():
                    print(f"  {label:30s}  {val:>12.0f}")
                else:
                    print(f"  {label:30s}  {val:>12.4f}")

            # Per-tag array (persistent subscribers)
            print(f"{'─' * 60}")
            for i in range(max_tags):
                tag_id = tag_subs[i]["ID"].get()
                if tag_id != tag_id:  # NaN — not published
                    break
                dist = tag_subs[i]["Distance (m)"].get()
                tx   = tag_subs[i]["Pose TX (m)"].get()
                ty   = tag_subs[i]["Pose TY (m)"].get()
                tz   = tag_subs[i]["Pose TZ (m)"].get()
                print(f"  Tag[{i}]  ID={int(tag_id):2d}  dist={dist:.3f}m  "
                      f"tx={tx:.3f}  ty={ty:.3f}  tz={tz:.3f}")

            print(f"{'═' * 60}")

    except KeyboardInterrupt:
        print("\n[monitor] Stopped.")
    finally:
        inst.stopClient()


if __name__ == "__main__":
    import sys

    # --- Monitor mode: python apriltag.py --monitor [rio_ip] ---
    if len(sys.argv) >= 2 and sys.argv[1] == '--monitor':
        rio_ip = sys.argv[2] if len(sys.argv) >= 3 else "10.0.67.2"
        monitor_rio(roborio_ip=rio_ip)
        sys.exit(0)

    # --- Calibration mode: python apriltag.py --calibrate [cols rows] ---
    if len(sys.argv) >= 2 and sys.argv[1] == '--calibrate':
        board = (9, 6)  # default chessboard inner corners
        if len(sys.argv) >= 4:
            board = (int(sys.argv[2]), int(sys.argv[3]))

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (1480, 1110), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        result = calibrate(picam2, board_size=board, square_size=0.025)
        picam2.stop()
        if result is None:
            print("Calibration failed.")
            sys.exit(1)
        print("Calibration complete.")
        sys.exit(0)

    # --- Detection mode ---
    calib = load_calibration()
    cam_matrix = None
    dist_coeff = None
    camera_params = None

    if calib is not None:
        cam_matrix, dist_coeff = calib
        # Extract fx, fy, cx, cy from calibrated camera matrix for AprilTag pose estimation
        camera_params = [cam_matrix[0, 0], cam_matrix[1, 1], cam_matrix[0, 2], cam_matrix[1, 2]]
    else:
        print("WARNING: No calibration file found. Running with estimated camera params and no undistortion.")

    detector = AprilTagDetector(
        tag_family='tag36h11',
        camera_params=camera_params,
    )

    try:
        detector.run(cam_matrix, dist_coeff)
    finally:
        detector.cleanup()

