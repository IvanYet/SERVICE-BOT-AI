"""ROS 2 monitor for receptionist identity, camera status, and Nav2."""

import json
import math
import threading
import time


class ReceptionistMonitor:
    """Keep recent person identity and system status from ROS topics."""

    def __init__(self, topic="/robot/vision/receptionist_data"):
        self.topic = topic
        self._lock = threading.Lock()
        self._received = threading.Event()
        self._description_received = threading.Event()
        self._vision_status_received = threading.Event()
        self._identity_status_received = threading.Event()
        self._people = None
        self._seats = None
        self._last_vision_message_at = 0.0
        self._vision_sequence = 0
        self._vision_status = {"camera_ready": False, "vlm_status": "unknown"}
        self._identity_status = {}
        self._description = None
        self._node = None
        self._executor = None
        self._thread = None
        self._rclpy = None
        self._navigate_to_pose = None
        self._spin = None
        self._camera_fx = None
        self._camera_cx = None

    def start(self):
        """Start a ROS 2 subscriber thread; return False when ROS is unavailable."""
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from std_msgs.msg import String
            from sensor_msgs.msg import CameraInfo
        except ImportError as exc:
            print(f"[VISION] ROS 2 Python packages unavailable: {exc}")
            return False

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self._rclpy = rclpy
            self._node = rclpy.create_node("robocup_edu_receptionist_monitor")
            self._node.create_subscription(String, self.topic, self._on_message, 10)
            self._node.create_subscription(
                CameraInfo, "/camera/camera/color/camera_info", self._on_camera_info, 10
            )
            self._node.create_subscription(
                String, "/robot/vision/description", self._on_description, 10
            )
            self._node.create_subscription(
                String, "/robot/vision/status", self._on_vision_status, 10
            )
            self._node.create_subscription(
                String, "/robot/vision/identity_status", self._on_identity_status, 10
            )
            self._identity_command_publisher = self._node.create_publisher(
                String, "/robot/vision/identity_command", 10
            )
            self._describe_publisher = self._node.create_publisher(
                String, "/robot/vision/describe_request", 10
            )
            from geometry_msgs.msg import PoseWithCovarianceStamped
            self._initial_pose_publisher = self._node.create_publisher(
                PoseWithCovarianceStamped, "/initialpose", 10
            )
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            self._navigate_to_pose = ActionClient(
                self._node, NavigateToPose, "/navigate_to_pose"
            )
            from nav2_msgs.action import Spin
            self._spin = ActionClient(self._node, Spin, "/spin")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()
            print(f"[VISION] Listening for guest identity data on {self.topic}")
            return True
        except Exception as exc:
            print(f"[VISION] Could not start the receptionist ROS monitor: {exc}")
            return False

    def set_initial_pose(self, x, y, yaw, timeout=2.0):
        """Publish the configured map pose for AMCL when Nav2 is listening."""
        publisher = getattr(self, "_initial_pose_publisher", None)
        if self._node is None or publisher is None:
            print("[NAV] ROS is unavailable; cannot publish the initial pose.")
            return False

        deadline = time.monotonic() + timeout
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        if publisher.get_subscription_count() == 0:
            print("[NAV] No /initialpose subscriber found; skipping the pose estimate.")
            return False

        from geometry_msgs.msg import PoseWithCovarianceStamped

        for _ in range(3):
            message = PoseWithCovarianceStamped()
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = "map"
            message.pose.pose.position.x = float(x)
            message.pose.pose.position.y = float(y)
            message.pose.pose.position.z = 0.0
            message.pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
            message.pose.pose.orientation.w = math.cos(float(yaw) / 2.0)
            message.pose.covariance[0] = 0.25
            message.pose.covariance[7] = 0.25
            message.pose.covariance[35] = 0.0685
            publisher.publish(message)
            time.sleep(0.2)

        print(f"[NAV] Published initial pose placeholder: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")
        return True

    def wait_for_navigation(self, timeout=30.0):
        """Wait until Nav2 exposes its NavigateToPose action server."""
        client = self._navigate_to_pose
        if client is None:
            print("[NAV] Navigation action client is unavailable.")
            return False
        ready = client.wait_for_server(timeout_sec=timeout)
        if not ready:
            print("[NAV] Timed out waiting for /navigate_to_pose.")
        return ready

    def wait_for_vision_ready(self, timeout=130.0):
        """Wait until the vision node reports a live camera and ready VLM."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._vision_status_received.wait(min(remaining, 0.5)):
                continue
            with self._lock:
                status = dict(self._vision_status)
            if status.get("camera_ready") and status.get("vlm_status") == "ready":
                return True
            if status.get("vlm_status") in ("unavailable", "disabled"):
                print(f"[VISION] Vision readiness failed: {status}")
                return False
            self._vision_status_received.clear()
        return False

    def wait_for_identity_ready(self, timeout=60.0):
        """Require a live camera and loaded OSNet model before starting guests."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._vision_status_received.wait(min(0.5, max(0.0, deadline - time.monotonic())))
            with self._lock:
                status = dict(self._vision_status)
            if status.get("camera_ready") and status.get("identity_model_ready"):
                return True
            self._vision_status_received.clear()
        print("[VISION] Camera or OSNet identity model did not become ready.")
        return False

    def navigate_to_pose(self, x, y, yaw, timeout=180.0):
        """Send one map-frame Nav2 goal and wait for its result."""
        if self._node is None or self._navigate_to_pose is None:
            print("[NAV] ROS navigation is unavailable.")
            return False
        if not self.wait_for_navigation(timeout=10.0):
            return False

        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose
        from action_msgs.msg import GoalStatus

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
        goal.pose.pose.orientation.w = math.cos(float(yaw) / 2.0)

        sent_event = threading.Event()
        sent_result = {}
        sent_future = self._navigate_to_pose.send_goal_async(goal)

        def on_goal_response(future):
            try:
                sent_result["handle"] = future.result()
            except Exception as exc:
                sent_result["error"] = exc
            finally:
                sent_event.set()

        sent_future.add_done_callback(on_goal_response)
        if not sent_event.wait(timeout=10.0):
            print("[NAV] Timed out waiting for Nav2 to accept the goal.")
            return False
        goal_handle = sent_result.get("handle")
        if goal_handle is None or not goal_handle.accepted:
            print(f"[NAV] Goal was rejected: {sent_result.get('error', 'no reason given')}")
            return False

        result_event = threading.Event()
        result_data = {}
        result_future = goal_handle.get_result_async()

        def on_result(future):
            try:
                result_data["result"] = future.result()
            except Exception as exc:
                result_data["error"] = exc
            finally:
                result_event.set()

        result_future.add_done_callback(on_result)
        if not result_event.wait(timeout=timeout):
            print("[NAV] Timed out waiting for the robot to reach its goal.")
            goal_handle.cancel_goal_async()
            return False

        result = result_data.get("result")
        succeeded = result is not None and result.status == GoalStatus.STATUS_SUCCEEDED
        if succeeded:
            print(f"[NAV] Reached goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.3f} rad")
            return True
        print(f"[NAV] Goal did not succeed (status={getattr(result, 'status', 'unknown')}).")
        return False

    def _on_camera_info(self, message):
        if len(message.k) < 6 or message.k[0] <= 0:
            return
        with self._lock:
            self._camera_fx = float(message.k[0])
            self._camera_cx = float(message.k[2])

    def select_unenrolled_seated_person(self):
        """Find one visible, unknown person overlapping a detected occupied seat."""
        with self._lock:
            if time.monotonic() - self._last_vision_message_at > 6.0:
                return None
            people = list(self._people or [])
            seats = list(self._seats or [])

        occupied_seats = [
            seat for seat in seats
            if seat.get("status", "").lower() == "occupied seat"
            and isinstance(seat.get("bbox"), (list, tuple))
            and len(seat["bbox"]) == 4
        ]
        candidates = []
        for person in people:
            if (person.get("guest_id") is not None
                    or person.get("identity_state") not in ("UNENROLLED", "NEW_PERSON")
                    or person.get("posture") != "sitting"
                    or person.get("tracker_id") is None):
                continue
            box = person.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            px1, py1, px2, py2 = (float(value) for value in box)
            person_area = max(1.0, (px2 - px1) * (py2 - py1))
            for seat in occupied_seats:
                sx1, sy1, sx2, sy2 = (float(value) for value in seat["bbox"])
                intersection = max(0.0, min(px2, sx2) - max(px1, sx1)) * max(
                    0.0, min(py2, sy2) - max(py1, sy1)
                )
                if intersection / person_area >= 0.05:
                    candidates.append(person)
                    break
        unique = {person["tracker_id"]: person for person in candidates}
        if len(unique) != 1:
            return None
        return next(iter(unique.values()))

    def _spin_by(self, angle, timeout=25.0):
        client = self._spin
        if client is None or not client.wait_for_server(timeout_sec=5.0):
            print("[NAV] Nav2 Spin action server is unavailable.")
            return False

        from nav2_msgs.action import Spin
        from action_msgs.msg import GoalStatus

        goal = Spin.Goal()
        goal.target_yaw = float(angle)
        goal.time_allowance.sec = int(timeout)
        goal.time_allowance.nanosec = int((timeout - int(timeout)) * 1_000_000_000)
        accepted_event = threading.Event()
        accepted = {}
        future = client.send_goal_async(goal)

        def on_goal_response(done):
            try:
                accepted["handle"] = done.result()
            except Exception as exc:
                accepted["error"] = exc
            finally:
                accepted_event.set()

        future.add_done_callback(on_goal_response)
        if not accepted_event.wait(timeout=10.0):
            print("[NAV] Timed out waiting for Nav2 to accept the Spin goal.")
            return False
        handle = accepted.get("handle")
        if handle is None or not handle.accepted:
            print(f"[NAV] Spin goal was rejected: {accepted.get('error', 'no reason given')}")
            return False

        result_event = threading.Event()
        result_data = {}
        result_future = handle.get_result_async()

        def on_result(done):
            try:
                result_data["result"] = done.result()
            except Exception as exc:
                result_data["error"] = exc
            finally:
                result_event.set()

        result_future.add_done_callback(on_result)
        if not result_event.wait(timeout=timeout + 2.0):
            handle.cancel_goal_async()
            print("[NAV] Timed out while turning toward the host.")
            return False
        result = result_data.get("result")
        return result is not None and result.status == GoalStatus.STATUS_SUCCEEDED

    def face_and_enroll_host(self, host_id, timeout=35.0):
        """Face the unique unknown seated person, enroll that track, then restore heading."""
        person = self.select_unenrolled_seated_person()
        if person is None:
            print("[VISION] Could not uniquely identify an unenrolled person in an occupied seat.")
            return False
        tracker_id = int(person["tracker_id"])
        with self._lock:
            fx, cx = self._camera_fx, self._camera_cx
        if fx is None or cx is None:
            print("[VISION] Camera calibration is unavailable; cannot calculate host bearing.")
            return False

        total_turn = 0.0
        centered = False
        enrolled = False
        try:
            for _ in range(8):
                with self._lock:
                    current = next(
                        (item for item in (self._people or [])
                         if item.get("tracker_id") == tracker_id),
                        None,
                    )
                    fresh = time.monotonic() - self._last_vision_message_at <= 6.0
                if current is None or not fresh:
                    print("[VISION] Host left the camera view while centering.")
                    break
                center_x = float(current.get("center_pixel_x", 0.0))
                bearing = math.atan2(center_x - cx, fx)
                if abs(bearing) <= 0.07:
                    centered = True
                    break
                # Positive image offset is camera-right, which is negative robot yaw.
                turn = max(-0.60, min(0.60, -bearing))
                if not self._spin_by(turn):
                    break
                total_turn += turn
                with self._lock:
                    previous_sequence = self._vision_sequence
                deadline = time.monotonic() + 7.0
                while time.monotonic() < deadline:
                    with self._lock:
                        if self._vision_sequence != previous_sequence:
                            break
                    time.sleep(0.05)
            if centered:
                enrolled = self.enroll_guest_identity(
                    host_id, timeout=timeout, tracker_id=tracker_id
                )
        finally:
            if abs(total_turn) > 0.01:
                with self._lock:
                    before_restore = self._vision_sequence
                if self._spin_by(-total_turn):
                    deadline = time.monotonic() + 7.0
                    while time.monotonic() < deadline:
                        with self._lock:
                            if self._vision_sequence != before_restore:
                                break
                        time.sleep(0.05)
                else:
                    print("[NAV] Warning: unable to restore the initial heading after host enrollment.")

        if not centered:
            print("[VISION] Could not center the host; enrollment skipped.")
            return False
        return enrolled

    def _on_message(self, message):
        try:
            payload = json.loads(message.data)
            people = payload.get("people", [])
            seats = payload.get("seats", [])
            if not isinstance(people, list) or not isinstance(seats, list):
                return
        except (json.JSONDecodeError, TypeError, AttributeError):
            return

        with self._lock:
            self._people = people
            self._seats = seats
            self._last_vision_message_at = time.monotonic()
            self._vision_sequence += 1
        self._received.set()

    def _on_description(self, message):
        with self._lock:
            self._description = message.data.strip()
        self._description_received.set()

    def _on_vision_status(self, message):
        try:
            status = json.loads(message.data)
            if not isinstance(status, dict):
                return
        except (json.JSONDecodeError, TypeError):
            return
        with self._lock:
            self._vision_status = status
        self._vision_status_received.set()

    def _on_identity_status(self, message):
        try:
            status = json.loads(message.data)
            if not isinstance(status, dict) or not status.get("guest_id"):
                return
        except (json.JSONDecodeError, TypeError):
            return
        with self._lock:
            self._identity_status[status["guest_id"]] = status
        self._identity_status_received.set()

    def enroll_guest_identity(self, guest_id, timeout=35.0, tracker_id=None):
        """Request an isolated multi-frame guest enrollment and wait for status."""
        publisher = getattr(self, "_identity_command_publisher", None)
        if self._node is None or publisher is None:
            print("[VISION] ROS identity enrollment is unavailable.")
            return False
        with self._lock:
            self._identity_status.pop(guest_id, None)
        self._identity_status_received.clear()

        from std_msgs.msg import String

        request = String()
        payload = {"action": "enroll", "guest_id": guest_id}
        if tracker_id is not None:
            payload["tracker_id"] = int(tracker_id)
        request.data = json.dumps(payload)
        publisher.publish(request)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                status = self._identity_status.get(guest_id)
            if status:
                if status.get("status") == "enrolled":
                    print(f"[VISION] Enrolled {guest_id}: {status.get('detail', '')}")
                    return True
                if status.get("status") == "failed":
                    print(f"[VISION] Could not enroll {guest_id}: {status.get('detail', '')}")
                    return False
            self._identity_status_received.wait(min(0.5, max(0.0, deadline - time.monotonic())))
            self._identity_status_received.clear()
        print(f"[VISION] Timed out enrolling {guest_id}.")
        return False

    def get_people(self):
        """Return fresh person identities from the latest vision snapshot."""
        with self._lock:
            if time.monotonic() - self._last_vision_message_at > 6.0:
                return None
            return None if self._people is None else list(self._people)

    def get_available_seat(self, timeout=0.0):
        """Return the nearest depth-valid free chair/couch from a fresh snapshot."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                seats = self._seats
                fresh = time.monotonic() - self._last_vision_message_at <= 6.0
            if seats is not None and fresh:
                available = [
                    seat for seat in seats
                    if seat.get("status", "").lower() == "empty seat"
                    and 0.0 < float(seat.get("distance_meters", 0.0)) <= 5.0
                ]
                if available:
                    return min(
                        available,
                        key=lambda seat: float(seat["distance_meters"]),
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.1, remaining))

    def wait_for_seat_occupied(self, seat_id, timeout=20.0):
        """Wait until the selected place is occupied according to pose tracking."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                seats = self._seats
                fresh = time.monotonic() - self._last_vision_message_at <= 6.0
            if fresh and seats is not None:
                seat = next((item for item in seats if item.get("id") == seat_id), None)
                if seat is not None and seat.get("status", "").lower() == "occupied seat":
                    return True
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return False

    def wait_for_guest_sitting(self, guest_id, selected_seat, timeout=20.0, confirm_duration=1.0):
        """Confirm the enrolled guest sits in the seat offered before it is occluded."""
        seat_box = selected_seat.get("bbox")
        if not isinstance(seat_box, (list, tuple)) or len(seat_box) != 4:
            print("[VISION] Cannot confirm seating: selected seat has no image bounds.")
            return False
        sx1, sy1, sx2, sy2 = (float(value) for value in seat_box)
        seat_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
        deadline = time.monotonic() + timeout
        sitting_since = None
        last_sequence = -1
        while time.monotonic() < deadline:
            with self._lock:
                people = self._people
                fresh = time.monotonic() - self._last_vision_message_at <= 2.0
                sequence = self._vision_sequence

            if fresh and people is not None and sequence != last_sequence:
                last_sequence = sequence
                guest_is_sitting = False
                for person in people:
                    person_box = person.get("bbox")
                    if not isinstance(person_box, (list, tuple)) or len(person_box) != 4:
                        continue
                    px1, py1, px2, py2 = (float(value) for value in person_box)
                    overlap = max(0.0, min(sx2, px2) - max(sx1, px1)) * max(
                        0.0, min(sy2, py2) - max(sy1, py1)
                    )
                    if (
                        person.get("guest_id") == guest_id
                        and person.get("identity_state") in ("TRACKING", "REID_RECOVERED")
                        and person.get("posture") == "sitting"
                        and float(person.get("pose_confidence", 0.0)) >= 0.35
                        and overlap / seat_area >= 0.15
                    ):
                        guest_is_sitting = True
                        break
                if guest_is_sitting:
                    if sitting_since is None:
                        sitting_since = time.monotonic()
                    elif time.monotonic() - sitting_since >= confirm_duration:
                        return True
                else:
                    sitting_since = None

            time.sleep(0.05)
        return False

    def wait_for_guest_identities(self, guest_ids, timeout=20.0):
        """Require each logical guest to be visibly and uniquely identified."""
        required = set(guest_ids)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            people = self.get_people()
            if people is not None:
                matches = {}
                for person in people:
                    guest_id = person.get("guest_id")
                    if guest_id in required and person.get("identity_state") in (
                        "TRACKING", "REID_RECOVERED"
                    ):
                        matches.setdefault(guest_id, []).append(person)
                if all(len(matches.get(guest_id, [])) == 1 for guest_id in required):
                    return matches
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return None

    def describe_scene(self, timeout=30.0):
        """Ask the vision node to describe its latest frame and await its reply."""
        if self._node is None or not hasattr(self, "_describe_publisher"):
            return None
        with self._lock:
            self._description = None
        self._description_received.clear()

        from std_msgs.msg import String

        request = String()
        request.data = "describe_latest_frame"
        self._describe_publisher.publish(request)
        if not self._description_received.wait(timeout):
            return None
        with self._lock:
            return self._description

    def stop(self):
        """Stop this subscriber without shutting down another ROS context."""
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

