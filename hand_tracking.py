"""
VisionCanvas AI — Hand Tracking Module
Encapsulates MediaPipe Hands into a reusable, high-performance HandTracker class.
Tracks 21 hand landmarks, identifies extended fingers, and extracts fingertip coordinates.
"""

import cv2
import mediapipe as mp
import time
import math


class HandTracker:
    """
    Modular Hand Tracking wrapper around MediaPipe Hands solution.
    """

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 1,
        detection_con: float = 0.7,
        track_con: float = 0.7,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.results = None
        self.landmark_list = []

    def find_hands(self, frame, draw: bool = True):
        """
        Processes a BGR frame, runs MediaPipe hand detection, and optionally draws landmarks.
        Returns the annotated frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame

    def find_positions(self, frame, hand_idx: int = 0, draw: bool = False):
        """
        Extracts pixel coordinates for all 21 landmarks of the specified hand.
        Returns a list of [id, x, y, z].
        """
        self.landmark_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_idx < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_idx]
                h, w, _ = frame.shape
                for lm_id, lm in enumerate(my_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([lm_id, cx, cy, lm.z])
                    if draw and lm_id == 8:
                        # Highlight index fingertip
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), cv2.FILLED)
        return self.landmark_list

    def fingers_up(self, hand_idx: int = 0):
        """
        Determines which fingers are extended.
        Returns: [thumb, index, middle, ring, pinky] with 1 for extended and 0 for folded.
        Returns [] if no hand landmarks are available.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return []

        if hand_idx >= len(self.results.multi_hand_landmarks):
            return []

        my_hand = self.results.multi_hand_landmarks[hand_idx]
        landmarks = my_hand.landmark
        fingers = []

        # Determine handedness (Right vs Left) if available
        # Note: on a horizontally flipped frame (mirror mode), user's right hand appears on right
        hand_label = "Right"
        if self.results.multi_handedness and hand_idx < len(self.results.multi_handedness):
            hand_label = self.results.multi_handedness[hand_idx].classification[0].label

        # 1. Thumb Check
        # Robust geometric check: compare distance from thumb tip (4) to pinky MCP base (17)
        # vs thumb IP joint (3) to pinky MCP base (17)
        dist_tip_to_pinky_base = math.hypot(
            landmarks[4].x - landmarks[17].x, landmarks[4].y - landmarks[17].y
        )
        dist_ip_to_pinky_base = math.hypot(
            landmarks[3].x - landmarks[17].x, landmarks[3].y - landmarks[17].y
        )

        if dist_tip_to_pinky_base > dist_ip_to_pinky_base:
            fingers.append(1)
        else:
            fingers.append(0)

        # 2. Four Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # Compare tip y to PIP joint y (6, 10, 14, 18)
        # Since y=0 is at the top of the image, tip.y < pip.y means finger is pointing upwards
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_index_tip(self, frame, hand_idx: int = 0):
        """
        Helper method to retrieve the pixel (x, y) coordinates of the index fingertip.
        Returns (x, y) or None if not detected.
        """
        lm_list = self.find_positions(frame, hand_idx=hand_idx, draw=False)
        if len(lm_list) > 8:
            return lm_list[8][1], lm_list[8][2]
        return None

    def close(self):
        """Releases underlying MediaPipe graph resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()


def main():
    """
    Visual test runner for HandTracker.
    """
    print("=" * 50)
    print("VisionCanvas AI — Hand Tracking Live Diagnostic")
    print("=" * 50)
    print("Initializing HandTracker...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    tracker = HandTracker(max_hands=1, detection_con=0.7, track_con=0.7)
    prev_time = time.time()

    print("Running live tracking preview. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Frame dropped.")
            break

        # Mirror display for natural air-drawing interaction
        frame = cv2.flip(frame, 1)

        # Track hands and draw landmarks
        frame = tracker.find_hands(frame, draw=True)
        lm_list = tracker.find_positions(frame, draw=True)
        fingers = tracker.fingers_up()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        # Display status on frame
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if fingers:
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            active_fingers = [name for name, val in zip(finger_names, fingers) if val == 1]
            status_text = f"Fingers: {active_fingers}" if active_fingers else "Fist / None"
            cv2.putText(
                frame,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if len(lm_list) > 8:
                tip_x, tip_y = lm_list[8][1], lm_list[8][2]
                cv2.putText(
                    frame,
                    f"Index Tip: ({tip_x}, {tip_y})",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

        cv2.imshow("VisionCanvas AI - Hand Tracking Diagnostic", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Hand tracking test complete.")


if __name__ == "__main__":
    main()