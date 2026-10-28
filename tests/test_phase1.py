"""
VisionCanvas AI — Phase 1 Test Suite
Automated unit and integration checks for environment, camera, and HandTracker.
"""

import unittest
import numpy as np
import cv2
import mediapipe as mp
from hand_tracking import HandTracker
from air_draw import AirCanvasApp


class TestPhase1Environment(unittest.TestCase):
    """Verifies that all foundational packages are properly installed and linked."""

    def test_imports(self):
        self.assertTrue(hasattr(cv2, "__version__"), "OpenCV must be accessible.")
        self.assertTrue(hasattr(mp, "__version__"), "MediaPipe must be accessible.")
        self.assertTrue(hasattr(np, "__version__"), "NumPy must be accessible.")

    def test_numpy_compatibility(self):
        """MediaPipe 0.10.x requires NumPy < 2.0.0."""
        major_ver = int(np.__version__.split(".")[0])
        self.assertLess(
            major_ver, 2, f"NumPy version is {np.__version__}, but must be < 2.0.0 for MediaPipe stability."
        )


class TestHandTracker(unittest.TestCase):
    """Verifies HandTracker initialization, landmark processing, and edge cases."""

    def setUp(self):
        self.tracker = HandTracker(max_hands=1, detection_con=0.5, track_con=0.5)

    def tearDown(self):
        if hasattr(self.tracker, "close"):
            self.tracker.close()

    def test_tracker_initialization(self):
        self.assertIsNotNone(self.tracker.hands)
        self.assertEqual(self.tracker.max_hands, 1)

    def test_blank_frame_processing(self):
        """A blank black image should process smoothly with 0 detections."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed = self.tracker.find_hands(blank_frame, draw=False)
        self.assertEqual(processed.shape, (480, 640, 3))

        lm_list = self.tracker.find_positions(blank_frame)
        self.assertEqual(lm_list, [], "Blank frame should yield 0 landmarks.")

        fingers = self.tracker.fingers_up()
        self.assertEqual(fingers, [], "Blank frame should yield empty finger array.")

        tip = self.tracker.get_index_tip(blank_frame)
        self.assertIsNone(tip, "Index tip should be None when no hand is present.")


class TestAirCanvasApp(unittest.TestCase):
    """Verifies AirCanvasApp state machine and rendering pipelines."""

    def test_app_initialization(self):
        app = AirCanvasApp()
        self.assertIsNotNone(app.tracker)
        self.assertEqual(app.brush_size, 6)
        self.assertEqual(app.is_eraser, False)
        self.assertIsNotNone(app.current_color)

        # Release capture if opened
        if app.cap.isOpened():
            app.cap.release()

    def test_hud_rendering(self):
        app = AirCanvasApp()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.canvas = np.zeros_like(mock_frame)
        app.render_hud(mock_frame, "STANDBY")
        # Ensure frame modified (HUD pixels drawn)
        self.assertGreater(np.count_nonzero(mock_frame), 0, "HUD should draw pixels on the frame.")
        if app.cap.isOpened():
            app.cap.release()


if __name__ == "__main__":
    unittest.main()
