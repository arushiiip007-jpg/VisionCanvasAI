"""
VisionCanvas AI — Air Drawing Application
Phase 1: Real-time hand tracking, jitter-free index finger drawing,
safe fist-hold clearing, vibrant bitwise canvas blending, and tool management.
"""

import os
import time
import math
from datetime import datetime
import cv2
import numpy as np
from hand_tracking import HandTracker

# Preset Color Palette (BGR format)
PALETTE = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 120, 0),
    "Yellow": (0, 255, 255),
    "Purple": (255, 0, 255),
    "White": (255, 255, 255),
}
COLOR_KEYS = list(PALETTE.keys())


class AirCanvasApp:
    def __init__(self):
        # Tracking & Video
        self.tracker = HandTracker(max_hands=1, detection_con=0.7, track_con=0.7)
        self.cap = cv2.VideoCapture(0)

        # Drawing state
        self.canvas = None
        self.current_color_idx = 0  # Red by default
        self.brush_size = 6
        self.eraser_size = 40
        self.is_eraser = False
        self.canvas_only_view = False

        # Trajectory smoothing (Exponential Moving Average)
        self.smooth_x, self.smooth_y = 0, 0
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.65  # Current vs historical weight

        # Safety fist-hold confirmation for clearing canvas
        self.fist_counter = 0
        self.fist_hold_threshold = 35  # ~1.2 seconds at 30 FPS
        self.clear_cooldown = 0
        self.status_message = "Ready"
        self.status_timer = 0

        # Performance monitoring
        self.prev_time = time.time()
        self.fps = 0.0

        # Save directory
        self.save_dir = "saved_drawings"
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def current_color(self):
        if self.is_eraser:
            return (0, 0, 0)
        color_name = COLOR_KEYS[self.current_color_idx]
        return PALETTE[color_name]

    def set_status(self, message: str, duration_sec: float = 2.0):
        self.status_message = message
        self.status_timer = time.time() + duration_sec

    def save_canvas(self):
        """Exports the digital canvas as a clean PNG image."""
        if self.canvas is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"drawing_{timestamp}.png")
        cv2.imwrite(filename, self.canvas)
        self.set_status(f"Saved: {os.path.basename(filename)}")
        print(f"[VisionCanvas AI] Drawing exported successfully: {filename}")

    def render_hud(self, display_frame, mode_text: str):
        """Renders a sleek top status bar and UI HUD overlay."""
        h, w, _ = display_frame.shape

        # Top Bar Background (Glassmorphism dark overlay)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

        # Title & FPS
        cv2.putText(
            display_frame,
            "VisionCanvas AI",
            (15, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display_frame,
            f"FPS: {self.fps:.0f}",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
        )

        # Active Mode & Tool Indicator
        color_name = "Eraser" if self.is_eraser else COLOR_KEYS[self.current_color_idx]
        active_color = (180, 180, 180) if self.is_eraser else self.current_color

        # Swatch circle
        cv2.circle(display_frame, (230, 32), 14, active_color, cv2.FILLED)
        cv2.circle(display_frame, (230, 32), 15, (255, 255, 255), 1)

        tool_size = self.eraser_size if self.is_eraser else self.brush_size
        cv2.putText(
            display_frame,
            f"{color_name} ({tool_size}px)",
            (255, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
        )

        # Mode Tag (Drawing vs Hovering vs Fist Hold)
        mode_color = (0, 255, 120) if "DRAW" in mode_text else (0, 200, 255)
        if "CLEAR" in mode_text:
            mode_color = (0, 0, 255)
        cv2.putText(
            display_frame,
            f"Mode: {mode_text}",
            (450, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            mode_color,
            2,
        )

        # Temporary Toast Notifications
        if time.time() < self.status_timer:
            cv2.putText(
                display_frame,
                self.status_message,
                (w - 380, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        else:
            # Show navigation shortcuts
            cv2.putText(
                display_frame,
                "1-6: Colors | E: Eraser | S: Save | C: Clear | V: View | Q: Quit",
                (w - 530, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )

        # Fist Hold Progress Bar (Visual Confirmation)
        if self.fist_counter > 0 and self.clear_cooldown == 0:
            progress = min(1.0, self.fist_counter / self.fist_hold_threshold)
            bar_w = int(220 * progress)
            bar_x = w // 2 - 110
            bar_y = h - 50
            cv2.rectangle(display_frame, (bar_x - 5, bar_y - 20), (bar_x + 225, bar_y + 15), (30, 30, 30), cv2.FILLED)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (0, 0, 255), cv2.FILLED)
            cv2.putText(
                display_frame,
                f"HOLD FIST TO CLEAR: {int(progress * 100)}%",
                (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    def run(self):
        if not self.cap.isOpened():
            print("[ERROR] Cannot open webcam. Check device connection.")
            return

        print("=" * 60)
        print("VisionCanvas AI — Air Drawing Active")
        print("=" * 60)
        print("Gestures:")
        print("  - Index Finger Only : Draw / Erase")
        print("  - Multiple Fingers  : Hover / Move cursor")
        print("  - Hold Closed Fist  : Safe Clear (Hold ~1.2s)")
        print("\nKeyboard Shortcuts:")
        print("  - 1 to 6 : Select Colors (Red, Green, Blue, Yellow, Purple, White)")
        print("  - E      : Toggle Eraser")
        print("  - + / -  : Increase / Decrease Brush Size")
        print("  - C      : Clear Canvas Immediately")
        print("  - S      : Save / Export Drawing PNG")
        print("  - V      : Toggle Canvas View (Overlay vs Pure Canvas)")
        print("  - Q      : Quit Application")
        print("=" * 60)

        while True:
            success, frame = self.cap.read()
            if not success:
                print("[WARNING] Frame capture dropped.")
                break

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)

            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Update FPS
            curr_time = time.time()
            self.fps = 1.0 / (curr_time - self.prev_time) if curr_time > self.prev_time else 0.0
            self.prev_time = curr_time

            if self.clear_cooldown > 0:
                self.clear_cooldown -= 1

            # Hand Tracking
            frame = self.tracker.find_hands(frame, draw=False)
            lm_list = self.tracker.find_positions(frame, draw=False)
            fingers = self.tracker.fingers_up()

            mode_text = "STANDBY"

            if fingers and len(lm_list) > 8:
                thumb, index, middle, ring, pinky = fingers
                raw_x, raw_y = lm_list[8][1], lm_list[8][2]

                # Exponential Moving Average for jitter reduction
                if self.smooth_x == 0 and self.smooth_y == 0:
                    self.smooth_x, self.smooth_y = raw_x, raw_y
                else:
                    self.smooth_x = int(self.smoothing_factor * raw_x + (1 - self.smoothing_factor) * self.smooth_x)
                    self.smooth_y = int(self.smoothing_factor * raw_y + (1 - self.smoothing_factor) * self.smooth_y)

                # 1. DRAWING MODE: Only Index finger is extended
                if index == 1 and middle == 0 and ring == 0 and pinky == 0:
                    self.fist_counter = 0
                    mode_text = "ERASING" if self.is_eraser else "DRAWING"

                    # Check for large jump / teleportation (stroke restart)
                    jump_dist = math.hypot(self.smooth_x - self.prev_x, self.smooth_y - self.prev_y)
                    if (self.prev_x == 0 and self.prev_y == 0) or jump_dist > 120:
                        self.prev_x, self.prev_y = self.smooth_x, self.smooth_y

                    current_thickness = self.eraser_size if self.is_eraser else self.brush_size
                    cv2.line(
                        self.canvas,
                        (self.prev_x, self.prev_y),
                        (self.smooth_x, self.smooth_y),
                        self.current_color,
                        current_thickness,
                    )
                    cv2.circle(
                        self.canvas,
                        (self.smooth_x, self.smooth_y),
                        max(1, current_thickness // 2),
                        self.current_color,
                        cv2.FILLED,
                    )
                    self.prev_x, self.prev_y = self.smooth_x, self.smooth_y

                    # Draw active cursor on frame
                    cursor_color = (200, 200, 200) if self.is_eraser else self.current_color
                    cv2.circle(frame, (self.smooth_x, self.smooth_y), current_thickness // 2 + 2, cursor_color, cv2.FILLED)

                # 2. CLOSED FIST: Safe Canvas Clear (with hold requirement)
                elif thumb == 0 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
                    self.prev_x, self.prev_y = 0, 0
                    if self.clear_cooldown == 0:
                        self.fist_counter += 1
                        mode_text = "HOLD FIST TO CLEAR"
                        if self.fist_counter >= self.fist_hold_threshold:
                            self.canvas = np.zeros_like(frame)
                            self.fist_counter = 0
                            self.clear_cooldown = 40  # Prevent accidental re-triggering
                            self.set_status("Canvas Cleared (Fist Hold)")
                    else:
                        mode_text = "CLEARED"

                # 3. HOVER / SELECTION MODE: Multiple fingers extended
                else:
                    self.prev_x, self.prev_y = 0, 0
                    self.fist_counter = 0
                    mode_text = "HOVER"

                    # Draw hover pointer circle
                    cv2.circle(frame, (self.smooth_x, self.smooth_y), 10, (255, 255, 0), 2)
                    cv2.circle(frame, (self.smooth_x, self.smooth_y), 3, (0, 255, 255), cv2.FILLED)

            else:
                self.prev_x, self.prev_y = 0, 0
                self.smooth_x, self.smooth_y = 0, 0
                self.fist_counter = 0

            # Vibrant Canvas Blending (Bitwise Masking)
            if self.canvas_only_view:
                # Digital whiteboard mode (inverted canvas on pure white or black background)
                display_output = self.canvas.copy()
            else:
                # Seamless blending: keeps webcam 100% bright, canvas 100% opaque
                canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
                display_output = cv2.add(frame_bg, canvas_fg)

            # Draw HUD
            self.render_hud(display_output, mode_text)

            cv2.imshow("VisionCanvas AI - Air Drawing", display_output)

            # Keyboard Events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
                self.set_status("Canvas Cleared (Key)")
            elif key == ord('e'):
                self.is_eraser = not self.is_eraser
                self.set_status("Eraser ON" if self.is_eraser else "Pen ON")
            elif key == ord('s'):
                self.save_canvas()
            elif key == ord('v'):
                self.canvas_only_view = not self.canvas_only_view
                self.set_status("Canvas Only View" if self.canvas_only_view else "Camera Overlay View")
            elif key in [ord('+'), ord('=')]:
                if self.is_eraser:
                    self.eraser_size = min(80, self.eraser_size + 5)
                else:
                    self.brush_size = min(30, self.brush_size + 2)
                self.set_status(f"Size: {self.eraser_size if self.is_eraser else self.brush_size}px")
            elif key in [ord('-'), ord('_')]:
                if self.is_eraser:
                    self.eraser_size = max(10, self.eraser_size - 5)
                else:
                    self.brush_size = max(2, self.brush_size - 2)
                self.set_status(f"Size: {self.eraser_size if self.is_eraser else self.brush_size}px")
            elif ord('1') <= key <= ord(str(len(COLOR_KEYS))):
                self.is_eraser = False
                self.current_color_idx = key - ord('1')
                self.set_status(f"Color: {COLOR_KEYS[self.current_color_idx]}")

        self.cap.release()
        cv2.destroyAllWindows()
        self.tracker.close()
        print("Application closed successfully.")


if __name__ == "__main__":
    app = AirCanvasApp()
    app.run()