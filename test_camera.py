"""
VisionCanvas AI — Camera Diagnostic Tool
Verifies webcam connectivity, driver compatibility, and frame capture rate.
"""

import cv2
import time


def main():
    print("=" * 50)
    print("VisionCanvas AI — Camera Diagnostics")
    print("=" * 50)
    print("Attempting to connect to camera index 0...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera could not be opened. Check permissions or connections.")
        return

    # Attempt to query resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[SUCCESS] Camera detected: {width}x{height} @ ~{fps:.1f} FPS")
    print("Streaming live preview window. Press 'q' in the preview window to exit.")

    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Frame capture dropped or camera disconnected.")
            break

        # Calculate live FPS
        curr_time = time.time()
        live_fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        # Render diagnostic overlay
        cv2.putText(
            frame,
            f"Camera OK | {width}x{height} | {live_fps:.1f} FPS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("VisionCanvas AI - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Diagnostic test complete.")


if __name__ == "__main__":
    main()