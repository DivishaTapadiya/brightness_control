"""
AIR BRIGHTNESS CONTROLLER 

How it works:
  - The program detects your HAND using skin-tone color detection (HSV).
  - It finds the largest skin-colored blob in frame -- that's your hand.
  - The hand's VERTICAL position (how high/low it is in the frame) controls
    brightness live -- move your hand up = brighter, down = dimmer, like an
    invisible slider.
"""

import cv2
import numpy as np

# tune these to your skin tone / lighting if detection is weak


# HSV range for skin tone. This is a common starting range; if it doesn't
# pick up your hand well under your lighting, we'll adjust these together.
SKIN_LOWER = np.array([0, 30, 60])
SKIN_UPPER = np.array([25, 150, 255])

MIN_HAND_AREA = 3000     # ignore small skin-colored specks (noise)

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def find_hand_centroid(hsv_frame):
 
    mask = cv2.inRange(hsv_frame, SKIN_LOWER, SKIN_UPPER)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_HAND_AREA:
        return None, mask, None

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    return (cx, cy), mask, (x, y, w, h)


def main():
    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    cv2.namedWindow("Air Brightness Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Brightness Controller", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    brightness_level = 128  # start at mid brightness (0-255 scale)
    frame_height = None

    print("Show your hand to the camera. Move it up = brighter, down = dimmer.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view -feels natural
        if frame_height is None:
            frame_height = frame.shape[0]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hand_centroid, hand_mask, hand_box = find_hand_centroid(hsv)

        if hand_centroid:
            cx, cy = hand_centroid
            x, y, w, h = hand_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, hand_centroid, 8, (0, 0, 255), -1)

            # Map vertical position to brightness.
            # cy near TOP of frame (small y) -> brighter
            # cy near BOTTOM of frame (large y) -> dimmer
            brightness_level = np.interp(cy, [0, frame_height], [255, 0])

            cv2.putText(frame, f"Hand Y: {cy}px", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Apply brightness live 
        beta_shift = np.interp(brightness_level, [0, 255], [-100, 100])
        bright_frame = cv2.convertScaleAbs(frame, alpha=1, beta=beta_shift)

        #  Draw brightness bar UI
        bar_x, bar_y, bar_w, bar_h = 20, 120, 30, 250
        fill_h = int(np.interp(brightness_level, [0, 255], [0, bar_h]))
        cv2.rectangle(bright_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.rectangle(bright_frame, (bar_x, bar_y + bar_h - fill_h),
                      (bar_x + bar_w, bar_y + bar_h), (0, 255, 255), -1)
        cv2.putText(bright_frame, f"{int(brightness_level / 255 * 100)}%",
                    (bar_x - 5, bar_y + bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Air Brightness Controller", bright_frame)
        cv2.imshow("Hand Mask", hand_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()