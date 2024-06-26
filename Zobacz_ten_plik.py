import cv2
import numpy as np
import time
from threading import Thread, Event
from pynput import keyboard

class ShapeDetector:
    def __init__(self):
        self.area = 0
        self.triangle_detected_time = None
        self.square_detected_time = None
        self.circle_detected_time = None
        self.detection_duration = 2  # Duration in seconds
        self.stop_controller = Event()
        self.chose_shape = None
        self.centering_in_progress = False
        self.cap = cv2.VideoCapture(0)

        # Let the user choose the shape in a separate thread
        self.chose_thread = Thread(target=self.chose)
        self.chose_thread.start()

        # Start the camera feed thread
        self.camera_thread = Thread(target=self.detect)
        self.camera_thread.start()

        # Start detection when an instance is created
        self.wait_gone()

    def mission_kwadr(self):
        print("wykonanie misji dla kwadrat")

    def mission_trojkat(self):
        print("wykonanie misji dla trojkat")

    def mission_kolo(self):
        print("wykonanie misji dla kolo")

    def on_press(self, key):
        try:
            print("H - kwadrat")
            print("J - trojkat")
            print("K - kolo")
            if key.char == 'h':
                self.chose_shape = 'square'
            elif key.char == 'j':
                self.chose_shape = 'triangle'
            elif key.char == 'k':
                self.chose_shape = 'circle'
        except AttributeError:
            pass

    def chose(self):
        while True:
            with keyboard.Listener(on_press=self.on_press) as listener:
                print("H - kwadrat")
                print("J - trojkat")
                print("K - kolo")
                listener.join()  # Block until a key is pressed

            if self.chose_shape is not None:
                break

    def nothing(self, x):
        pass

    def check_detection_duration(self, shape_detected, shape_type, center):
        current_time = time.time()

        if shape_type == 'triangle':
            if shape_detected:
                if self.triangle_detected_time is None:
                    self.triangle_detected_time = current_time
                elif current_time - self.triangle_detected_time >= self.detection_duration:
                    self.centering_in_progress = True
                    centering_thread = Thread(target=self.wait_until_centered, args=(center, 'triangle'))
                    centering_thread.start()
                    return True
            else:
                self.triangle_detected_time = None

        if shape_type == 'square':
            if shape_detected:
                if self.square_detected_time is None:
                    self.square_detected_time = current_time
                elif current_time - self.square_detected_time >= self.detection_duration:
                    self.centering_in_progress = True
                    centering_thread = Thread(target=self.wait_until_centered, args=(center, 'square'))
                    centering_thread.start()
                    return True
            else:
                self.square_detected_time = None

        if shape_type == 'circle':
            if shape_detected:
                if self.circle_detected_time is None:
                    self.circle_detected_time = current_time
                elif current_time - self.circle_detected_time >= self.detection_duration:
                    self.centering_in_progress = True
                    centering_thread = Thread(target=self.wait_until_centered, args=(center, 'circle'))
                    centering_thread.start()
                    return True
            else:
                self.circle_detected_time = None

        return False

    def angle_between_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
        if magnitudes == 0:
            return 0
        cos_angle = dot_product / magnitudes
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def center_check(self, center):
        frame_center = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
        tolerance = 50  # Tolerance in pixels

        if abs(center[0] - frame_center[0]) < tolerance and abs(center[1] - frame_center[1]) < tolerance:
            return True
        else:
            if center[0] < frame_center[0] - tolerance:
                print("Shape is to the left")
            elif center[0] > frame_center[0] + tolerance:
                print("Shape is to the right")

            if center[1] < frame_center[1] - tolerance:
                print("Shape is above")
            elif center[1] > frame_center[1] + tolerance:
                print("Shape is below")

            return False

    def wait_until_centered(self, initial_center, shape_type):
        while self.centering_in_progress:
            ret, frame = self.cap.read()  # Continuously read from the camera to avoid freezing
            if not ret:
                break

            # Update the position of the shape based on new frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low_h = cv2.getTrackbarPos('Low H', 'Kontury')
            high_h = cv2.getTrackbarPos('High H', 'Kontury')
            low_s = cv2.getTrackbarPos('Low S', 'Kontury')
            high_s = cv2.getTrackbarPos('High S', 'Kontury')
            low_v = cv2.getTrackbarPos('Low V', 'Kontury')
            high_v = cv2.getTrackbarPos('High V', 'Kontury')
            lower_bound = np.array([low_h, low_s, low_v])
            upper_bound = np.array([high_h, high_s, high_v])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            shape_center = None
            shape_detected = False

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                area = cv2.contourArea(contour)
                if area < 500:
                    continue
                if len(approx) == 3 and shape_type == 'triangle':
                    v1 = approx[1][0] - approx[0][0]
                    v2 = approx[2][0] - approx[1][0]
                    v3 = approx[0][0] - approx[2][0]
                    angle1 = self.angle_between_vectors(v1, -v3)
                    angle2 = self.angle_between_vectors(v2, -v1)
                    angle3 = self.angle_between_vectors(v3, -v2)
                    if (np.isclose(angle1, 60, atol=10) and
                        np.isclose(angle2, 60, atol=10) and
                        np.isclose(angle3, 60, atol=10)):
                        shape_center = np.mean(approx, axis=0)[0]
                        shape_detected = True
                        break
                elif len(approx) == 4 and shape_type == 'square':
                    x, y, w, h = cv2.boundingRect(approx)
                    if abs(w - h) < 0.1 * max(w, h):
                        shape_center = np.mean(approx, axis=0)[0]
                        shape_detected = True
                        break
                elif len(approx) > 4 and shape_type == 'circle':
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    shape_center = (int(x), int(y))
                    if abs(cv2.contourArea(contour) - (np.pi * radius * radius)) < 0.2 * (np.pi * radius * radius):
                        shape_center = (int(x), int(y))
                        shape_detected = True
                        break

            if not shape_detected:
                print(f"{shape_type.capitalize()} disappeared. Restarting detection...")
                self.centering_in_progress = False
                self.triangle_detected_time = None
                self.square_detected_time = None
                self.circle_detected_time = None
                self.chose_shape = shape_type  # Continue searching for the same shape
                self.wait_gone()  # Restart detection
                return

            if self.center_check(shape_center):
                self.centering_in_progress = False
                if shape_type == 'triangle':
                    self.mission_trojkat()
                elif shape_type == 'square':
                    self.mission_kwadr()
                elif shape_type == 'circle':
                    self.mission_kolo()

    def detect(self):
        cv2.namedWindow('Kontury')
        cv2.createTrackbar('Low H', 'Kontury', 0, 179, self.nothing)
        cv2.createTrackbar('High H', 'Kontury', 179, 179, self.nothing)
        cv2.createTrackbar('Low S', 'Kontury', 0, 255, self.nothing)
        cv2.createTrackbar('High S', 'Kontury', 110, 255, self.nothing)
        cv2.createTrackbar('Low V', 'Kontury', 0, 255, self.nothing)
        cv2.createTrackbar('High V', 'Kontury', 110, 255, self.nothing)

        while not self.stop_controller.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low_h = cv2.getTrackbarPos('Low H', 'Kontury')
            high_h = cv2.getTrackbarPos('High H', 'Kontury')
            low_s = cv2.getTrackbarPos('Low S', 'Kontury')
            high_s = cv2.getTrackbarPos('High S', 'Kontury')
            low_v = cv2.getTrackbarPos('Low V', 'Kontury')
            high_v = cv2.getTrackbarPos('High V', 'Kontury')

            lower_bound = np.array([low_h, low_s, low_v])
            upper_bound = np.array([high_h, high_s, high_v])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            triangle_detected = False
            kwadr_detected = False
            kolo_detected = False
            shape_center = None

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                self.area = cv2.contourArea(contour)
                if self.area < 500:
                    continue

                if len(approx) == 3:
                    v1 = approx[1][0] - approx[0][0]
                    v2 = approx[2][0] - approx[1][0]
                    v3 = approx[0][0] - approx[2][0]
                    angle1 = self.angle_between_vectors(v1, -v3)
                    angle2 = self.angle_between_vectors(v2, -v1)
                    angle3 = self.angle_between_vectors(v3, -v2)
                    if (np.isclose(angle1, 60, atol=10) and
                        np.isclose(angle2, 60, atol=10) and
                        np.isclose(angle3, 60, atol=10)):
                        cv2.drawContours(frame, [approx], 0, (0, 255, 255), 2)
                        cv2.putText(frame, "triangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 255))
                        triangle_detected = True
                        shape_center = np.mean(approx, axis=0)[0]

                elif len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    if abs(w - h) < 0.1 * max(w, h):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "kwadrat", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                        kwadr_detected = True
                        shape_center = np.mean(approx, axis=0)[0]
                else:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    if abs(cv2.contourArea(contour) - (np.pi * radius * radius)) < 0.2 * (np.pi * radius * radius):
                        cv2.circle(frame, center, radius, (255, 0, 0), 2)
                        cv2.putText(frame, "circle", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                        kolo_detected = True
                        shape_center = center

            if not self.centering_in_progress:
                if self.chose_shape == 'triangle':
                    if self.check_detection_duration(triangle_detected, 'triangle', shape_center):
                        self.chose_shape = None
                elif self.chose_shape == 'square':
                    if self.check_detection_duration(kwadr_detected, 'square', shape_center):
                        self.chose_shape = None
                elif self.chose_shape == 'circle':
                    if self.check_detection_duration(kolo_detected, 'circle', shape_center):
                        self.chose_shape = None

            cv2.imshow('Kontury', frame)
            cv2.imshow('Maska', mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_controller.set()
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def wait_gone(self):
        while not self.stop_controller.is_set():
            time.sleep(0.3)  # Slight delay to avoid busy-waiting

if __name__ == '__main__':
    ShapeDetector()
