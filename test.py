import cv2
import keyboard
import numpy as np
import time
from threading import Thread, Event


class ShapeDetector:
    class TelloTimer(Thread):
        interval = 1.0
        running = None
        func = None

        def __init__(self, interval, event, func):
            Thread.__init__(self)
            self.running = event
            self.interval = interval
            self.func = func

        def run(self):
            while not self.running.wait(self.interval):
                self.func()

    def mission_kwadr(self):
        print("wykonanie misji dla kwadrat")

    def mission_trojkat(self):
        print("wykonanie misji dla trojkat")

    def mission_kolo(self):
        print("wykonanie misji dla kolo")

    def chose(self):
        self.chose_shape = None
        while True:
            print("H - kwadrat")
            print("J - trojkat")
            print("K - kolo")
            key = keyboard.read_key()
            if key == "h":
                self.chose_shape = 'square'
                break
            elif key == "j":
                self.chose_shape = 'triangle'
                break
            elif key == "k":
                self.chose_shape = 'circle'
                break

    def nothing(self, x):
        pass

    def check_detection_duration(self, shape_detected, shape_type):
        current_time = time.time()

        if shape_type == 'triangle':
            if shape_detected:
                if self.triangle_detected_time is None:
                    self.triangle_detected_time = current_time
                elif current_time - self.triangle_detected_time >= self.detection_duration:
                    print("hura - triangle")
                    self.mission_trojkat()  # Perform the mission for triangle
                    self.triangle_detected_time = None  # Reset after detection
                    return True
            else:
                self.triangle_detected_time = None

        if shape_type == 'square':
            if shape_detected:
                if self.square_detected_time is None:
                    self.square_detected_time = current_time
                elif current_time - self.square_detected_time >= self.detection_duration:
                    print("hura - square")
                    self.mission_kwadr()  # Perform the mission for square
                    self.square_detected_time = None  # Reset after detection
                    return True
            else:
                self.square_detected_time = None

        if shape_type == 'circle':
            if shape_detected:
                if self.circle_detected_time is None:
                    self.circle_detected_time = current_time
                elif current_time - self.circle_detected_time >= self.detection_duration:
                    print("hura - circle")
                    self.mission_kolo()  # Perform the mission for circle
                    self.circle_detected_time = None  # Reset after detection
                    return True
            else:
                self.circle_detected_time = None

        return False

    def detect(self):
        # Inicjalizacja kamerki
        cap = cv2.VideoCapture(0)

        # Tworzenie okna
        cv2.namedWindow('Kontury')

        # Tworzenie trackbarów
        cv2.createTrackbar('Low H', 'Kontury', 0, 179, self.nothing)
        cv2.createTrackbar('High H', 'Kontury', 173, 179, self.nothing)
        cv2.createTrackbar('Low S', 'Kontury', 0, 255, self.nothing)
        cv2.createTrackbar('High S', 'Kontury', 93, 255, self.nothing)
        cv2.createTrackbar('Low V', 'Kontury', 40, 255, self.nothing)
        cv2.createTrackbar('High V', 'Kontury', 98, 255, self.nothing)

        while True:
            # Pobranie klatki z kamerki
            ret, frame = cap.read()
            if not ret:
                break

            # Konwersja obrazu do przestrzeni barw HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Odczytanie wartości z trackbarów
            low_h = cv2.getTrackbarPos('Low H', 'Kontury')
            high_h = cv2.getTrackbarPos('High H', 'Kontury')
            low_s = cv2.getTrackbarPos('Low S', 'Kontury')
            high_s = cv2.getTrackbarPos('High S', 'Kontury')
            low_v = cv2.getTrackbarPos('Low V', 'Kontury')
            high_v = cv2.getTrackbarPos('High V', 'Kontury')

            # Tworzenie maski na podstawie wartości HSV
            lower_bound = np.array([low_h, low_s, low_v])
            upper_bound = np.array([high_h, high_s, high_v])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Usuwanie szumów za pomocą rozmycia Gaussa
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Operacje morfologiczne
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Znajdowanie konturów
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            triangle_detected = False
            kwadr_detected = False
            kolo_detected = False

            for contour in contours:
                # Przybliżenie konturu do wielokąta
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Sprawdzenie powierzchni konturu
                self.area = cv2.contourArea(contour)
                if self.area < 100:
                    continue

                # Sprawdzenie kształtu konturu
                if len(approx) == 3:
                    # Rysowanie trójkąta
                    cv2.drawContours(frame, [approx], 0, (0, 255, 255), 2)
                    cv2.putText(frame, "triangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255))
                    triangle_detected = True
                elif len(approx) == 4:
                    # Rysowanie prostokąta/kwadratu
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "kwadrat", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                    kwadr_detected = True
                else:
                    # Sprawdzenie, czy kontur jest wystarczająco okrągły, aby był kołem
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    if abs(cv2.contourArea(contour) - (np.pi * radius * radius)) < 0.2 * (np.pi * radius * radius):
                        # Rysowanie koła
                        cv2.circle(frame, center, radius, (255, 0, 0), 2)
                        cv2.putText(frame, "circle", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                        kolo_detected = True

            # Check for detection duration for each shape
            if self.chose_shape == 'triangle':
                if self.check_detection_duration(triangle_detected, 'triangle'):
                    self.chose()  # Ask for the shape again
            elif self.chose_shape == 'square':
                if self.check_detection_duration(kwadr_detected, 'square'):
                    self.chose()  # Ask for the shape again
            elif self.chose_shape == 'circle':
                if self.check_detection_duration(kolo_detected, 'circle'):
                    self.chose()  # Ask for the shape again

            # Wyświetlanie wyników
            cv2.imshow('Kontury', frame)
            cv2.imshow('Maska', mask)

            # Wyjście z pętli po naciśnięciu klawisza 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Zwolnienie kamerki i zamknięcie wszystkich okien
        cap.release()
        cv2.destroyAllWindows()

    def __init__(self):
        self.area = 0
        self.triangle_detected_time = None
        self.square_detected_time = None
        self.circle_detected_time = None
        self.detection_duration = 2  # Duration in seconds
        self.stop_controller = Event()


        self.chose()  # Let the user choose the shape before starting detection
        self.detect()  # Start detection when an instance is created


# Create an instance of the ShapeDetector and run the detection
if __name__ == '__main__':
    ShapeDetector()