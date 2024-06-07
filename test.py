import cv2
import numpy as np
import time

class ShapeDetector:
    def __init__(self):
        self.area = 0

    def nothing(self, x):
        pass

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

        triangle_detected_time = None
        detection_duration = 2  # Duration in seconds

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
                    kolo_detected = True
                    if abs(cv2.contourArea(contour) - (np.pi * radius * radius)) < 0.2 * (np.pi * radius * radius):
                        # Rysowanie koła
                        cv2.circle(frame, center, radius, (255, 0, 0), 2)

            # Check for triangle detection duration
            if triangle_detected:
                if triangle_detected_time is None:
                    triangle_detected_time = time.time()
                elif time.time() - triangle_detected_time >= detection_duration:
                    print("hura")
                    triangle_detected_time = None  # Reset after detection
            elif kwadr_detected:
                if triangle_detected_time is None:
                    triangle_detected_time = time.time()
                elif time.time() - triangle_detected_time >= detection_duration:
                    print("hura")
                    triangle_detected_time = None  # Reset after detection
            elif kolo_detected:
                if triangle_detected_time is None:
                    triangle_detected_time = time.time()
                elif time.time() - triangle_detected_time >= detection_duration:
                    print("hura")
                    triangle_detected_time = None  # Reset after detection
            else:
                triangle_detected_time = None

            # Wyświetlanie wyników
            cv2.imshow('Kontury', frame)
            cv2.imshow('Maska', mask)

            # Wyjście z pętli po naciśnięciu klawisza 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Zwolnienie kamerki i zamknięcie wszystkich okien
        cap.release()
        cv2.destroyAllWindows()

# Create an instance of the ShapeDetector and run the detection
shape_detector = ShapeDetector()
shape_detector.detect()
