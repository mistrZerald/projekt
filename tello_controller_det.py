import numpy as np
from djitellopy import tello
from threading import Thread, Event
import keyboard
import csv
import time
import cv2
import matplotlib.pyplot as plt
#from ultralytics import YOLO
#from ultralytics.utils.plotting import Annotator
import asyncio


class TelloController:

    class TelloKillSwitch(Thread):

        tc_handler = None

        def __init__(self, tc_handler):
            Thread.__init__(self)
            self.tc_handler = tc_handler

        def run(self):
            keyboard.wait('space')
            self.tc_handler.force_emergency_stop()

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

    tello_drone = None
    stop_controller = None
    #color_name = "None"
    #
    # def detect_count(self):
    #
    #     if self.area > 2:
    #         if self.area > 1:
    #             self.MissionSequence.append("Blue")
    #             print("Added Blue")
    #         # elif self.color_name == "Green":
    #         #     self.MissionSequence.append("Green")
    #         #     print("Added Green")
    #         # elif self.color_name == "Red":
    #         #     self.MissionSequence.append("Red")
    #         #     print("Added Red")
    #     else:
    #         pass

    def nothing(x):
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

        frame_read = self.tello_drone.get_frame_read()

        while True:
            frame = frame_read.frame

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
                if self.area < 10000:
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

    def batt_warning(self):
        if self.tello_drone.get_battery() <= 20:
            print("Battery below 20%!")





    def force_emergency_stop(self):
        self.tello_drone.emergency()
        self.stop_controller.set()




    def __init__(self):

        #p = open('data.csv', 'w', newline='')
        #r = csv.writer(p, delimiter='-')
        #r.writerow(['Battery', 'Pitch', 'Yaw', 'Roll', 'Speed X', 'Speed Y', 'Speed Z', 'Acceleration X', 'Acceleration Y', 'Acceleration Z', 'Flight Time'])
        #p.close()

        self.kill_switch = self.TelloKillSwitch(self)
        self.kill_switch.start()

        self.stop_controller = Event()
        
        self.tello_drone = tello.Tello()
        self.tello_drone.connect()

        self.tello_drone.streamon()
        self.color_name = "None"
        self.area = 0
        self.MissionSequence = []

        #self.battery_check = self.TelloTimer(1, self.stop_controller, self.battery_check_func)
        #self.battery_check.start()

        #self.acc_check = self.TelloTimer(0.1, self.stop_controller, self.acc_check_func)
        #self.acc_check.start()

        #self.imu_check = self.TelloTimer(0.1, self.stop_controller, self.imu_check_func)
        #self.imu_check.start()

        #self.write_csv = self.TelloTimer(0.1, self.stop_controller, self.csv_write_func)
        #self.write_csv.start()

        #self.batt_warning = self.TelloTimer(1, self.stop_controller, self.batt_warning)
        #self.batt_warning.start()

        #self.take_picture()

        #self.video_recorder()

        #self.rpy_graph_func()

        #self.real_time_yaw_func()

        #self.onboard_camera_func()

        # self.lab_mission_func_count_colors = self.TelloTimer(3, self.stop_controller, self.lab_mission_func_count_colors)
        # self.lab_mission_func_count_colors.start()
        #
        # self.lab_mission_func()
        self.detect()


        #self.horizon_func()

        #self.mission_func()

        #self.mission_func_2()

        #self.project_mission_func()

        time.sleep(5)

        cv2.destroyAllWindows()
        self.stop_controller.set()
        self.tello_drone.end()


if __name__ == '__main__':
    tc = TelloController()
