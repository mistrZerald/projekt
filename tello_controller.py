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

    def battery_check_func(self):
        print("Battery: " + str(self.tello_drone.get_battery()) + "%")

    def air_time_func(self):
        print("Air time: " + str(self.tello_drone.get_flight_time()))

    def acc_check_func(self):
        print("Acceleration X axis: " + str(self.tello_drone.get_acceleration_x()))
        print("Acceleration Y axis: " + str(self.tello_drone.get_acceleration_y()))
        print("Acceleration Z axis: " + str(self.tello_drone.get_acceleration_z()))

    def imu_check_func(self):
        print("Pitch: " + str(self.tello_drone.get_pitch()))
        print("Yaw: " + str(self.tello_drone.get_yaw()))
        print("Roll: " + str(self.tello_drone.get_roll()))

    def batt_warning(self):
        if self.tello_drone.get_battery() <= 20:
            print("Battery below 20%!")

    def csv_write_func(self):
        o = open('data.csv', 'a', newline='')
        l = csv.writer(o, delimiter='-')
        l.writerow([self.tello_drone.get_battery(), self.tello_drone.get_pitch(), self.tello_drone.get_yaw(), self.tello_drone.get_roll(), self.tello_drone.get_speed_x(),
                    self.tello_drone.get_speed_y(), self.tello_drone.get_speed_z(), self.tello_drone.get_acceleration_x(), self.tello_drone.get_acceleration_y(), self.tello_drone.get_acceleration_z(),
                    self.tello_drone.get_flight_time()])
        o.close()

    def mission_func(self):
        self.tello_drone.takeoff()
        time.sleep(1)
        self.tello_drone.land()

    def mission_func_2(self):
        self.tello_drone.takeoff()
        time.sleep(1)
        self.tello_drone.rotate_clockwise(45)
        time.sleep(1)
        self.tello_drone.land()

    def take_picture(self):
        frame_read = self.tello_drone.get_frame_read()
        cv2.imwrite("picture.png", frame_read.frame)

    def onboard_camera_func(self):

        framex = self.tello_drone.get_frame_read()
        while True:
            frame = framex.frame
            battery_level = self.tello_drone.get_battery()
            if battery_level >= 60:
                color = (0, 255, 0)
            elif battery_level >= 35 and battery_level < 60:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.putText(frame, f"Battery level: {int(battery_level)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Tello Stream", frame)
            cv2.waitKey(1)

    def video_recorder(self):
        keepRecording = True
        frame_read = self.tello_drone.get_frame_read()

        def videoRecorder():
            height, width, _ = frame_read.frame.shape

            video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
            while keepRecording:
                video.write(frame_read.frame)
                time.sleep(1 / 60)

            video.release()

        recorder = Thread(target=videoRecorder)
        recorder.start()

        time.sleep(5.0)
        keepRecording = False

        recorder.join()

    def force_emergency_stop(self):
        self.tello_drone.emergency()
        self.stop_controller.set()

    def horizon_func(self):
        while True:
            roll = self.tello_drone.get_roll()
            pitch = self.tello_drone.get_pitch()
            altitude = self.tello_drone.get_height()

            horizon_image = np.zeros((600, 800, 3), dtype=np.uint8)
            horizon_image.fill(200)

            center_x, center_y = 400, 300
            radius = 250

            cv2.circle(horizon_image, (center_x, center_y), radius, (50, 50, 50), -1)

            pitch_offset = int(pitch * 2)

            sky_points = np.array([
                [center_x - radius, center_y - pitch_offset + radius],
                [center_x + radius, center_y - pitch_offset + radius],
                [center_x + radius, center_y - pitch_offset - radius],
                [center_x - radius, center_y - pitch_offset - radius]
            ])

            earth_points = np.array([
                [center_x - radius, center_y - pitch_offset + radius],
                [center_x + radius, center_y - pitch_offset + radius],
                [center_x + radius, center_y - pitch_offset + 2 * radius],
                [center_x - radius, center_y - pitch_offset + 2 * radius]
            ])

            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -roll, 1.0)
            sky_points = cv2.transform(np.array([sky_points]), rotation_matrix)[0]
            earth_points = cv2.transform(np.array([earth_points]), rotation_matrix)[0]

            cv2.fillPoly(horizon_image, [sky_points.astype(np.int32)], (255, 255, 255))
            cv2.fillPoly(horizon_image, [earth_points.astype(np.int32)], (150, 75, 0))

            cv2.circle(horizon_image, (center_x, center_y), radius, (0, 0, 0), 5)
            cv2.line(horizon_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 0), 5)

            cv2.line(horizon_image, (center_x, center_y - radius), (center_x, center_y + radius), (0, 0, 0),2)
            for i in range(-3, 4):
                y = center_y - i * 50
                cv2.line(horizon_image, (center_x - 20, y), (center_x + 20, y), (0, 0, 0), 2)

            cv2.putText(horizon_image, 'Attitude Indicator', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(horizon_image, f'Roll: {roll:.1f}°', (center_x - 60, center_y + 160), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 0), 2)
            cv2.putText(horizon_image, f'Pitch: {pitch:.1f}°', (center_x - 60, center_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(horizon_image, f'Altitude: {altitude:.1f}cm', (center_x - 60, center_y + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.imshow('Artificial Horizon', horizon_image)
            cv2.waitKey(1)

    def rpy_graph_func(self):
        x = []
        y = []
        y1 = []
        y2 = []

        plt.plot(x, y)
        plt.plot(x, y1)
        plt.plot(x, y2)

        cnt = 0
        while True:

            pitch = self.tello_drone.get_pitch()
            roll = self.tello_drone.get_roll()
            yaw = self.tello_drone.get_yaw()

            x.append(cnt)
            y.append(pitch)
            y1.append(roll)
            y2.append(yaw)

            plt.gca().lines[0].set_xdata(x)
            plt.gca().lines[1].set_xdata(x)
            plt.gca().lines[2].set_xdata(x)
            plt.gca().lines[0].set_ydata(y)
            plt.gca().lines[1].set_ydata(y1)
            plt.gca().lines[2].set_ydata(y2)
            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.033)

            cnt += 0.033

            if cnt > 5:
                del x[0]
                del y[0]
                del y1[0]
                del y2[0]

    def real_time_yaw_func(self):
        k = []
        j = []

        plt.plot(k, j)

        count = 0
        while True:
            Yaw = self.tello_drone.get_yaw()

            k.append(count)
            j.append(Yaw)

            plt.gca().lines[0].set_kdata(k)
            plt.gca().lines[0].set_jdata(j)
            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.033)

            count += 0.033

            if count > 5:
                del k[0]
                del j[0]

    def lab_mission_func_count_colors(self):

        if self.area > 10000:
            if self.color_name == "Blue":
                self.MissionSequence.append("Blue")
                print("Added Blue")
            elif self.color_name == "Green":
                self.MissionSequence.append("Green")
                print("Added Green")
            elif self.color_name == "Red":
                self.MissionSequence.append("Red")
                print("Added Red")
        else:
            pass

    def nothing(x):
        pass

    def process_color(self, frame, mask, color_name, color_bgr):
        self.color_name = color_name
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = 0
        for pic, contour in enumerate(contours):
            self.area = cv2.contourArea(contour)
            cnt += 0.033
            if self.area > 10000:
                SizeThresholdColor = 255
            else:
                SizeThresholdColor = 0

            if self.area > 4000:
                x, y, w, h = cv2.boundingRect(contour)
                height, width, _ = frame.shape
                Horizontal = x / width
                HorizontalColor = 255 * Horizontal

                # Get the mean color inside the contour
                mask_roi = mask[y:y + h, x:x + w]
                frame_roi = frame[y:y + h, x:x + w]
                mean_color = cv2.mean(frame_roi, mask=mask_roi)[:3]

                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame, f"{color_name} Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color_bgr, 1)
                cv2.putText(frame, f"Area: {self.area}", (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, SizeThresholdColor, 0), 1)
                cv2.putText(frame, f"Horizontal: {Horizontal:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, HorizontalColor), 1)
                # cv2.putText(frame, f"Mission Sequence: {MissionSequence[0]:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                # (255, 255, HorizontalColor), 1)
                # cv2.putText(frame, f"Color: {mean_color}", (x, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # if len.MissionSequence

    def lab_mission_func(self):

        frame_read = self.tello_drone.get_frame_read()

        # Create a window
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trackbars", 1000, 1000)  # Set the desired size
        
        # Create trackbars for color change
        cv2.createTrackbar("Blue Lower Hue", "Trackbars", 94, 180, self.nothing)
        cv2.createTrackbar("Blue Upper Hue", "Trackbars", 150, 180, self.nothing)
        cv2.createTrackbar("Blue Lower Sat", "Trackbars", 50, 255, self.nothing)
        cv2.createTrackbar("Blue Upper Sat", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Blue Lower Val", "Trackbars", 50, 255, self.nothing)
        cv2.createTrackbar("Blue Upper Val", "Trackbars", 255, 255, self.nothing)

        cv2.createTrackbar("Red Lower Hue 1", "Trackbars", 0, 180, self.nothing)
        cv2.createTrackbar("Red Upper Hue 1", "Trackbars", 4, 180, self.nothing)
        cv2.createTrackbar("Red Lower Hue 2", "Trackbars", 160, 180, self.nothing)
        cv2.createTrackbar("Red Upper Hue 2", "Trackbars", 180, 180, self.nothing)
        cv2.createTrackbar("Red Lower Sat", "Trackbars", 156, 255, self.nothing)
        cv2.createTrackbar("Red Upper Sat", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Red Lower Val", "Trackbars", 100, 255, self.nothing)
        cv2.createTrackbar("Red Upper Val", "Trackbars", 255, 255, self.nothing)

        cv2.createTrackbar("Green Lower Hue", "Trackbars", 45, 180, self.nothing)
        cv2.createTrackbar("Green Upper Hue", "Trackbars", 85, 180, self.nothing)
        cv2.createTrackbar("Green Lower Sat", "Trackbars", 50, 255, self.nothing)
        cv2.createTrackbar("Green Upper Sat", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Green Lower Val", "Trackbars", 50, 255, self.nothing)
        cv2.createTrackbar("Green Upper Val", "Trackbars", 255, 255, self.nothing)
        self.MissionSequence = []

        while True:
            frame = frame_read.frame

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))  # Convert tuple to list
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Get current positions of the trackbars
            blue_lower_hue = cv2.getTrackbarPos("Blue Lower Hue", "Trackbars")
            blue_upper_hue = cv2.getTrackbarPos("Blue Upper Hue", "Trackbars")
            blue_lower_sat = cv2.getTrackbarPos("Blue Lower Sat", "Trackbars")
            blue_upper_sat = cv2.getTrackbarPos("Blue Upper Sat", "Trackbars")
            blue_lower_val = cv2.getTrackbarPos("Blue Lower Val", "Trackbars")
            blue_upper_val = cv2.getTrackbarPos("Blue Upper Val", "Trackbars")

            red_lower_hue1 = cv2.getTrackbarPos("Red Lower Hue 1", "Trackbars")
            red_upper_hue1 = cv2.getTrackbarPos("Red Upper Hue 1", "Trackbars")
            red_lower_hue2 = cv2.getTrackbarPos("Red Lower Hue 2", "Trackbars")
            red_upper_hue2 = cv2.getTrackbarPos("Red Upper Hue 2", "Trackbars")
            red_lower_sat = cv2.getTrackbarPos("Red Lower Sat", "Trackbars")
            red_upper_sat = cv2.getTrackbarPos("Red Upper Sat", "Trackbars")
            red_lower_val = cv2.getTrackbarPos("Red Lower Val", "Trackbars")
            red_upper_val = cv2.getTrackbarPos("Red Upper Val", "Trackbars")

            green_lower_hue = cv2.getTrackbarPos("Green Lower Hue", "Trackbars")
            green_upper_hue = cv2.getTrackbarPos("Green Upper Hue", "Trackbars")
            green_lower_sat = cv2.getTrackbarPos("Green Lower Sat", "Trackbars")
            green_upper_sat = cv2.getTrackbarPos("Green Upper Sat", "Trackbars")
            green_lower_val = cv2.getTrackbarPos("Green Lower Val", "Trackbars")
            green_upper_val = cv2.getTrackbarPos("Green Upper Val", "Trackbars")

            # Define HSV range for blue, red, and green colors
            blue_lower = np.array([blue_lower_hue, blue_lower_sat, blue_lower_val], np.uint8)
            blue_upper = np.array([blue_upper_hue, blue_upper_sat, blue_upper_val], np.uint8)
            red_lower1 = np.array([red_lower_hue1, red_lower_sat, red_lower_val], np.uint8)
            red_upper1 = np.array([red_upper_hue1, red_upper_sat, red_upper_val], np.uint8)
            red_lower2 = np.array([red_lower_hue2, red_lower_sat, red_lower_val], np.uint8)
            red_upper2 = np.array([red_upper_hue2, red_upper_sat, red_upper_val], np.uint8)
            green_lower = np.array([green_lower_hue, green_lower_sat, green_lower_val], np.uint8)
            green_upper = np.array([green_upper_hue, green_upper_sat, green_upper_val], np.uint8)

            # Create masks for blue, red, and green colors
            blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
            red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

            # Dilate masks to fill in gaps
            kernel = np.ones((5, 5), "uint8")
            blue_mask = cv2.dilate(blue_mask, kernel)
            red_mask = cv2.dilate(red_mask, kernel)
            green_mask = cv2.dilate(green_mask, kernel)

            # Find contours and label them

            self.process_color(frame, blue_mask, "Blue", (255, 0, 0))
            self.process_color(frame, red_mask, "Red", (0, 0, 255))
            self.process_color(frame, green_mask, "Green", (0, 255, 0))



            cv2.imshow("Multiple Color Detection in Real-Time", frame)

            if len(self.MissionSequence) == 3:
                print("Executing Mission Sequence:", self.MissionSequence)
                # Perform the tasks based on the MissionSequence list
                for task in self.MissionSequence:
                    # Perform each task here based on the value in the MissionSequence list
                    print("Performing task:", task)
                break  # Exit the loop once the MissionSequence is complete

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def project_mission_func(self):
        frame_read = self.tello_drone.get_frame_read()

        while True:
            frame = frame_read.frame

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))  # Convert tuple to list
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            blue_lower = np.array([100, 50, 50], np.uint8)
            blue_upper = np.array([140, 255, 255], np.uint8)
            blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

            kernel = np.ones((5, 5), "uint8")

            blue_mask = cv2.dilate(blue_mask, kernel)
            res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

            contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if area > 10000:
                    SizeThresholdColor = 255
                else:
                    SizeThresholdColor = 0

                if area > 4000:
                    x, y, w, h = cv2.boundingRect(contour)
                    height, width, _ = frame.shape
                    Horizontal = x / width
                    HorizontalColor = 255*Horizontal

                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                    cv2.putText(frame, str(area), (x, y+75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, SizeThresholdColor, 0))
                    cv2.putText(frame, str(Horizontal), (x, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, HorizontalColor))


                    if area > 10000:

                        if (x+(w/2)) / int(width) < 0.45:
                            continue
                            #HorizontalColor =
                        elif (x+(w/2)) / int(width) > 0.55:
                            continue
                        else:
                            print('On point')
                    else:
                        SizeThresholdColor = 0
                    continue



            cv2.imshow("Multiple Color Detection in Real-Time", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


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

        self.lab_mission_func_count_colors = self.TelloTimer(3, self.stop_controller, self.lab_mission_func_count_colors)
        self.lab_mission_func_count_colors.start()

        self.lab_mission_func()

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
