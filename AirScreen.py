import cv2
import mediapipe as mp
import numpy as np
import os
import math
import speech_recognition as sr
import threading

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 30)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

imgCanvas = np.zeros((height, width, 3), np.uint8)

folderPath = 'Header'
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]

header = overlayList[0]
drawColor = (0, 0, 255)
thickness = 20
tipIds = [4, 8, 12, 16, 20]
xp, yp = 0, 0

r = sr.Recognizer()
transcribed_text = ""

def speech_recognition_thread():
    global transcribed_text
    while True:
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                transcribed_text = text
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

threading.Thread(target=speech_recognition_thread, daemon=True).start()

try:
    with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(imageRGB)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    points = []
                    for lm in hand_landmarks.landmark:
                        points.append([int(lm.x * width), int(lm.y * height)])

                    if len(points) != 0:
                        x1, y1 = points[8] 
                        x2, y2 = points[12]
                        x3, y3 = points[4]   
                        x4, y4 = points[20] 

                        fingers = []
                        
                        if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                        for id in range(1, 5):
                            if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)

                        nonSel = [0, 3, 4] 
                        if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                            xp, yp = x1, y1

                            if y1 < 125:
                                if 170 < x1 < 295:
                                    header = overlayList[0]
                                    drawColor = (0, 0, 255)
                                elif 436 < x1 < 561:
                                    header = overlayList[1]
                                    drawColor = (255, 0, 0)
                                elif 700 < x1 < 825:
                                    header = overlayList[2]
                                    drawColor = (0, 255, 0)
                                elif 980 < x1 < 1105:
                                    header = overlayList[3]
                                    drawColor = (0, 0, 0)

                            cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                        nonDraw = [0, 2, 3, 4]
                        if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                            cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                            if xp == 0 and yp == 0:
                                xp, yp = x1, y1
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                            xp, yp = x1, y1

                        selecting = [1, 1, 0, 0, 0] 
                        setting = [1, 1, 0, 0, 1]   
                        if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(fingers[i] == j for i, j in zip(range(0, 5), setting)):
                            r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2) / 3)
                            x0, y0 = [(x1 + x3) / 2, (y1 + y3) / 2]
                            v1, v2 = [x1 - x3, y1 - y3]
                            v1, v2 = [-v2, v1]

                            mod_v = math.sqrt(v1**2 + v2**2)
                            v1, v2 = [v1 / mod_v, v2 / mod_v]
                            c = 3 + r
                            x0, y0 = [int(x0 - v1 * c), int(y0 - v2 * c)]
                            cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                            if fingers[4]:                        
                                thickness = r
                                cv2.putText(image, 'Check', (x4-25, y4-8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 1)

                            xp, yp = x1, y1

            # Merge canvas and image
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, imgInv)
            image = cv2.bitwise_or(image, imgCanvas)

            cv2.rectangle(image, (10, height - 90), (width - 10, height - 40), (0, 0, 0), -1)
            cv2.putText(image, transcribed_text, (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            image[0:125, 0:width] = header

            cv2.imshow('Air Canvas', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Program interrupted by user. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
