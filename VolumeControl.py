import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
# pycaw, this library allows to change the volume of computer
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
wCam, hCam = 640, 480
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
previousTime = 0

detector = htm.HandDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

while True:
    # check the success of the capture then read it
    success, img = cap.read()

    ############################################################
    # Find Hand
    ############################################################
    img = detector.findHands(img)
    # Getting the position
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        ############################################################
        # Filter the positions based on size
        ############################################################
        # This is done bcz when we are really far away from the camera
        # and measuring the distance between the thumb_tip and the index_finger_tip
        # it will give very small values which will not be usable
        # To resolve this issue we need a bounding box of the hand
        widthB, heightB = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        area = widthB * heightB//100
        # print(area)
        if 250 < area < 1000:

            ############################################################
            # Find Distance between index and Thumb
            ############################################################
            ## We need the value number 4 (thumb_tip) and 8 (index_finger_tip)
            # print(lmList[4], lmList[8])

            ## x and y coordinates of centre points of landmark 4
            # x1, y1 = lmList[4][1], lmList[4][2]
            ## x and y coordinates of centre points of landmark 8
            # x2, y2 = lmList [8][1], lmList[8][2]

            ## Get the centre of the line
            # cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            ## Create a circle around thumb_tip
            # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            ## Create a circle around thumb_tip
            # cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            ## Create a line between the thumb_tip and index_finger_tip
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            ## Create a circle on the centre of the line
            # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # length = math.hypot(x2 - x1, y2 - y1)
            ## print(length)

            ## Hand range 50 - 300
            ## Volume Range -65 - 0

            # Create a method so that with just the input of two values
            # we can get the distance between those two landmarks
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)

            #########################################################################
            # Convert hand range(length) into volume range(volume of the system)
            #########################################################################
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            #########################################################################
            # Reduce Resolution to make it smoother
            #########################################################################
            smoothness = 10
            volPer = smoothness * round(volPer / smoothness)

            #########################################################################
            # Check fingers up
            #########################################################################
            # change the volume with thumb n index finger but set the volume with pinky finger
            fingers = detector.fingersUp()
            # print(fingers)

            #########################################################################
            # If pinky is down set volume
            #########################################################################
            # To make the system smoother we will use the pinky finger to set the final volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 5, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

            #########################################################################
            # Drawings to show the volume bar on the side to see what is the volume at any given time
            #########################################################################
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)
            currentVol = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(img, f'Vol Set: {int(currentVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, colorVol, 3)

    #########################################################################
    # Frame rate
    #########################################################################
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)