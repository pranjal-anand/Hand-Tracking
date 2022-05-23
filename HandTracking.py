import cv2
import mediapipe as mp
# to check frame rate
import time

# create video object --> video capture
cap = cv2.VideoCapture(0)

# create an object from class hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# draw lines between landmark points (21)
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    # this gives us our frame
    success, img = cap.read()

    # convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # send the RGB image to object(results)
    # because hands class only uses RGB images
    results = hands.process(imgRGB)

    # extract the information from the received object
    # results.multi_hand_landmarks --> can have multiple hands
    if results.multi_hand_landmarks:
        # for loop to check if we have multiple hands or not
        for handLms in results.multi_hand_landmarks:
            # handLms represents single hand
            for id_no, lm in enumerate(handLms.landmark):
                # each id has a corresponding landmark
                # lm --> landmark info gives x,y and z coordinates

                # height, width and channels of image
                h, w, c = img.shape

                # x, y values used to find location for  landmark on hand
                # but values are decimal places and location should be in pixels
                # lm.x, lm.y are giving a ratio of the image
                # multiply with width and height to get pixel value
                # cx, cy position of the center
                cx, cy = int(lm.x * w), int(lm.y * h)

            # not displaying the RGB image but original image BGR so img
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # frame rate
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    # display on the screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    # to run a webcam
    cv2.imshow("Image", img)
    cv2.waitKey(1)
