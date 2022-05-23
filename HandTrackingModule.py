# Creating a hand Tracking module for future use
import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        # self.lmList = []

    # detection part
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    # Here we are getting all the different points of our hand
    def findPosition(self, img, handNo=0, draw=True):
        # For the bounding box we need to find the maximum x&y and minimum x&y of any of the points
        xList = []  # This list will contain all the values of x
        yList = []  # This list will contain all the values of y
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                xList.append(cx)
                yList.append(cy)

                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    # method to find which fingers are up
    def fingersUp(self):
        fingers = []
        self.tipIds = [4, 8, 12, 16, 20]
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    # method to find distance between 2 values
    def findDistance(self, p1, p2, img, draw=True):

        # x and y coordinates of centre points of landmark 4
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        # x and y coordinates of centre points of landmark 8
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]

        # Get the centre of the line
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Create a circle around thumb_tip
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            # Create a circle around thumb_tip
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)

            # Create a line between the thumb_tip and index_finger_tip
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Create a circle on the centre of the line
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        # print(lmList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
        main()
