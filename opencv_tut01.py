import cv2

cap = cv2.VideoCapture(0)
five = []
while True:
    ret, img = cap.read()
    five.append(img)
    cv2.imshow("img", img)
    cv2.waitKey(500)
    if five.__len__() == 1000:
        break

i = 0
for img in five:
    cv2.imwrite("img-" + i++ + '.jpg')


cv2.destroyAllWindows()
cv2.VideoCapture(0).release()