import cv2
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

roi_top_left = (100, 100)
roi_bottom_right = (500, 500)
roi_color_inside = (0, 255, 0)
roi_color_outside = (0, 0, 255)

outside_start_time = None
time_outside = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

    person_inside = False
    for (x, y, w, h) in boxes:
        person_center = (x + w // 2, y + h // 2)
        if (roi_top_left[0] < person_center[0] < roi_bottom_right[0] and
                roi_top_left[1] < person_center[1] < roi_bottom_right[1]):
            person_inside = True
            break

    if person_inside:
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, roi_color_inside, 2)
        if outside_start_time is not None:
            time_outside += time.time() - outside_start_time
            outside_start_time = None
    else:
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, roi_color_outside, 2)
        if outside_start_time is None:
            outside_start_time = time.time()

    cv2.putText(frame, f'Time outside: {time_outside:.2f} s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
