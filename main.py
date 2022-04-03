import cv2
import time
import datetime

# Set up a video source
SOURCE = 0
recorder = cv2.VideoCapture(SOURCE)

# Detection features by OpenCV
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detection then record logic
detection = False
last_detected_time = None
POST_DETECTION_TIMER = 4
FACE_DETECTION_BOX_COLOR = (0, 255, 0)
FACE_DETECTION_BOX_BORDER = 2
FPS = 60

# Recording frame size
frame_size = (int(recorder.get(3)), int(recorder.get(4)))
# Video format
video_format = cv2.VideoWriter_fourcc("m", "p", "4", "v")

while True:
    # Frame will be the current frame
    x, frame = recorder.read()
    # Generate a greyscale frame, since Haarcascade only works on greyscale images
    greyscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The first number is the delay of sorts and the second is the accuracy of sorts
    detected_faces = face_detection.detectMultiScale(greyscale_image, 1.5, 6)
    # Draw the rectangles
    for (top_left_x, top_left_y, width, height) in detected_faces:
        cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), FACE_DETECTION_BOX_COLOR, FACE_DETECTION_BOX_BORDER)

    # When a person is detected, start recording
    if len(detected_faces) > 0:
        if detection:
            last_detected_time = None
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("Date_%d-%m-%Y-Time_%H-%M-%S")
            # Output stream for recording
            output_stream = cv2.VideoWriter(f"caughtIn4K-{current_time}.mp4", video_format, FPS, frame_size)
            print(f"Started recording @ {current_time}")
    # If we detected someone but they are gone for a while, start a timer to stop recording
    elif detection:
        if last_detected_time:
            if time.time() - last_detected_time >= POST_DETECTION_TIMER:
                detection = False
                output_stream.release()
                print("Stopped recording")
        else:
            last_detected_time = time.time()
    # If detected someone, record the frame
    if detection:
        output_stream.write(frame)
    # Shows the current frame with a window
    cv2.imshow(f"Video source {SOURCE}", frame)
    # Exit condition
    if (cv2.waitKey(1) == ord('p')):
        break

# Release your device
recorder.release()
output_stream.release()
# Destroy all windows associated with OpenCV
cv2.destroyAllWindows()