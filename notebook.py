import numpy as np
import cv2
import tflearn
import numpy as np
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops
from pygame import mixer
from pyzbar.pyzbar import decode

ops.reset_default_graph()
convnet = input_data(shape=[None, 50, 50, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.load('EyeDet.model')


def gstreamer_pipeline(
        capture_width=1920,
        capture_height=1080,
        display_width=1920,
        display_height=1080,
        framerate=30,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/6 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def eye_predict(img):
    try:
        img = cv2.resize(img, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        data = img.reshape(50, 50, 1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1:
            return 'Closed'
        else:
            return 'Open'
    except:
        return 'Open'


def face_detect():
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")
    # eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    left_eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")

    count = 0
    counter = 0
    mixer.init()
    sound = mixer.Sound('alarm.mp3')
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    #
    init_img = cv2.imread('tmp/123.jpg')
    eye_predict(init_img)

    # cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    counter = 0

    if cap.isOpened():
        cv2.namedWindow("Drowsiness driver Capstone 2021", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("Drowsiness driver Capstone 2021", 0) >= 0:
            ret, img = cap.read()
            # img = cv2.resize(img, (800, 480))
            height, width = img.shape[:2]

            counter += 1
            if counter >= 15:
                counter = 0
                for code in decode(img):
                    print(counter)
                    print(code.type)
                    print(code.data.decode('utf-8'))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                h = int(h / 2) + 30
                x -= 20
                y -= 20
                w += 40
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = gray[y: y + h, x: x + w]
                roi_color = img[y: y + h, x: x + w]
                # eyes = eye_cascade.detectMultiScale(roi_gray)
                l_eyes = left_eye_cascade.detectMultiScale(roi_gray)

                count_eye = 0
                for (ex, ey, ew, eh) in l_eyes:
                    count_eye += 1
                    eh += 10
                    ex -= 5
                    ey -= 5
                    ew += 10

                    crop_img = roi_color[ey: ey + eh, ex: ex + ew]

                    eye_status = eye_predict(crop_img)
                    if eye_status == "Closed":
                        tmp = cv2.rectangle(
                            roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2
                        )
                    else:
                        tmp = cv2.rectangle(
                            roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                        )

                    if eye_status == "Closed" and count < 90:
                        count += 5
                    elif count >= 5:
                        count -= 5

                    if count_eye == 2:
                        break

                    # s1 = 'C:\\Users\\Admin\\Desktop\\Eye Dataset\\tmp\\{}.jpg'.format(counter)
                    # counter = count+1
                    # cv2.imwrite(s1, crop_img)

                if count_eye < 2:
                    r_eyes = right_eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in r_eyes:
                        eh += 10
                        ex -= 5
                        ey -= 5
                        ew += 10

                        count_eye += 1
                        crop_img = roi_color[ey: ey + eh, ex: ex + ew]

                        eye_status = eye_predict(crop_img)
                        if eye_status == "Closed":
                            tmp = cv2.rectangle(
                                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2
                            )
                        else:
                            tmp = cv2.rectangle(
                                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                            )

                        if eye_status == "Closed" and count < 90:
                            count += 5
                        elif count >= 5:
                            count -= 5

                        if count_eye == 2:
                            break

            if count < 0:
                count = 0
            if count >= 70 and not mixer.get_busy():
                sound.play()
            elif count <= 20:
                sound.stop()

            if count < 70:
                cv2.putText(img, 'warning:' + str(count) + '%, counter: ' + str(counter), (100, height - 20), font, 1,
                            (0, 255, 0), 1,
                            cv2.LINE_AA)
            else:
                cv2.putText(img, 'warning:' + str(count) + '%', (100, height - 20), font, 1, (0, 0, 255), 1,
                            cv2.LINE_AA)

            cv2.imshow("Drowsiness driver Capstone 2021", img)
            key_code = cv2.waitKey(5) & 0xFF
            # Stop the program on the ESC key
            if key_code == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    face_detect()
