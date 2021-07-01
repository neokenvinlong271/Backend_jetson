import numpy as np
import cv2
from pygame import mixer
from pyzbar.pyzbar import decode
import file_extractor as fe
import restful
import model_loader
import json
import time

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return ("nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/6 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (capture_width, capture_height, framerate, flip_method, display_width, display_height,)
    )

def eye_predict(img, model):
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
    model = model_loader.load_model('dnn_model/EyeDet.model')

    face_cascade = cv2.CascadeClassifier("haar_model/lbpcascade_frontalface_improved.xml")
    left_eye_cascade = cv2.CascadeClassifier("haar_model/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("haar_model/haarcascade_righteye_2splits.xml")

    count = 0
    counter = -2
    mixer.init()
    command='activate'
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    times_drowsiness = 0
    times_drowsiness_start = 0
    
    #init cookies
    cookies=fe.file_extractor('tmp/cookies.txt')
    DEVICE_ID = cookies['deviceid']
    USERNAME = cookies['username']
    USER_ID = cookies['userid']
    TIME_ALERT = cookies['time']    

    #init login
    is_login = False
    try:
        response = restful.login('datpro7703@gmail.com','datpro7703@gmail.com')
        json_data = json.loads(response.content)
        bearer_token = json_data['data']['type'] + ' ' + json_data['data']['token']
        if len(bearer_token) > 50 and len(json_data['data']['userId']) > 0:
            USER_ID = json_data['data']['userId']
            is_login = True
            restful.upload_data_tracking(bearer_token)
    except Exception as e: # work on python 3.x
        pass

    #init model
    init_img = cv2.imread('tmp/123.jpg')
    eye_predict(init_img, model)

    #cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        cv2.namedWindow("Drowsiness driver Capstone 2021", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("Drowsiness driver Capstone 2021", 0) >= 0:
            ret, img = cap.read()
            ret = img
            #img = cv2.resize(img, (800, 480))
            height, width = img.shape[:2]

            counter += 1
            if counter >= 20:
                counter = 0
                for code in decode(img):
                    counter = -1
                    command = (code.data.decode('utf-8')).split('=')[1]

            if command == 'deactivated':
                if counter == -1:
                    sound = mixer.Sound('sound/deactivated.mp3')
                    sound.play()
                    while mixer.get_busy():
                        pass
            elif command == 'activate':
                if counter == -1:
                    sound = mixer.Sound('sound/activated.mp3')
                    sound.play()
                    while mixer.get_busy():
                        pass

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                count_eye = 0

                for (x, y, w, h) in faces:
                    h = int(h / 2) + 30
                    x -= 20
                    y -= 20
                    w += 40
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

                    roi_gray = gray[y: y + h, x: x + w]
                    roi_color = img[y: y + h, x: x + w]
                    l_eyes = left_eye_cascade.detectMultiScale(roi_gray)

                    for (ex, ey, ew, eh) in l_eyes:
                        detected_eye = True
                        count_eye += 1
                        eh += 10
                        ex -= 5
                        ey -= 5
                        ew += 10

                        crop_img = roi_color[ey: ey + eh, ex: ex + ew]
                        eye_status = eye_predict(crop_img, model)

                        if eye_status == "Closed": tmp = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
                        else: tmp = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

                        if eye_status == "Closed":
                            if count < 2:
                                count += 1
                        else: count = 0

                        if count_eye == 2: break

                        #s1 = 'C:\\Users\\Admin\\Desktop\\Eye Dataset\\tmp\\{}.jpg'.format(counter)
                        #counter = count+1
                        #cv2.imwrite(s1, crop_img)

                    if count_eye < 2:
                        r_eyes = right_eye_cascade.detectMultiScale(roi_gray)
                        for (ex, ey, ew, eh) in r_eyes:
                            eh += 10
                            ex -= 5
                            ey -= 5
                            ew += 10

                            count_eye += 1
                            crop_img = roi_color[ey: ey + eh, ex: ex + ew]

                            eye_status = eye_predict(crop_img, model)
                            if eye_status == "Closed": tmp = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
                            else: tmp = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

                            if eye_status == "Closed": 
                                if count < 2:
                                    count += 1
                            else: count = 0

                            if count_eye == 2: break

                if count == 2 and count_eye > 0:
                    if times_drowsiness_start == 0:
                        times_drowsiness_start = time.time()
                    elif time.time() - times_drowsiness_start >= float(TIME_ALERT):
                        if not mixer.get_busy():
                            sound = mixer.Sound('sound/alarm.mp3')
                            sound.play()
                        if time.time() * 1000 - times_drowsiness > 15000:
                            times_drowsiness = round(time.time() * 1000)
                            s1 = 'tmp\\detected\\{time}_{deviceid}_{userid}.jpg'.format(time=times_drowsiness, deviceid=DEVICE_ID, userid=USER_ID)
                            cv2.imwrite(s1, ret)                                              
                        
                elif count < 2 or count_eye == 0: 
                    times_drowsiness_start = time.time()
                    if mixer.get_busy():
                        sound.stop()
            
                if count < 2: cv2.putText(img, 'warning:' + str(count) + ' ,Time: ' + str(round(time.time() - times_drowsiness_start)) + ', Count_eye: ' + str(count_eye), (100, height - 20), font, 1, (0, 255, 0), 1,cv2.LINE_AA)
                else: cv2.putText(img, 'warning:' + str(count) + ' ,Time: ' + str(round(time.time() - times_drowsiness_start)) + ', Count_eye: ' + str(count_eye), (100, height - 20), font, 1, (0, 0, 255), 1,cv2.LINE_AA)

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
