import json
import requests
from requests.models import Response
import re
import os

def login(username, password):
    api_url = "https://dhdev-drowsiness123.herokuapp.com/api/auth/login"
    json_data = {
        'username':username,
        'password':password
    }
    headers = {
        'Content-type':'application/json', 
        'Accept':'application/json'
    }
    
    response = requests.post(api_url, json=json_data, headers=headers)
    return response

def connect_user_in_device(user_id, device_id, bearer_token):
    api_url = "https://dhdev-drowsiness123.herokuapp.com/api/v1/user-devices"
    json_data = {
        'deviceId':device_id,
        'userId':user_id
    }
    headers = {
        'Authorization':bearer_token,
        'Content-type':'application/json', 
        'Accept':'application/json'
    }
    
    response = requests.post(api_url, json=json_data, headers=headers)
    return response

def create_tracking_with_image(time, user_id, device_id, image_path, bearer_token):
    api_url = "https://dhdev-drowsiness123.herokuapp.com/api/v1/data-trackings/users/devices/image"
    files = {'file': open(image_path,'rb')}

    headers = {
        'Authorization':bearer_token,
        'Accept':'application/json'
    }
    json_data = {
        'deviceId':device_id,
        'userId':user_id,
        'trackingAt':time
    }
    
    response = requests.post(api_url, data=json_data , files=files, headers=headers)
    return response

def upload_data_tracking(bearer_token):
    try:
        dir_path = 'tmp//detected//'
        files = os.listdir(dir_path)

        for file in files:
            if os.path.isfile(os.path.join(dir_path, file)):
                f = open(os.path.join(dir_path, file),'r')
                f_name = f.name.removeprefix('tmp//detected//').removesuffix('.jpg')
                f_name = re.split('_', f_name)

                time = f_name[0]
                device_id = f_name[1]
                user_id = f_name[2]

                create_tracking_with_image(time, user_id, device_id, f.name, bearer_token)
                #response = create_tracking_with_image(time, user_id, device_id, f.name, bearer_token)
                #print(response.content)
                #print(response.status_code)

                f.close()
                os.remove(f.name)
    except:
        pass

if __name__ == "__main__":
    DEVICE_ID = 'e83b4b73-6ba1-414e-b7ec-60f5ce7253ec'
    response = login('datpro7703@gmail.com','datpro7703@gmail.com')

    json_data = json.loads(response.content)
    bearer_token = json_data['data']['type'] + ' ' + json_data['data']['token']
    user_id = json_data['data']['userId']

    #response = connect_user_in_device(user_id, DEVICE_ID, bearer_token)
    #response = create_tracking_with_image(1625115802344, user_id, DEVICE_ID, r'C:\Users\DAT\Desktop\Capstone\Eye Dataset\Test_Data2\display.png', bearer_token)
    upload_data_tracking(bearer_token)

#response.status_code
#response.content
#json_data = json.loads(response.content)
#print(json_data['data']['username'])
