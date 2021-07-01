import os
import re


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

        f.close()
        os.remove(f.name)