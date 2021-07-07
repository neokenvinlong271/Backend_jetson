import re

def file_extractor(file_path):
    with open(file_path, 'r') as f:
        matches = re.findall(r'^(.+)=(.*)$', f.read(), flags=re.M)
        d = dict(matches)
    return d

def makeFile(d, file_path) :
    # Save it to a file
    with open(file_path, "w") as file :
        for key in d:
            file.write("%s=%s\n" % (key,d[key]))
        return
