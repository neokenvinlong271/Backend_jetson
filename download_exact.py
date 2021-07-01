from urllib.request import urlopen
from zipfile import ZipFile


def download_and_unzip(url, extract_to):
    zipresp = urlopen(url)
    tempzip = open(extract_to + "\\model.zip", "wb")
    tempzip.write(zipresp.read())
    tempzip.close()
    zf = ZipFile(extract_to + "\\model.zip")
    zf.extractall(path=extract_to)
    zf.close()

if __name__ == "__main__":
    url = "https://drive.google.com/u/1/uc?id=1jspsQ4BjGB668oVQImyL7KkW95jgbBip&export=download"
    extract_to = "\\Users\\Admin\\Desktop\\Eye Dataset\\tmp"
    download_and_unzip(url, extract_to)