from os.path import exists
import os
import gdown

def download_models():
  os.makedirs("models", exist_ok=True)

  GENDER_PROTO_URL = "https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ"
  GENDER_PROTO = 'models/deploy_gender.prototxt'
  if not exists(GENDER_PROTO):
    gdown.download(GENDER_PROTO_URL, GENDER_PROTO)

  GENDER_MODEL_URL = "https://drive.google.com/uc?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP"
  GENDER_MODEL = 'models/gender_net.caffemodel'
  if not exists(GENDER_MODEL):
    gdown.download(GENDER_MODEL_URL, GENDER_MODEL)

  AGE_PROTO_URL = "https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW"
  AGE_PROTO = 'models/deploy_age.prototxt'
  if not exists(AGE_PROTO):
    gdown.download(AGE_PROTO_URL, AGE_PROTO)

  AGE_MODEL_URL = "https://drive.google.com/uc?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl"
  AGE_MODEL = 'models/age_net.caffemodel'
  if not exists(AGE_MODEL):
    gdown.download(AGE_MODEL_URL, AGE_MODEL)

if __name__ == "__main__":
  download_models()
