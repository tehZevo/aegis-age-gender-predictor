import os
import cv2
import numpy as np
from protopost import ProtoPost

from utils import b64_to_img
from download_models import download_models

download_models()

PORT = int(os.getenv("PORT", 80))

GENDER_PROTO = 'models/deploy_gender.prototxt'
GENDER_MODEL = 'models/gender_net.caffemodel'
AGE_PROTO = 'models/deploy_age.prototxt'
AGE_MODEL = 'models/age_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['male', 'female']
#age ranges based on https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender
AGE_INTERVALS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']

#load models
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_gender(face):
  #run model
  blob = cv2.dnn.blobFromImage(
    image=face, scalefactor=1.0, size=(227, 227),
    mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
  )
  gender_net.setInput(blob)
  preds = gender_net.forward()[0]

  #determine gender class
  gender_id = int(preds.argmax())
  #get confidence
  conf = float(preds[gender_id])
  #determine gender string
  gender = GENDER_LIST[gender_id]

  return {
    "gender": gender,
    "gender_confidence": conf,
    "gender_id": gender_id
  }

def get_age(face):
  #run model
  blob = cv2.dnn.blobFromImage(
    image=face, scalefactor=1.0, size=(227, 227),
    mean=MODEL_MEAN_VALUES, swapRB=False
  )
  age_net.setInput(blob)
  preds = age_net.forward()[0]

  #determine age class
  age_id = int(preds.argmax())
  #get confidence
  conf = float(preds[age_id])
  #determine age string
  age = AGE_INTERVALS[age_id]

  return {
    "age": age,
    "age_confidence": conf,
    "age_id": age_id
  }

def handler(data):
  img = b64_to_img(data)
  age = get_age(img)
  gender = get_gender(img)
  return {**age, **gender}

routes = {
  "": handler
}

ProtoPost(routes).start(PORT)
