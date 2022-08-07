import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib.cbook import deprecated
import json

class model_load:

  def __init__(self, path_models, speed, jsons, num_objects = 10):
    self.model = path_models
    self.loaded = False
    self.num_objects = num_objects
    self.json = jsons
    if speed == "fast":
      self.size = 100
    if speed == "medium":
      self.size = 160
    if speed == "slow":
      self.size = 224
    self.classes = json.load(open(jsons))
    self.collect = []
    model = tf.keras.applications.InceptionV3(input_shape=(self.size, self.size, 3), weights = self.model, classes = self.num_objects)
    self.collect.append(model)
    

  def classify(self, path_img, result_count = 3):
    classification_results = []
    classification_probabilities = []

    if type(path_img) is str:
      image_to_predict = tf.keras.preprocessing.image.load_img(path_img, target_size=(self.size, self.size))
      image_to_predict = tf.keras.preprocessing.image.img_to_array(image_to_predict, data_format="channels_last")
      image_to_predict = np.expand_dims(image_to_predict, axis=0)

      image_to_predict = tf.keras.applications.inception_v3.preprocess_input(image_to_predict)
    else:
      image_input = Image.fromarray(np.uint8(path_img))
      image_input = image_input.resize((self.size, self.size))
      image_input = np.expand_dims(image_input, axis=0)
      image_to_predict = image_input.copy()
      image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
      
    model = self.collect[0]
    prediction = model.predict(image_to_predict, steps=1)
    
    predictiondata = []
    
    for pred in prediction:
        top_indices = pred.argsort()[-result_count:][::-1]
        for i in top_indices:
          each_result = []
          each_result.append(self.classes[str(i)])
          each_result.append(pred[i])
          predictiondata.append(each_result)

    return predictiondata