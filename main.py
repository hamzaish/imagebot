from imageai.Classification.Custom import CustomImageClassification

model = "/content/drive/Shareddrives/VoltEDGE Robotics/model_train/models/model_ex-094_acc-0.997927.h5"
prediction = CustomImageClassification()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(model)
prediction.setJsonPath("drive/Shareddrives/VoltEDGE Robotics/model_train/json/model_class.json")
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.classifyImage("dog.jpeg", result_count=1)

print(f"{predictions} : {probabilities}")