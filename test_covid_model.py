# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

# Teste com uma contendo sinais de Covid 
print("[INFO] loading trained model for covid classification...")
model = tf.keras.models.load_model('covid_detector.model')
image = cv2.imread("covid_2/test/Covid/0100.jpeg")
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)
(covid, viralPneumonia) = model.predict(img)[0]
label = "Covid" if covid > viralPneumonia else "Viral Pneumonia"
color = (0, 255, 0) if label == "Covid" else (0, 0, 255)
label = "{}: {:.2f}%".format(label, max(covid, viralPneumonia) * 100)
cv2.putText(image, label, (100, 100 - 10),
cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()


# Teste com uma imagem correspontende a Pneumonia viral
print("[INFO] loading trained model for covid classification...")
model = tf.keras.models.load_model('covid_detector.model')
image = cv2.imread("covid_2/test/Viral-Pneumonia/0105.jpeg")
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)
(covid, viralPneumonia) = model.predict(img)[0]
label = "Covid" if covid > viralPneumonia else "Viral Pneumonia"
color = (0, 255, 0) if label == "Covid" else (0, 0, 255)
label = "{}: {:.2f}%".format(label, max(covid, viralPneumonia) * 100)
cv2.putText(image, label, (100, 100 - 10),
cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()