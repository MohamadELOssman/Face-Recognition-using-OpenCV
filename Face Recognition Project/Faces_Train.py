import os
import numpy as np
import cv2
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "training_images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cur_id = 0
label_id = {}
y_label = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root)
			#print(label, path)
			
			if not label in label_id:
				label_id[label] = cur_id
				cur_id += 1
			id_ = label_id[label]

			#print(label_id)
			#y_label.append(label)
			#x_train.append(path) # turn into a NUMPY arrray

			pil_image = Image.open(path).convert("L") # Grayscale
			size = (550 , 550)
			final_img = pil_image.resize(size, Image.ANTIALIAS)
			img_array = np.array(final_img, "uint8")
			#print(img_array)
			faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = img_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_label.append(id_)


#print(y_label)
#print(x_train)

with open("labels/face_labels.pickle", 'wb') as f:
	pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("recognizers/face_trainner.yml")