import numpy as np
import cv2
import os


root ='C:/Users/nngbao/Desktop/100ANDRO'
out = 'C:/Users/nngbao/Desktop/New folder (4)'
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

i = 0
for file in os.listdir(root):
	file = os.path.join(root, file)

	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		i += 1
		print(w, h)
		#img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		try:
			roi_color = img[y-int(0.1*h):y+h+int(0.1*h), x-int(0.1*w):x+w+int(0.1*w)]
			filename = str(i) + '.jpg'
			print(filename)
			cv2.imwrite(os.path.join(out, filename), roi_color)
		except:
			pass
		
	try:
		a = 0
		'''cv2.imshow('img',roi_color)
		cv2.waitKey(0)
		cv2.destroyAllWindows()'''
	except:
		pass