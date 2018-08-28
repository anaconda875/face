from PIL import Image
import os

def resize_image(root, size):
	for dir in os.listdir(root):
		dir = os.path.join(root, dir)
		if os.path.isdir(dir):
			for file in os.listdir(dir):
				file = os.path.join(dir, file)
				if file[-3:] == 'png' or file[-3:] == 'jpg' or file[-3:] == 'PNG' or file[-3:] == 'JPG':
					img = Image.open(file).resize(size, Image.ANTIALIAS)
					img.save(file[:-3] + 'jpg')
					
resize_image('C:\\work\\face\\face_new\\train', (96, 96))