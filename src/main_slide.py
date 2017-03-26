import cv2
import numpy as np
from scipy import ndimage
import pdb
import os
import copy

labels = [0,1,2,3,4,6,8,9]


def DecideOverlap(r1,r2):
	x1 = r1[0][0]
	y1 = r1[0][1]
	width1 = r1[1][0]
	height1 = r1[1][1]  
	
	x2 = r2[0][0]  
	y2 = r2[0][1]  
	width2 = r2[1][0]  
	height2 = r2[1][1] 
	endx = max(x1+width1,x2+width2)  
	startx = min(x1,x2)  
	width = width1+width2-(endx-startx)  
	endy = max(y1+height1,y2+height2)  
	starty = min(y1,y2)  
	height = height1+height2-(endy-starty)
	ratio = 0.
	Area = 0.
	Area1 = 0.
	Area2 = 0.
	
	if width<=0 or height<=0:
		return 0. 
	else:
		Area = width*height
		Area1 = width1*height1
		Area2 = width2*height2
		ratio = Area /(Area1+Area2-Area)  
		return ratio


def get_trained_model():
	trained_model = []
	for index in labels:
		file_name = "../train/"+ str(index) +".bmp"
		a =  cv2.imread(file_name,0)
		trained_model.append(a)
	return trained_model
	
def preprepare(pic):
	equ = cv2.equalizeHist(pic)
	equ = np.float32(equ)
	gauss = cv2.blur(equ,(5,5))
	return pic
	
def detect_pic(img_test,trained_model):
	maxrate = 0
	minrate = 1
	threshold = 0.95
	IUO = 0.5
	result = []
	for k in range(len(labels)):
		for scale in [1, 0.75, 0.5, 1.5, 2.0]:
			modle_shape = trained_model[k].shape
			modle_shape = (int(modle_shape[0]*scale),int(modle_shape[1]*scale))			
			img_train = cv2.resize(trained_model[k],modle_shape)
			trained_vec = np.float32(img_train.flatten())
			for i in range(img_test.shape[0] - modle_shape[0]):
				for j in range(img_test.shape[1] - modle_shape[1]):
					block = img_test[i:i+modle_shape[0],j:j+modle_shape[1]]
					block_vec = block.flatten()
					print len(block_vec)
					print len(trained_vec)
					rate = np.dot(block_vec,trained_vec)/np.linalg.norm(block_vec)/np.linalg.norm(trained_vec)
					if rate > maxrate:
						maxrate = rate
					if rate < minrate:
						minrate = rate	
					if(rate > threshold):
						flag = False
						for old_frame in result:
							rect1 = ((i,j),modle_shape)
							rect2 = (old_frame[1],old_frame[2])
							if DecideOverlap(rect1,rect2) > IUO:
								flag = True
								if rate > old_frame[0]:
									old_frame[0] = rate
									old_frame[1] = (i,j)
									old_frame[2] = modle_shape
								break
						if(not flag):
							result.append([rate,(i,j),modle_shape])
	return 	result					
	
	
if __name__ == "__main__":
	trained_model = get_trained_model()
	pic_name = "../test/1.bmp"
	img_test = cv2.imread(pic_name,0)
	img_gray = preprepare(img_test)
	rects = detect_pic(img_gray,trained_model)
	for rect in rects:
		cv2.rectangle(img_test,rect[1],(rect[1][0]+rect[2][0],rect[1][1]+rect[2][1]),(55,255,155),2)
	cv2.imshow("tst",img_test)
	cv2.waitKey()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	