import cv2
import numpy as np
from scipy import ndimage
import pdb
import os

def binary_pic(pic):
	kernel_1=np.uint8(np.zeros((5,5)))  
	for x in range(5):  
		kernel_1[x,2]=1;  
		kernel_1[2,x]=1; 
		kernel_1[1,1]=1; 
		kernel_1[1,3]=1; 
		kernel_1[3,1]=1; 
		kernel_1[3,3]=1; 
	
	kernel_2=np.uint8(np.zeros((3,3)))  
	for x in range(3):  
		kernel_2[x,1]=1;  
		kernel_2[1,x]=1; 
	
	equ = cv2.equalizeHist(255 - pic)
	gauss = cv2.GaussianBlur(equ,(9,9),0)
	th2 = cv2.adaptiveThreshold(gauss,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
	erode = cv2.erode(th2,kernel_1)
	dilate = cv2.dilate(erode,kernel_2)
	
	return dilate
	
def preprepare(pic):
	equ = cv2.equalizeHist(pic)
	gauss = cv2.GaussianBlur(equ,(9,9),0)
	
	return 255 - gauss

def correlation(pic,model):
	pic_shape = pic.shape
	result = 0.
	for i in range(pic_shape[0]):
		for j in range(pic_shape[1]):
			result = result + pic[i][j] * model[i][j]
	return result

	
def slide_comp(pic,model,pt):
	model = cv2.imresize(model,(pt,pt))
	pic_shape = pic.shape
	model_shape = (pt,pt)
	slide_shape = pic_shape - model_shape + (1,1)
	result_window = numpy.zeros(slide_shape,dtype = 'float32')
	for x_index in xrange(slide_shape[0]):
		for y_index in xrange (slide_shape[1]):
			result_window[x_index][y_index] = correlation(pic[x_index:x_index + pt,y_index : y_index + pt],model)
	return result_window
	
def get_trained_model():
	trained_model = []
	for index in range(0,10):
		file_name = "../train/"+ str(index) +".bmp"
		if os.path.exists(file_name):
			a = cv2.imread("../train/"+ str(index) +".bmp",0)
			a = binary_pic(a)
			a_pair = [index,a]
			trained_model.append(a_pair)
	return trained_model
	
def detect_pic(pic):
		pic = preprepare(pic)
		
		
		return 0
	

if __name__ == "__main__":
	trained_model = get_trained_model()
	pic_name = "../test/1.bmp"
	pic = cv2.imread(pic_name,0)
	pic = preprepare(pic)
	er = slide_comp(pic,trained_model[0][1],30)	
	cv2.imshow("test",pic)
	pdb.set_trace()
	