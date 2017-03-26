import cv2
import numpy as np
from scipy import ndimage
import pdb
import os
import copy


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
	
	result = -np.float32(np.ones(dilate.shape))
	result = result + 3*(dilate > 128)
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
			result = result + float(pic[i][j]) *  float(model[i][j])
	return result

	
def slide_comp(pic,model,pt):
	model = cv2.resize(model,(pt,pt))
	pic_shape = pic.shape
	model_shape = (pt,pt)
	slide_shape = (pic_shape[0] - model_shape[0] + 1, pic_shape[1] - model_shape[1] + 1)
	result_window = np.zeros(slide_shape,dtype = 'float32')
	for x_index in xrange(slide_shape[0]):
		for y_index in xrange (slide_shape[1]):
			result_window[x_index][y_index] = correlation(pic[x_index:x_index + pt,y_index : y_index + pt],model)
	return result_window/(pt*pt)
	
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
	
def detect_pic(pic,trained_model):
		pic = preprepare(pic)
		pts = []
		pt = 10.
		
		strong_map = np.zeros(pic.shape,dtype = 'float32')
		index_map = -np.ones(pic.shape,dtype = 'float32')
		pt_map = np.zeros(pic.shape,dtype = 'float32')
		
		while pt <= 30:
			pts.append(int(pt))
			pt = pt * 1.2
		for pt_2 in pts:
			print pt_2
			for index in trained_model:
				print index[0]
				sub_result = slide_comp(pic,index[1],2*pt_2)
				for x in range(sub_result.shape[0]):
					for y in range(sub_result.shape[1]):
						if(sub_result[x][y] > strong_map[x+pt_2][y+pt_2]):
							strong_map[x+pt_2][y+pt_2] = sub_result[x][y]
							index_map[x+pt_2][y+pt_2] = index[0]
							pt_map[x+pt_2][y+pt_2] = 2*pt_2
		
		strong_map_new = copy.deepcopy(strong_map)
		for x in range(strong_map.shape[0]):
			for y in range(strong_map.shape[1]):
				if(strong_map[x][y] > 0):
					for x_10 in xrange(-10,10):
						for y_10 in xrange(-10,10):
							if strong_map[x+x_10][y+y_10] > strong_map[x][y]:
								strong_map_new[x][y] = 0;
		
		numhit = []
		
		for x in range(strong_map_new.shape[0]):
			for y in range(strong_map_new.shape[1]):
				if(strong_map_new[x][y] > 0):
					numhit.append(((x,y),strong_map[x][y],index_map[x][y],pt_map[x][y]))
		
		return numhit
	

if __name__ == "__main__":
	trained_model = get_trained_model()
	pic_name = "../test/1.bmp"
	pic = cv2.imread(pic_name,0)
	numhit = detect_pic(pic,trained_model)
	for n in numhit:
		cv2.rectangle(pic,(int(n[0][0]-n[3]),int(n[0][1]-n[3])),(int(n[0][0]+n[3]),int(n[0][1]+n[3])),(55,255,155),2)
	cv2.imshow("tst",pic)
	cv2.waitKey()
	