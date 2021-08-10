import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from Stitcher import Stitcher
from GenToFile import GenToFile

def Transform(imageA, imageB, H):
	result = cv.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
	result[0:imageB.shape[0], 0:int(imageB.shape[1]*(3/4))] = imageB[0:imageB.shape[0], 0:int(imageB.shape[1]*(3/4))]
	return result

start_addr = 1024
data_length = 9
i=0
x=0b1100

cimg1 = cv.imread("ph1.jpg")
cimg2 = cv.imread("ph2.jpg")
Homography = Stitcher(cimg1, cimg2)
(H, vis) = Homography.stitch(showMatches=False)

result = Transform(cimg2, cimg1, H)

data = np.array(H).ravel()
wr_str = GenToFile()

f = open("demofile2.txt", "w")

while i < data_length:
    HB, LB = divmod(start_addr+i,256)
    n_data = int(data[i])
    str = wr_str.GenStr(n_data, HB, LB)
    f.seek(0,2)
    f.write(str) 
    f.write("\n")
    i = i+1

cv.imshow("res",result)
cv.imwrite("res.jpg", result)
#plt.imshow(result), plt.show()

cv.waitKey(0)
    
cv.destroyAllWindows()
