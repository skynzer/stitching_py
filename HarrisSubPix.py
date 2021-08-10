import cv2 as cv
import numpy as np

class HarrisSubPix:

    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def Detect(self):
        corners1 = self.HarrisFeatureDetect(self.img1)
        corners2 = self.HarrisFeatureDetect(self.img2)
        return corners1, corners2

    def HarrisFeatureDetect(self, gray):
        dst = cv.cornerHarris(gray,2,3,0.04)
        dst = cv.dilate(dst,None)
        ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        return corners