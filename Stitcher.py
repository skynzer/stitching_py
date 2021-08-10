
import numpy as np
import imutils
import cv2 as cv

class Stitcher:
	def __init__(self, img1, img2):
		self.imageB = img1 
		self.imageA = img2

	def stitch(self, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		(kpsA, featuresA) = self.detectAndDescribe(self.imageA)
		(kpsB, featuresB) = self.detectAndDescribe(self.imageB)
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
		if M is None:
			return (None, None, None)
		else:
			(matches, H, status) = M
			if showMatches:
				vis = self.drawMatches(kpsA, kpsB, matches, status)
				return (H, vis)
			else:
				return (H, None)

	def detectAndDescribe(self, image):
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		descriptor = cv.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)
		kps = np.float32([kp.pt for kp in kps])
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
		matcher = cv.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		for m in rawMatches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		if len(matches) > 4:
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			(H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC,reprojThresh)
			return (matches, H, status)
		return None

	def drawMatches(self, kpsA, kpsB, matches, status):
		(hA, wA) = self.imageA.shape[:2]
		(hB, wB) = self.imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = self.imageA
		vis[0:hB, wA:] = self.imageB
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv.line(vis, ptA, ptB, (0, 255, 0), 1)
		return vis