
import numpy as np
import imutils
import cv2 as cv

class Stitcher:
	def __init__(self, img1, img2):
		self.imageB = img1 
		self.imageA = img2

	def stitch(self, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(kpsA, featuresA) = self.detectAndDescribe(self.imageA)
		(kpsB, featuresB) = self.detectAndDescribe(self.imageB)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None
		else:
			(matches, H, status) = M
			result = self.Transform(H)
			if showMatches:
				vis = self.drawMatches(kpsA, kpsB, matches, status)
				return (result, vis)
			else:
				return (result)

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		# detect and extract features from the image
		descriptor = cv.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)
		# otherwise, we are using OpenCV 2.4.X
		kps = np.float32([kp.pt for kp in kps])
		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# compute the homography between the two sets of points
			(H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC,
				reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = self.imageA.shape[:2]
		(hB, wB) = self.imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = self.imageA
		vis[0:hB, wA:] = self.imageB
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv.line(vis, ptA, ptB, (0, 255, 0), 1)
		# return the visualization
		return vis

	def Transform(self, H):
		result = cv.warpPerspective(self.imageA, H,(self.imageA.shape[1] + self.imageB.shape[1], self.imageA.shape[0]))
		result[0:self.imageB.shape[0], 0:self.imageB.shape[1]] = self.imageB
		# return the stitched image
		return result