import numpy as np
import cv2 as cv

class FindCoefs:

    MIN_MATCH_COUNT = 10;

    def __init__(self,coord1,coord2):
        self.coord1 = coord1
        self.coord2 = coord2

    #def __init__(self,good, kp1, kp2):
    #    self.good = good
    #    self.kp1 = kp1
    #    self.kp2 = kp2

    def FindUsablePoints(self):
        npoints = np.empty((1,0), int)
        min = 14881337
        max = 0
        for i in range(len(self.coord1)):
            sum = self.coord1[i][0]+self.coord1[i][1]
            if sum < min:
                min = sum
                cmin = i
            if sum > max:
                max = sum
                cmax = i
        npoints = np.append(npoints, cmin)
        npoints = np.append(npoints, cmax)
        for i in range(0,2):
            n = np.random.random_integers(0,len(self.coord1)-1)
            while n == cmax or n == cmin:
                n = np.random.random_integers(0,len(self.coord1)-1)
            npoints = np.append(npoints, n)
        return npoints

    def FindCoefs(self, npoints):
        A1 = np.empty((8,9), float)
        H = np.empty((3,3), float)
        for i in range (0,8):
            quot, rem = divmod(i,2)
            coord_o = np.concatenate((self.coord1[npoints[quot]], np.array([1])))
            coord_t = np.concatenate((self.coord2[npoints[quot]], np.array([1])))
            print(coord_o, " ", coord_t, " ", quot, " ", rem)
            if rem == 1:
                A1[i] = np.concatenate((np.array([0, 0, 0]), np.dot(-1, np.transpose(coord_o)), np.dot(coord_t[1], np.transpose(coord_o))))
            else:
                A1[i] = np.concatenate((np.transpose(coord_o), np.array([0, 0, 0]), np.dot(-1*coord_t[0], np.transpose(coord_o))))
        U,S,V = cv.SVDecomp(A1)
        c = V[len(V)-1]
        H[0] = c[0:3]
        H[1] = c[3:6]
        H[2] = c[6:9]
        return H

        