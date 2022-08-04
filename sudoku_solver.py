import cv2
from cv2 import THRESH_BINARY
import numpy as np
img = cv2.imread("form4.jpeg")
img = cv2.resize(img,(500,500))
grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thres1 = cv2.threshold(grey_img,150,255,cv2.THRESH_BINARY) #simple thresholding
thres2 = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #adaptive thresholding
#cv2.imshow("Segmented image",thres1)
#cv2.imshow("Segmented image adaptive",thres2)
corners = cv2.goodFeaturesToTrack(thres2,20,0.01,250)   #the 2nd last parameter is the confidence in the detection of corner
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),5,(0,0,255),-1)
cv2.imshow("image",thres2)                                #the last parameter above is the euclidean distance between the 2 corners
cv2.imshow("Original_image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()