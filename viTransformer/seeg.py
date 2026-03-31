import  cv2
import numpy as np  
IMAGE_PATH = r"D:\test\1_1.jpg"
img = cv2.imread(IMAGE_PATH)
cv2.imshow("img", img )
cv2.waitKey(0)
