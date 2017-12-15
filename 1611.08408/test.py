import cv2
import numpy as np

a = np.array([[[1,2,3]], [[1,2,3]]]).astype(np.uint8)
a = cv2.resize(a,(2,6),interpolation=cv2.INTER_LINEAR)

print a.shape, a