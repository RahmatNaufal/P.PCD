import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image = cv2.imread('Citra2.png')   
if image is None: 
    print("Gagal memuat gambar. Pastikan file citra2.png ada di folder yang sama") 
else:
    cv2.imshow('Show Image GUI', image)

cv2.waitKey(0)
cv2.destroyAllWindows()