from utils import get_img
import cv2 


img = get_img(8) 

cv2.imwrite("8.jpg", img)


swith = True

if swith:
    for i in range(10):
        img = get_img(i)
        cv2.imwrite(f"{i}.jpg", img)
	

