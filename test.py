import cv2 as cv
import numpy as np


tree_path = './data/sunset.jpg'
sunset_path = './data/sunny.jpg'
tree_img = cv.imread(tree_path)
sunset_img = cv.imread(sunset_path)

tree_np = np.array(cv.resize(tree_img, (256, 256)))
sunset_np = np.array(cv.resize(sunset_img, np.shape(tree_np)[:2]))
print(np.shape(sunset_np))
mean_sunset = np.mean(sunset_np, axis=(0,1))
mean_tree = np.mean(tree_np, axis=(0,1))

std_sunset = np.std(sunset_np, axis=(0,1))
std_tree = np.std(tree_np, axis=(0,1))

adain = std_sunset * (tree_np - mean_tree) / std_tree + mean_sunset
# adain = np.where(adain > 255, 255, adain)
# adain = np.where(adain < 0 , 0, adain)
adain = adain/255 

tree_np = tree_np / 255
concat = np.concatenate((tree_np, sunset_np/255, adain), axis=1)

cv.imshow('test', concat)
cv.waitKey(3000)


# print(mean_sunset, mean_tree)
# print(std_sunset, std_tree)
# print(np.min(adain))
# print(np.shape(adain))
