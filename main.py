import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from PIL import Image, ImageTk

img1 = cv.imread('img/1.png')
img2 = cv.imread('img/2.png')

if img1.shape ==img2.shape:
    print("The images are in same size and channel")
    diffrence= cv.subtract(img1,img2)
    b,g,r=cv.split(diffrence)

    if cv.countNonZero(b)==0 and   cv.countNonZero(g)==0 and cv.countNonZero(r)==0 :
        print("The images are same at all")
    else:
        print("The image not equal")

# Initiate SIFT detector
# sift = cv.SIFT_create()
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)



number_keypoints = 0
if len(kp1) <= len(kp2):
    number_keypoints = len(kp1)
else:
    number_keypoints = len(kp2)
print("Keypoints 1ST Image: " + str(len(kp1)))
print("Keypoints 2ND Image: " + str(len(kp2)))


# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print(type(img3))
img = Image.fromarray(img3, 'RGB')
img.save('my.png')
# img.show()
print(type(img))
print("GOOD Matches:", len(good))

print("matchs percentage: ", len(good) / number_keypoints * 100, "%")
percentage="matchs percentage: ", len(good) / number_keypoints * 100, "%";
# plt.imshow(img3),plt.show()



root = Tk()
canvas = Canvas(root, width = img.width+40, height = img.height+60)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("output.png"))
canvas.create_image(20, 20, anchor=NW, image=img)

var = StringVar()
label = Label( root, textvariable=var, relief=FLAT )
var.set(percentage)
label.pack()

var2 = StringVar()
label2 = Label( root, textvariable=var2, relief=FLAT )
var2.set("Keypoints 1ST Image (Left): " + str(len(kp1)))
label2.pack()

var3 = StringVar()
label3 = Label( root, textvariable=var3, relief=FLAT )
var3.set("Keypoints 2nd Image (Right): " + str(len(kp2)))
label3.pack()


root.mainloop()