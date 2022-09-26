
# Python program to illustrate 
# adaptive thresholding type on an image
       
# organizing imports 
import cv2 
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt
   
# path to input image is specified and  
# image is loaded with imread command 
root = Tk()
# hide root window
root.overrideredirect(True)
root.geometry('0x0+0+0')

# lift window to top level
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)

path = askopenfilename(title='Choose test image', \
                       initialdir='C:/Users/Hafizi/PycharmProjects/Cell Segmentation/1-data/all scales/')

image1 = cv2.imread(path) 
print('IMAGE FILE:', path.split('/')[-1])
   


