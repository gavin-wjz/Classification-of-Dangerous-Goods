#Preprocess.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np

# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)#得到HSV色系下的V亮度图像
    imgGrayscale = Sharp(imgGrayscale)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)#增强对比度

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)#高斯滤波，5*5高斯滤波算子

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)#图像自适应二值化
    '''
    adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
        第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
        第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
        第四个参数threshold_type  指取阈值类型：必须是下者之一  
                                     •  CV_THRESH_BINARY,
                            • CV_THRESH_BINARY_INV
         第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
        第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
    '''
    Matrix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, Matrix)  # 闭运算
    #imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, Matrix)  # 开运算
    return imgGrayscale, imgThresh
# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)#RGB色系转换为HSV色系 H色调 S饱和度 V亮度

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):         #最大化对比度

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)#顶帽运算
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)#黑帽运算

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function
def Sharp(img):         #锐化
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst=dst+img
    return dst