#DetectPossibleknife.py
# -*- coding: utf-8 -*-
import Preprocess
import cv2
import numpy as np
import Possibleknife
import os

#原始图像缩放尺寸
size=1024 #原图像等比例缩放，方便后续计算
MIN_PIXEL_AREA=10000 #待检测物的外接矩阵最小面积
pre_savename = 'H:\\picture1'
showSteps=False
def detectknifeInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # this will be the return value
    imgScene=image_resize(imgOriginalScene)
    height, width, numChannels = imgScene.shape
    print('height:',height,'width',width,'numChannels',numChannels)
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()     #关闭所有窗口

    if showSteps == True: # show steps #######################################################
        cv2.imshow("0", imgScene)
    # end if # show steps #########################################################################
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgScene)  # preprocess to get grayscale and threshold images

    if showSteps == True:  # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)  # HSV色系下V亮度图像,图像为二维图像
        cv2.imshow("1b", imgThreshScene)  # 经过对比度增强后的二值图像，图像为二维图像
    # end if # show steps #########################################################################
    listcolorscene=detectcolor(imgScene)
    print('listcolorscene',len(listcolorscene))
    listpossibleknife=findPossibleknifesInScene(imgThreshScene)
    print('listpossibleknife',len(listpossibleknife))
    list=checkscene(listcolorscene,listpossibleknife)
    if showSteps == True: # show steps #######################################################
        print ("step 2 - intCountOfPossibleknifes = ",len(list))       # 列出可能有目标检测物的局部图片的个数
        for knife in list:
            cv2.rectangle(imgThreshScene, (knife.intBoundingRectX,knife.intBoundingRectY),(knife.intBoundingRectX +knife.intBoundingRectWidth ,knife.intBoundingRectY+knife.intBoundingRectHeight), (255, 255, 255), 2)
            print(knife.intBoundingRectX,knife.intBoundingRectY)
        cv2.imshow("2a", imgThreshScene)
        # end for
    # end if # show steps #########################################################################
    #save_new_image(imgScene,list)
    return imgScene,list




def image_resize(imgOriginalScene):
    newsize = (size, size)
    # 获取图像尺寸
    (h, w) = imgOriginalScene.shape[:2]
    if size is None:
        return imgOriginalScene
    # 高度算缩放比例
    else:
        if h<w:
            n = size / float(w)
            newsize = (size, int(h * n))
        else:
            n=size/float(h)
            newsize=(int(w*n),size)

    # 缩放图像
    newimage = cv2.resize(imgOriginalScene,newsize, interpolation=cv2.INTER_LINEAR)
    return newimage

def findPossibleknifesInScene(imgThresh):
    listOfPossibleknifes = []                # this will be the return value
    imgContours, contours, npaHierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # find all contours
    '''
    第一个参数是寻找轮廓的图像；

    第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
    cv2.RETR_EXTERNAL表示只检测外轮廓
    cv2.RETR_LIST检测的轮廓不建立等级关系
    cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE建立一个等级树结构的轮廓。

    第三个参数method为轮廓的近似办法
    cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    返回值 contours为一个列表，每一个元素代表一个轮廓，npaHierarchy为各轮廓间关系列表
    '''
    for i in range(0,len(contours)):  # for each contour
        possibleknife = Possibleknife.Possibleknife(contours[i])
        if checkPossibleknife(listOfPossibleknifes,possibleknife):                   # 检查去除一些不可能存在目标检测物的轮廓和重复包含轮廓
            listOfPossibleknifes.append(possibleknife)
    return listOfPossibleknifes
# end function

def checkPossibleknife(listOfPossibleknifes,possibleknife):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    sign=True
    dellist=[]
    for knife in listOfPossibleknifes:
        if (possibleknife.intBoundingRectArea<MIN_PIXEL_AREA ):
            sign=False
            break
        if (possibleknife.intBoundingRectX < knife.intBoundingRectX and
            possibleknife.intBoundingRectY < knife.intBoundingRectY and
            (possibleknife.intBoundingRectY + possibleknife.intBoundingRectHeight) > (knife.intBoundingRectY + knife.intBoundingRectHeight)and
            (possibleknife.intBoundingRectX + possibleknife.intBoundingRectWidth) > (knife.intBoundingRectX + knife.intBoundingRectWidth)):
            dellist.append(knife)
            sign=2
    if sign==2:
        for knife in dellist:
            listOfPossibleknifes.remove(knife)
        listOfPossibleknifes.append(possibleknife)
        sign=False
    return sign
        # end if

# end function

def detectcolor(img):
    listcolordetect=[]
    h,w=img.shape[:2]
    img_area=h*w
    lower_sliver = np.array([0, 0, 150])  # 银色
    upper_sliver = np.array([180, 80, 255])
    lower_black = np.array([0, 0, 0])  # 黑色
    upper_black = np.array([180, 255, 46])
    lower_blue = np.array([100, 43, 46])  # 蓝色
    upper_blue = np.array([124, 255, 255])
    lower_green = np.array([35, 43, 46])  # 青绿色
    upper_green = np.array([99, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv",hsv)
    mask_sliver = cv2.inRange(hsv, lower_sliver,upper_sliver)  # 就是将低于lower_sliver和高于upper_sliver的部分分别变成0，lower_red～upper_red之间的值变成255
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)
    mask_green=cv2.inRange(hsv,lower_green,upper_green)

    out_sliver= cv2.bitwise_and(hsv, hsv, mask=mask_sliver)  # 把原图中的非目标颜色区域去掉剩下的图像
    out_black=cv2.bitwise_and(hsv,hsv,mask_black)
    out_blue=cv2.bitwise_and(hsv,hsv,mask_blue)
    out_green=cv2.bitwise_and(hsv,hsv,mask_green)

    out_sliver = cv2.cvtColor(out_sliver, cv2.COLOR_BGR2GRAY)  # 图像变为二维
    out_black = cv2.cvtColor(out_black, cv2.COLOR_BGR2GRAY)  # 图像变为二维
    out_blue=cv2.cvtColor(out_blue,cv2.COLOR_RGB2GRAY)
    out_green=cv2.cvtColor(out_green,cv2.COLOR_RGB2GRAY)

    Matrix = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # Matrix = np.ones((10, 10), np.uint8)

    out_sliver = cv2.morphologyEx(out_sliver, cv2.MORPH_CLOSE, Matrix)  # 闭运算
    out_black = cv2.morphologyEx(out_black, cv2.MORPH_CLOSE, Matrix)  # 闭运算
    out_blue=cv2.morphologyEx(out_blue,cv2.MORPH_CLOSE,Matrix)
    out_green=cv2.morphologyEx(out_green,cv2.MORPH_CLOSE,Matrix)

    imgContour, contours, npaHierarchy = cv2.findContours(out_sliver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):  # for each contour
        colorscene = Possibleknife.Possibleknife(contours[i])
        if colorscene.intBoundingRectArea==img_area:
            continue
        if checkPossibleknife(listcolordetect,colorscene):                   # 检查去除一些不可能存在目标检测物的轮廓和重复包含轮廓
            listcolordetect.append(colorscene)

    imgContour, contours, npaHierarchy = cv2.findContours(out_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):  # for each contour
        colorscene = Possibleknife.Possibleknife(contours[i])
        if colorscene.intBoundingRectArea==img_area:
            continue
        if checkPossibleknife(listcolordetect, colorscene):  # 检查去除一些不可能存在目标检测物的轮廓和重复包含轮廓
            listcolordetect.append(colorscene)
    imgContour, contours, npaHierarchy = cv2.findContours(out_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):  # for each contour
        colorscene = Possibleknife.Possibleknife(contours[i])
        if colorscene.intBoundingRectArea == img_area:
            continue
        if checkPossibleknife(listcolordetect, colorscene):  # 检查去除一些不可能存在目标检测物的轮廓和重复包含轮廓
            listcolordetect.append(colorscene)

    imgContour, contours, npaHierarchy = cv2.findContours(out_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):  # for each contour
        colorscene = Possibleknife.Possibleknife(contours[i])
        if colorscene.intBoundingRectArea == img_area:
            continue
        if checkPossibleknife(listcolordetect, colorscene):  # 检查去除一些不可能存在目标检测物的轮廓和重复包含轮廓
            listcolordetect.append(colorscene)

    return listcolordetect

#通过轮廓检测和颜色定位双重定位确定目标检测物可能出现的地方
def checkscene(list1,list2):
    list=[]
    for boundingrect1 in list1:
        for boundingrect2 in list2:
            if (boundingrect1.intBoundingRectX>boundingrect2.intBoundingRectX and
                boundingrect1.intBoundingRectY >boundingrect2.intBoundingRectY and
                boundingrect1.intBoundingRectX <(boundingrect2.intBoundingRectX + boundingrect2.intBoundingRectWidth) and
                boundingrect1.intBoundingRectY <(boundingrect2.intBoundingRectY + boundingrect2.intBoundingRectHeight)):
                list.append(boundingrect2)
                break
            if ((boundingrect1.intBoundingRectX+boundingrect1.intBoundingRectWidth)  > boundingrect2.intBoundingRectX and
                (boundingrect1.intBoundingRectY+boundingrect1.intBoundingRectHeight) > boundingrect2.intBoundingRectY and
                (boundingrect1.intBoundingRectX+boundingrect1.intBoundingRectWidth)  < (boundingrect2.intBoundingRectX + boundingrect2.intBoundingRectWidth) and
                (boundingrect1.intBoundingRectY+boundingrect1.intBoundingRectHeight) < (boundingrect2.intBoundingRectY + boundingrect2.intBoundingRectHeight)):
                list.append(boundingrect2)
                break
    if len(list)==0:
        return list2
    else:
        return list

#根据最小外接矩阵截取目标检测物并将新图片保存到指定位置
def save_new_image(img,list):
    for size in list:
        newimage = img[size.intBoundingRectY:size.intBoundingRectY + size.intBoundingRectHeight, size.intBoundingRectX:size.intBoundingRectX + size.intBoundingRectWidth]  # 先用y确定高，再用x确定宽
        savename = os.path.join(pre_savename, str(main.count)+'.png')
        main.count=main.count+1
        print(savename)
        cv2.imwrite(savename, newimage)