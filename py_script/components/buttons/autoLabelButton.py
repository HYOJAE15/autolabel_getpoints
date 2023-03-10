
from cmath import e
import cv2
import sys

import numpy as np 
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import csv

from utils.utils import *

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import inference_segmentor



class AutoLabelButton :
    def __init__(self) :
        super().__init__()
                

    def roi256(self):
        print("roi256")
        self.brushButton.setChecked(False)
        self.roiAutoLabelButton.setChecked(True)
        self.getPointsButton.setChecked(False)

        self.use_brush = False

        self.use_erase = False

        self.set_roi = False

        self.get_points_roi = False

        self.set_roi_256 = True


    def roiRec(self):
        self.brushButton.setChecked(False)
        self.roiAutoLabelButton.setChecked(True)
        self.getPointsButton.setChecked(False)
        print(f"self.use_brush {self.use_brush}")

        self.use_brush = False

        self.use_erase = False

        self.set_roi = True
        
        self.get_points_roi = False

        self.set_roi_256 = False

        print(f"self.set_roi {self.set_roi}")
        print(f"self.use_brush {self.use_brush}")

    def getPoints(self):
        print("getPoints")
        self.get_points_roi = 1 - self.get_points_roi
        
        if self.get_points_roi:

            self.roiAutoLabelButton.setChecked(True)
            self.getPointsButton.setChecked(True)
        else :
            self.roiAutoLabelButton.setChecked(False)
            self.getPointsButton.setChecked(False)

        if self.set_roi :
            self.set_roi = False
        
        if self.set_roi_256 :
            self.set_roi_256 = False

        if self.use_erase : 
                self.use_erase = False
                self.eraseButton.setChecked(False)

        if  hasattr(self, 'eraseMenu'):   
            self.eraseMenu.close()

        if self.use_brush :
            self.use_brush = False
            self.brushButton.setChecked(False)
            
        if hasattr(self, 'brushMenu'):
            self.brushMenu.close()

        

    def getPointsRoi (self, event):
        print("getPointsRoi")
        try : 

            self.x_r256, self.y_r256 = getScaledPoint(event, self.scale)
            
            """
            NOTE: 256*256 Automatic Labeling on Getpoints Mode Calculation Coordinate

            FIXME: ????????? ?????????????????? Auto Labeling ??? ????????? ????????? shape ????????? ?????????.
            TODO: 
            1. ROI??? ????????? ???????????? ???????????? ?????? 8????????? ????????? ????????? ???????????? ???????????? ?????? 
                ??? 9?????? ????????? ?????? ????????? ???????????? (x, y) cv2??? ????????? (y, x)
            2. np.clip?????? image shape????????? ????????? ????????? ??????
            """
            # ????????? ?????? ??????
            if self.x_r256 <= 128 and self.y_r256 <= 128 :
                print(f"????????? ?????? ??????")
                self.rect_start = [0, 0]
                self.rect_end = [self.x_r256+128, self.y_r256+128]
            
            # ????????? ?????? (??????, ??????)
            elif self.x_r256 < 128 and self.y_r256 > 128 :
                print(f"????????? ?????? (??????, ??????)")
                self.rect_start = [0, self.y_r256-128]
                # self.rect_end = self.x_r256+128, self.y_r256+128
                self.rect_end = [self.x_r256+128, self.y_r256+128] if self.img.shape[0]>=(self.y_r256+128) else [self.x_r256+128, self.img.shape[0]]
                print(f"????????? ??????: {self.x_r256, self.y_r256}")
                print(f"????????? ?????? ??????: {self.rect_end}")

            # ????????? ?????? (??????, ??????)
            elif self.x_r256+128 > self.img.shape[1] and self.y_r256-128 >= 0 :
                print(f"????????? ?????? (??????, ??????)") 
                self.rect_start = [self.x_r256-128, self.y_r256-128]
                self.rect_end = [self.img.shape[1], self.y_r256+128] if (self.y_r256+128)<self.img.shape[0] else [self.img.shape[1], self.img.shape[0]] 
                print(f"self.rect_end: {self.rect_end}")

            # ????????? ?????? (??????, ??????)
            elif self.y_r256 < 128 and self.x_r256 > 128 :
                print(f"????????? ?????? (??????, ??????)")
                self.rect_start = [self.x_r256-128, 0]
                # self.rect_end = self.x_r256+128, self.y_r256+128
                self.rect_end = [self.x_r256+128, self.y_r256+128] if self.img.shape[1]>=(self.x_r256+128) else [self.img.shape[1], self.y_r256+128] 
            
            # ????????? ??????, ????????? ?????? ??????
            else :
                print(f"????????? ??????, ????????? ?????? ??????")
                self.rect_start = [self.x_r256-128, self.y_r256-128]
                # self.rect_end = [self.x_r256+128, self.y_r256+128]
                self.rect_end = [self.x_r256+128, self.img.shape[0]] if (self.y_r256+128)>self.img.shape[0] else [self.x_r256+128, self.y_r256+128]
            

            """
            NOTE: Automatic Labeling by Deep Learning
            """
            result = inference_segmentor(self.model, self.img[self.rect_start[1]: self.rect_end[1],
                                            self.rect_start[0]: self.rect_end[0], :])

            print(f'modelListindex {self.label_segmentation-1}')

            cv2.imshow("cropImage", self.img[self.rect_start[1]: self.rect_end[1],
                                            self.rect_start[0]: self.rect_end[0], :])
            

            print(f'cropImage.shape {self.img[self.rect_start[1]: self.rect_end[1], self.rect_start[0]: self.rect_end[0], :].shape}')
            

            idx = np.argwhere(result[0] == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]
            x_idx = x_idx + self.rect_start[0]
            y_idx = y_idx + self.rect_start[1]

            self.label[y_idx, x_idx] = self.label_segmentation # label_palette ??? ????????? ????????? ??????
            
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()

            


            """
            NOTE: ROI Box Computer Visualize
            TODO: ?????? ????????? mainwindow?????? ???????????? ?????? ???????????? ?????? ????????????
            """
            thickness = 2    

            rect_256 = cv2.rectangle(
                self.colormap.copy(), self.rect_start, self.rect_end, (255, 255, 255), thickness
            )
            self.pixmap = QPixmap(cvtArrayToQImage(rect_256))
            self.resize_image()


            # ????????? ????????? ?????? ????????? ?????? ??? ????????? ??????
            # uint (Unsigned int: ?????? ?????? ???????????? ??????... ???????????? ??????), uint8 (0~2^8: 0~255, ??? 256??? ??????) 
            # roi_coord_img = np.zeros(self.img.shape, np.uint8)
            
            
            self.roi_coord_img = np.zeros(self.img.shape)
            
            self.roi_union = np.zeros(self.img.shape[:2]) 
            
            self.roi_last = np.zeros(self.img.shape[:2])
            self.roi_last = cv2.rectangle(self.roi_last, self.rect_start, self.rect_end, (1, 1, 1), -1)
            
            
            """
            NOTE:?????? ?????? ?????? ??? csv ?????? ????????? Overlap Rate(OR) ??????
            
            FIXME: ????????? ?????? ?????? ?????? ?????? ?????? ????????? ????????? ?????? ?????? ????????? ...
            VSD??? ?????? ??? ????????? ?????? ?????? ??????, GUI??? ?????? ????????? ????????? ???????????? ????????? ?????? ??????  
            ?????? ?????? ?????? ??????, ????????? ?????? ?????? ?????? ?????? ?????? (?????? ??????)
            ????????? ?????? ??? ?????? ????????? csv ????????? ????????? ????????? ?????? ?????? ?????? 
            """
            
            # Create csv file and save img coordinate, overlap rate
            print(f"autolabelScripts : {self.imgPath}" )
            self.saveFolderName = os.path.dirname(self.imgPath)
            self.saveFolderName = os.path.dirname(self.saveFolderName)
            self.saveFolderName = os.path.dirname(self.saveFolderName)
            self.saveFolderName = os.path.join(self.saveFolderName, "Coordinate")
            # ????????? ??????(gtFine, leftImg8bit)
            self.saveImgName = os.path.basename(self.imgPath)
                    
            self.csvImgName = self.saveImgName.replace("_leftImg8bit.png", ".csv")
            if os.path.exists(self.saveFolderName) == False :
                os.mkdir(self.saveFolderName)

            elif os.path.exists(self.saveFolderName) == True :
                print("Folder Exists")

            self.points = open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="")

            self.situationLabel.setText(self.csvImgName + "???(???) Coordinate ????????? ?????????????????????.")
            

                          
            # overlap rate
            overlap = open(os.path.join(self.saveFolderName, self.csvImgName), "r", encoding="cp949", newline="")
            overlap_list = csv.reader(overlap)
            # print(f"overlap_list: {overlap_list}")
            # coordinate [y_start, y_end, x_start, x_end]
            self.overlap_list = [[0,0,0,0]]
            for line in overlap_list :
                self.overlap_list.append(line)
                
            for idx in self.overlap_list[1:]:   
                start=[int(idx[2]), int(idx[0])]
                end=[int(idx[3]), int(idx[1])]
                self.roi_coord_img = cv2.rectangle(self.roi_coord_img, start, end, (1, 1, 1), -1)
                self.roi_coord_img = cv2.rectangle(self.roi_coord_img, self.rect_start, self.rect_end, (0, 255, 0), -1)
                cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("ROI", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
                cv2.imshow("ROI", self.roi_coord_img)
                
                self.roi_union = cv2.rectangle(self.roi_union, start, end, (1, 1, 1), -1)


            
            
            
            """
            ?????? ?????? ROI??? ??? ?????? ROI ??????????????? ?????????
            """
            union = np.count_nonzero(self.roi_union)
            print(f"union: {union}")
            last_roi = cv2.countNonZero(self.roi_last)
            print(f"last_roi: {last_roi}")
            if union > 0 : 
                intersection = cv2.bitwise_and(self.roi_last, self.roi_union)
                # cv2.namedWindow("intersection", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("intersection", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
                # cv2.imshow("intersection", intersection)
                intersection_roi = cv2.countNonZero(intersection)
                print(f"intersection: {intersection_roi}")
                self.overlap_rate = intersection_roi/last_roi
                print(f"overlap_rate: {self.overlap_rate}")
            elif union == 0 :
                self.overlap_rate = 0
                print(f"union_0_overlap_rate: {self.overlap_rate}")

             

            # if cv2.countNonZero(self.roi_union) > 0 :
            #     overlap_rectangle = cv2.bitwise_and(self.roi_last, self.roi_union)
            #     overlap_rate = cv2.countNonZero(overlap_rectangle)
            #     print(f"overlap_rate: {overlap_rate}")
            #     plt.imshow(overlap_rectangle)
            #     plt.show()
            

            # overlap_rectangle = (abs(int(self.overlap_list[-1][3])-int(self.rect_start[0]))*abs(int(self.overlap_list[-1][1])-int(self.rect_start[1])))
            # print(f"overlap_rec: {overlap_rectangle}")
            # current_rectangle = abs(int(self.rect_end[0])-int(self.rect_start[0]))*abs(int(self.rect_end[1])-int(self.rect_start[1]))
            # print(f"current_rectangle: {current_rectangle}")

            
            # overlap_rate = overlap_rectangle/current_rectangle    
            # print(f"overlap_rate: {overlap_rate}")
            
            

            print(f"currentimgshape: {self.img.shape}")

            self.pointsList = [self.rect_start[1], self.rect_end[1],
                               self.rect_start[0], self.rect_end[0],
                               f"x: {self.x_r256}", f"y: {self.y_r256}",
                               f"classIdx:{self.label_segmentation}",
                               f"overlap rate:",
                               f"{self.overlap_rate}"]
            
            csvWriter = csv.writer(self.points)
            csvWriter.writerow(self.pointsList)

            
            """
            NOTE: Test
            """
            # plt.imshow(self.roi_coord_img)
            # plt.show()

            # cv2.imshow("rec", self.roi_coord_img)
            # # cv2.waitKey(0)
            
            
            


        except ZeroDivisionError as e :
            print(e)

    # Get Points set Rectangle roi  
    def GPRpress (self, event):
        print("GPRpress")
        x, y = getScaledPoint(event, self.scale)

        self.rect_start_GP = [x, y]
        print(f"rect_start: {self.rect_start_GP}")

    def GPRmove (self, event) :
        x, y = getScaledPoint(event, self.scale)

        self.rect_end_GP = [x, y]

        thickness = 5

        rect_hover = cv2.rectangle(
            self.colormap.copy(), self.rect_start_GP, self.rect_end_GP, (255, 255, 255), thickness) # , color, thickness, lineType,)
        self.pixmap = QPixmap(cvtArrayToQImage(rect_hover))
        self.resize_image()
    def GPRrelease (self, event) :
        print("GPRrelease")
        x, y = getScaledPoint(event, self.scale)
        print(f"x, y: {x},{y}")
    
        if x < self.rect_start_GP[0] : 
            temp = x 
            x = self.rect_start_GP[0]
            print(f"x: {x}")
            self.rect_start_GP[0] = temp
            print(f"rect_start: {self.rect_start_GP[0]}") 
            
        if y < self.rect_start_GP[1] : 
            temp = y 
            y = self.rect_start_GP[1]
            self.rect_start_GP[1] = temp 

        self.rect_end_GP = [x, y] 

        if (self.rect_end_GP[0] == self.rect_start_GP[0]) | (self.rect_end_GP[1] == self.rect_start_GP[1]):
            print(f"????????? ??????")
            self.rect_start_GP[0] -= 128
            self.rect_start_GP[1] -= 128  

            self.rect_end_GP[0] += 128 
            self.rect_end_GP[1] += 128 

        print(f"self.label.shape: {self.label.shape}")
        self.rect_start_GP[0] = np.clip(self.rect_start_GP[0], 0, self.label.shape[1])
        self.rect_start_GP[1] = np.clip(self.rect_start_GP[1], 0, self.label.shape[0])

        self.rect_end_GP[0] = np.clip(self.rect_end_GP[0], 0, self.label.shape[1])
        self.rect_end_GP[1] = np.clip(self.rect_end_GP[1], 0, self.label.shape[0])
            

        result = inference_segmentor(self.model, self.img[self.rect_start_GP[1]: self.rect_end_GP[1], self.rect_start_GP[0]: self.rect_end_GP[0], :])
        print(f"GPRrelease Coordinate: {self.rect_start_GP[1], self.rect_end_GP[1], self.rect_start_GP[0], self.rect_end_GP[0]}")
        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + self.rect_start_GP[0]
        y_idx = y_idx + self.rect_start_GP[1]

        self.label[y_idx, x_idx] = self.label_segmentation
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()

        thickness = 2    

        rect_256 = cv2.rectangle(
                self.colormap.copy(), self.rect_start_GP, self.rect_end_GP, (255, 255, 255), thickness
            )
        self.pixmap = QPixmap(cvtArrayToQImage(rect_256))
        self.resize_image()


        self.roi_coord_img = np.zeros(self.img.shape)
            
        self.roi_union = np.zeros(self.img.shape[:2]) 
        
        self.roi_last = np.zeros(self.img.shape[:2])
        self.roi_last = cv2.rectangle(self.roi_last, self.rect_start_GP, self.rect_end_GP, (1, 1, 1), -1)
        

        # Create csv File

        print(f"autolabelScripts : {self.imgPath}" )
        self.saveFolderName = os.path.dirname(self.imgPath)
        self.saveFolderName = os.path.dirname(self.saveFolderName)
        self.saveFolderName = os.path.dirname(self.saveFolderName)
        self.saveFolderName = os.path.join(self.saveFolderName, "Coordinate")
        # ????????? ??????(gtFine, leftImg8bit)
        self.saveImgName = os.path.basename(self.imgPath)
                
        self.csvImgName = self.saveImgName.replace("_leftImg8bit.png", ".csv")
        if os.path.exists(self.saveFolderName) == False :
            os.mkdir(self.saveFolderName)

        elif os.path.exists(self.saveFolderName) == True :
            print("Folder Exists")

        self.points = open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="")

        self.situationLabel.setText(self.csvImgName + "???(???) Coordinate ????????? ?????????????????????.")
        

        # overlap rate
        overlap = open(os.path.join(self.saveFolderName, self.csvImgName), "r", encoding="cp949", newline="")
        overlap_list = csv.reader(overlap)
        # print(f"overlap_list: {overlap_list}")
        # coordinate [y_start, y_end, x_start, x_end]
        self.overlap_list = [[0,0,0,0]]
        for line in overlap_list :
            self.overlap_list.append(line)
            
        for idx in self.overlap_list[1:]:   
            start=[int(idx[2]), int(idx[0])]
            end=[int(idx[3]), int(idx[1])]
            self.roi_coord_img = cv2.rectangle(self.roi_coord_img, start, end, (1, 1, 1), -1)
            self.roi_coord_img = cv2.rectangle(self.roi_coord_img, self.rect_start_GP, self.rect_end_GP, (0, 255, 0), -1)
            cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROI", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
            cv2.imshow("ROI", self.roi_coord_img)
            
            self.roi_union = cv2.rectangle(self.roi_union, start, end, (1, 1, 1), -1)


        
        
        
        """
        ?????? ?????? ROI??? ??? ?????? ROI ??????????????? ?????????
        """
        union = np.count_nonzero(self.roi_union)
        print(f"union: {union}")
        last_roi = cv2.countNonZero(self.roi_last)
        print(f"last_roi: {last_roi}")
        if union > 0 : 
            intersection = cv2.bitwise_and(self.roi_last, self.roi_union)
            # cv2.namedWindow("intersection", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("intersection", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
            # cv2.imshow("intersection", intersection)
            intersection_roi = cv2.countNonZero(intersection)
            print(f"intersection: {intersection_roi}")
            self.overlap_rate = intersection_roi/last_roi
            print(f"overlap_rate: {self.overlap_rate}")
        elif union == 0 :
            self.overlap_rate = 0
            print(f"union_0_overlap_rate: {self.overlap_rate}")

        
        
        
        
        self.pointsList = [self.rect_start_GP[1], self.rect_end_GP[1],
                           self.rect_start_GP[0], self.rect_end_GP[0],
                           f"x: ", f"y: ",
                           f"class idx:{self.label_segmentation}",
                           f"overlap rate:",
                           f"{self.overlap_rate}"]
        
        csvWriter = csv.writer(self.points)
        csvWriter.writerow(self.pointsList)




    def roi256PressPoint(self, event):

        try : 

            self.x_r256, self.y_r256 = getScaledPoint(event, self.scale)
            if self.x_r256 < 128 and self.y_r256 < 128 :
                self.rect_start = 0, 0
                self.rect_end = self.x_r256+128, self.y_r256+128
            elif self.x_r256 < 128 :
                self.rect_start = 0, self.y_r256-128
                self.rect_end = self.x_r256+128, self.y_r256+128
            elif self.y_r256 < 128 :
                self.rect_start = self.x_r256-128, 0
                self.rect_end = self.x_r256+128, self.y_r256+128 
            else :
                self.rect_start = self.x_r256-128, self.y_r256-128
                self.rect_end = self.x_r256+128, self.y_r256+128

            
            result = inference_segmentor(self.model, self.img[self.rect_start[1]: self.rect_end[1],
                                            self.rect_start[0]: self.rect_end[0], :])

            print(f'modelListindex {self.label_segmentation-1}')

            cv2.imshow("cropImage", self.img[self.rect_start[1]: self.rect_end[1],
                                            self.rect_start[0]: self.rect_end[0], :])
            

            print(f'cropImage.shape {self.img[self.rect_start[1]: self.rect_end[1], self.rect_start[0]: self.rect_end[0], :].shape}')
            

            idx = np.argwhere(result[0] == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]
            x_idx = x_idx + self.rect_start[0]
            y_idx = y_idx + self.rect_start[1]

            self.label[y_idx, x_idx] = self.label_segmentation # label_palette ??? ????????? ????????? ??????
            
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()


            thickness = 2    

            rect_256 = cv2.rectangle(
                self.colormap.copy(), self.rect_start, self.rect_end, (255, 255, 255), thickness)

            print(f"rectangle size {self.rect_start, self.rect_end}")
            self.pixmap = QPixmap(cvtArrayToQImage(rect_256))
            self.resize_image()
            
        except ZeroDivisionError as e :
            print(e)

        
    def roiPressPoint(self, event):

        x, y = getScaledPoint(event, self.scale)

        self.rect_start = [x, y]

    def roiMovingPoint(self, event):

        x, y = getScaledPoint(event, self.scale)

        self.rect_end = [x, y]

        thickness = 5

        rect_hover = cv2.rectangle(
            self.colormap.copy(), self.rect_start, self.rect_end, (255, 255, 255), thickness) # , color, thickness, lineType,)
        self.pixmap = QPixmap(cvtArrayToQImage(rect_hover))
        self.resize_image()
        

    def roiReleasePoint(self, event):

        x, y = getScaledPoint(event, self.scale)
    
        if x < self.rect_start[0] : 
            temp = x 
            x = self.rect_start[0]
            self.rect_start[0] = temp 
            
        if y < self.rect_start[1] : 
            temp = y 
            y = self.rect_start[1]
            self.rect_start[1] = temp 

        self.rect_end = [x, y] 

        if (self.rect_end[0] == self.rect_start[0]) | (self.rect_end[1] == self.rect_start[1]):
            self.rect_start[0] -= int(128/self.scale)
            self.rect_start[1] -= int(128/self.scale)  

            self.rect_end[0] += int(128/self.scale) 
            self.rect_end[1] += int(128/self.scale) 

        self.rect_start[0] = np.clip(self.rect_start[0], 0, self.label.shape[1])
        self.rect_start[1] = np.clip(self.rect_start[1], 0, self.label.shape[0])

        self.rect_end[0] = np.clip(self.rect_end[0], 0, self.label.shape[1])
        self.rect_end[1] = np.clip(self.rect_end[1], 0, self.label.shape[0])
            

        result = inference_segmentor(self.model, self.img[self.rect_start[1]: y, self.rect_start[0]: x, :])

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + self.rect_start[0]
        y_idx = y_idx + self.rect_start[1]

        self.label[y_idx, x_idx] = self.label_segmentation
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()


    def pointsRoi(self, y_start, y_end, x_start, x_end):
        # self.x_r256, self.y_r256 = getScaledPoint(event, self.scale)
        # if self.x_r256 < 128 and self.y_r256 < 128 :
        #     self.rect_start = 0, 0
        #     self.rect_end = self.x_r256+128, self.y_r256+128
        # elif self.x_r256 < 128 :
        #     self.rect_start = 0, self.y_r256-128
        #     self.rect_end = self.x_r256+128, self.y_r256+128
        # elif self.y_r256 < 128 :
        #     self.rect_start = self.x_r256-128, 0
        #     self.rect_end = self.x_r256+128, self.y_r256+128 
        # else :
        #     self.rect_start = self.x_r256-128, self.y_r256-128
        #     self.rect_end = self.x_r256+128, self.y_r256+128

        
        result = inference_segmentor(self.model, self.img[y_start: y_end,
                                        x_start: x_end, :])

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + x_start
        y_idx = y_idx + y_start

        self.label[y_idx, x_idx] = self.label_segmentation # label_palette ??? ????????? ????????? ??????
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()

        
          
        

        # self.rect_start_vi = [x_start, y_start]
        # self.rect_end_vi = [x_end, y_end]

        # thickness = 2    

        # # ????????? ?????? ???????????? ?????? ???????????? rect??? ?????? ????????? ??????
        # self.colormap = cv2.rectangle(
        #     self.colormap, self.rect_start_vi, self.rect_end_vi, (255, 255, 255), thickness)

        # # print(f"rectangle size {rect_start, rect_end}")
        # self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        # self.resize_image()
        
        


        
    
