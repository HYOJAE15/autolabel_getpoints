import sys
import cv2

import json
import os
import csv


import numpy as np 

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


from scipy import ndimage

from utils.utils import *


from components.actions.actionFile import ActionFile

from components.buttons.autoLabelButton import AutoLabelButton
from components.buttons.brushButton import BrushButton
from components.buttons.eraseButton import EraseButton
from components.buttons.zoomButton import ZoomButton

from components.dialogs.brushMenuDialog import BrushMenu
from components.dialogs.eraseMenuDialog import EraseMenu
from components.dialogs.newProjectDialog import newProjectDialog
from components.dialogs.setCategoryDialog import setCategoryDialog

from components.opener.dialogOpener import dialogOpener

from components.widgets.treeView import TreeView

from components.dnnModel.damage import DL_Model


sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import init_segmentor, inference_segmentor

import time

# Select folder "autolabel"
# MainWindow UI
project_ui = '../../ui_design/mainWindow.ui'

form = resource_path(project_ui)
form_class_main = uic.loadUiType(form)[0]

# Mainwindow class

class MainWindow(QMainWindow, form_class_main,
                 AutoLabelButton, BrushButton, EraseButton,
                 dialogOpener, 
                 ActionFile, TreeView) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        #### Attributes #### 
        
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        self.brushSize = 2
        self.eraseSize = 2
        self.ver_scale = 1
        self.hzn_scale = 1
        self.x = 0 
        self.y = 0 
        self.label_class = 0
        self.DL_class = 0
        self.label_segmentation = 1
        self.alpha = 0.5
        self.use_brush = False
        self.use_erase = False
        self.set_roi = False
        self.set_roi_256 = False
        # get_points_roi Mode set_roi_256
        self.get_points_roi = False
        # get_points_roi Mode set_roi
        self.get_points_roi_setRec = False
        self.stopwatch = False
        self.circle = True
        
        
        # ??????????????? ?????? ????????? ???????????? checkpoint ?????? ???????????? 

        config_file = './dnn/checkpoints/2022.01.06 cgnet general crack 2048/cgnet_2048x2048_60k_CrackAsCityscapes.py'
        checkpoint_file = './dnn/checkpoints/2022.01.06 cgnet general crack 2048/iter_60000.pth'
        self.model = init_segmentor(config_file, 
                                    checkpoint_file, 
                                    device='cuda:0')


        # config_file_efflorescence = './dnn/checkpoints/2022.07.28_cgnet_1024x1024_concrete_efflorescence/cgnet_1024x1024_60k_cityscapes.py'
        # checkpoint_file_efflorescence = './dnn/checkpoints/2022.07.28_cgnet_1024x1024_concrete_efflorescence/cgnet_1024x1024_iter_60000.pth'
        # self.model_efflorescence = init_segmentor(config_file_efflorescence, 
        #                                           checkpoint_file_efflorescence,
        #                                           device='cuda:0')
        
        # config_file_rebarexposure = './dnn/checkpoints/cgnet_rebarexposure/cgnet_1024x1024_concrete_rebar_60k_cityscapes.py'
        # checkpoint_file_rebarexposure = './dnn/checkpoints/cgnet_rebarexposure/iter_60000.pth'
        # self.model_rebarexposure = init_segmentor(config_file_rebarexposure, 
        #                                           checkpoint_file_rebarexposure,
        #                                           device='cuda:0')

        # config_file_spalling = './dnn/checkpoints/cgnet_spalling/cgnet_1024x1024_concrete_spalling_60k_cityscapes.py'
        # checkpoint_file_spalling = './dnn/checkpoints/cgnet_spalling/iter_60000.pth'
        # self.model_spalling = init_segmentor(config_file_spalling, 
        #                                           checkpoint_file_spalling,
        #                                           device='cuda:0')
        

        # self.model_list = [
        #                     self.model, 
        #                     self.model_efflorescence, 
        #                     self.model_rebarexposure,
        #                     self.model_spalling
        #                    ]

        
        # treeview setting 
        self.openFolderPath = None
        self.imgPath = None
        self.folderPath = None
        self.pathRoot = QtCore.QDir.rootPath()
        self.treeModel = QFileSystemModel(self)
        self.dialog = QFileDialog()   # Find the Folder or File Dialog
        self.treeView.clicked.connect(self.treeViewImage)
        self.treeView.clicked.connect(self.askSave)
        # self.treeView.keyPressEvent.connect(self.pressKey)
        
        # 1. Menu
        self.actionOpenFolder.triggered.connect(self.actionOpenFolderFunction)
        self.actionAddNewImages.triggered.connect(self.addNewImages)
        self.actionNewProject.triggered.connect(self.createNewProjectDialog)
        self.actionOpenProject.triggered.connect(self.openExistingProject)
        # self.actionCreate_a_Project.triggered.connect(self.openCreateProjectDialog)

        # 2. Zoom in and out
        self.ControlKey = False
        self.scale = 1
        
        # 3. brush & erase tools
        self.brushButton.clicked.connect(self.openBrushDialog)
        self.eraseButton.clicked.connect(self.openEraseDialog)

        # 4. main Image Viewer
        self.mainImageViewer.mousePressEvent = self.mousePressEvent
        self.mainImageViewer.mouseMoveEvent = self.mouseMoveEvent
        self.mainImageViewer.mouseReleaseEvent = self.mouseReleaseEvent
        self.mainImageViewer.wheelEvent = self.storeXY

        self.scrollArea.wheelEvent = self.wheelEventScroll


        # 5. listWidget
        self.listWidget.itemClicked.connect(self.getListWidgetIndex)
        self.DL_listWidget.itemClicked.connect(self.getDlListWidgetIndex)

        # 6. label opacity
        self.lableOpacitySlider.valueChanged.connect(self.showHorizontalSliderValue)
        self.labelOpacityCheckBox.stateChanged.connect(self.labelOpacityOnOff)

        # 7. auto label tools 
        self.roiMenu = QMenu()
        self.roiMenu.addAction("256*256", self.roi256)
        self.roiMenu.addAction("Set Rectangle", self.roiRec)
        self.roiAutoLabelButton.setMenu(self.roiMenu)
        self.roiAutoLabelButton.clicked.connect(self.showRoiMenu)

        self.getPointsButton.clicked.connect(self.getPoints)
        #self.roiAutoLabelButton.clicked.connect(self.runRoiAutoLabel)
    
        # 8. handMoveTool
        self.hKey = False
        self.icon = QPixmap("./Icon/square.png")
        self.scaled_icon = self.icon.scaled(QSize(5, 5), Qt.KeepAspectRatio)
        self.custom_cursor = QCursor(self.scaled_icon)

        

    def storeXY(self, event):
        if self.ControlKey:
            self.img_v_x = event.pos().x()
            self.img_v_y = event.pos().y()
            

    #### Methods ##### 

    ######################## 
    ### Image Processing ###
    ########################


    def addNewImages(self):
        
        try :

            if self.openFolderPath :
                self.imgPath = self.openFolderPath
                print(self.openFolderPath)
                print(self.imgPath)

            else :
                print(f'dang {self.imgPath}')
                # self.imgPath = self.openFolderPath
                # print(f"cityscapedataset ????????? {self.openFolderPath}")

            readFilePath = self.dialog.getOpenFileNames(
                caption="Add images to current working directory", filter="Images (*.png *.jpg *.tiff)"
                )
            images = readFilePath[0]


                # check if images are from same folder
            if self.treeModel.rootPath() in os.path.dirname(images[0]):
                print("same foler")
                return None

            if self.imgPath :

                dotSplit_imgPath = self.imgPath.split(".")
                slashSplit_imgPath = self.imgPath.split("/")
              
                    # clicked img_file
                if 'png' in dotSplit_imgPath and 'leftImg8bit' in slashSplit_imgPath :

                    img_save_folder = os.path.dirname(self.imgPath)
                   
                    img_label_folder = os.path.dirname(self.labelPath)

                    print("png, left")
                
                    # clicked img_folder
                elif 'png' not in dotSplit_imgPath and 'leftImg8bit' in slashSplit_imgPath :
    
                    img_save_folder = self.imgPath
                    img_save_folder = img_save_folder.replace( '_leftImg8bit.png', '')  
                
                    img_label_folder = img_save_folder.replace('/leftImg8bit/', '/gtFine/')
                    img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '')
                    print('left')

                else :   # ????????? ????????? ?????????????????? ???????????? ??? ?????? ?????? ?????? ?????? ?????? UnboundLocalError
                    print('not cityscapeDataset')
    
                for img in images:
                
                    temp_img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                    img_filename = os.path.basename(img) # -> basename is file name
                    img_filename = img_filename.replace(' ', '')
                    img_filename = img_filename.replace('.jpg', '.png')
                    img_filename = img_filename.replace('.JPG', '.png')
                    img_filename = img_filename.replace('.tiff', '.png')
                    img_filename = img_filename.replace('.png', '_leftImg8bit.png')

                    img_gt_filename = img_filename.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
                    gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

                    is_success, org_img = cv2.imencode(".png", temp_img)
                    org_img.tofile(os.path.join(img_save_folder, img_filename))

                    is_success, gt_img = cv2.imencode(".png", gt)
                    gt_img.tofile(os.path.join(img_label_folder, img_gt_filename))

                    # check file extension -> change extension to png 
                    # create corresponding label file 

                    print(f'7 {os.path.join(img_save_folder, img_filename)}')

            else :
                print("self.imgPath is None")
        
        except IndexError as e :
            print(e)

        except UnboundLocalError as e :
            print(e)


    def updateLayers(self, x, y):
        start_time = time.time()
        try : 
            print(f"label_class {self.label_class}")
            print(type(self.label_class))
            if self.use_brush :
                self.layers[self.label_class][y, x] = 1

            elif self.use_erase :
                self.layers[self.label_class][y, x] = 0
            
        except BaseException as e : 
            print(e)
        print("---updateLayers %s seconds ---" % (time.time() - start_time))


    def updateLabelFromLayers(self, x, y):
        start_time = time.time()
        self.label[y, x] = 0
        temp_label = self.label[y, x]
        for idx in reversed(range(1, len(self.layers))): 
            temp_label = np.where(self.layers[idx][y, x], idx, temp_label) 
        self.label[y, x] = temp_label

        print("---updateLabelFromLayers %s seconds ---" % (time.time() - start_time))

    def updateColormapFromLabel(self, x, y):
        start_time = time.time()
        try :             
            self.colormap[y, x] = self.img[y, x] * self.alpha + self.label_palette[self.label[y, x]] * (1-self.alpha)

            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        except BaseException as e : 
            print(e)
        print("---updateLabelFromLayers %s seconds ---" % (time.time() - start_time))


    def updateLabelandColormap(self, x, y):
        
        if self.use_brush :
            x, y = self.applyBrushSize(x, y)
        elif self.use_erase :
            x, y = self.applyEraseSize(x, y)


        try : 
            print(f"label_class {self.label_class}")
            print(type(self.label_class))
            if self.use_brush :
                self.label[y, x] = self.label_class
                self.colormap[y, x] = self.img[y, x] * self.alpha + self.label_palette[self.label_class] * (1-self.alpha)

            elif self.use_erase :
                self.label[y, x] = 0
                print("eraseMode")
                self.colormap[y, x] = self.img[y, x] * self.alpha + self.label_palette[0] * (1-self.alpha)

            
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        except BaseException as e : 
            print(e)

    


    def openExistingProject(self):

        try :

            readFilePath = self.dialog.getOpenFileName(
                caption="Select Project File", filter="*.hdr"
                )
            hdr_path = readFilePath[0]
            
            folderPath = os.path.dirname(hdr_path)
            print(folderPath)
            cityscapeDataset_folderPath = os.path.join(folderPath, "leftImg8bit")
                # openFolderPath ??? None ?????? ?????? treeView ?????? ????????? ?????? ?????? ?????? ????????? ?????????.
            self.openFolderPath = None
            print(os.path.join(folderPath, "leftImg8bit"))
            self.fileNameLabel.setText(cityscapeDataset_folderPath)
            self.treeModel.setRootPath(os.path.join(folderPath, 'leftImg8bit'))
            self.indexRoot = self.treeModel.index(self.treeModel.rootPath())
            self.treeView.setModel(self.treeModel)
            self.treeView.setRootIndex(self.indexRoot)
            

            with open(hdr_path) as f:
                hdr = json.load(f)

            self.listWidget.clear()

            self.label_palette = []

            for idx, cat in enumerate(hdr['categories']):
                name, color = cat[0], cat[1]
                color = json.loads(color)
                self.listWidget.addItem(name)
                iconPixmap = QPixmap(20, 20)
                iconPixmap.fill(QColor(color[0], color[1], color[2]))
                self.listWidget.item(idx).setIcon(QIcon(iconPixmap))
                self.label_palette.append(color)

            self.label_palette = np.array(self.label_palette)

        except FileNotFoundError as e:
            print(e)


    def createNewProjectDialog(self, event):
            # new_project_info ??? ???????????? ??????????????? ?????? ??????.
        self.new_project_info = {}

        self.newProjectDialog = newProjectDialog()
            # textProjectName : QTextEdit
        self.newProjectDialog.textProjectName.textChanged.connect(self.setProjectName)
        self.newProjectDialog.nextButton.clicked.connect(self.openCategoryInfoDialog)
        self.newProjectDialog.folderButton.clicked.connect(self.setFolderPath)

        self.newProjectDialog.exec()
        

    def setProjectName(self):
            # ???????????? ??????????????? ????????? ?????? ?????? key ??? ??????????????? ?????? key ??? value ?????? ?????? ??????.
            # self.new_project_info = {'project_name': self.newProjectDialog.textProjectName.toPlainText() }
        self.new_project_info['project_name'] = self.newProjectDialog.textProjectName.toPlainText()
        print(self.new_project_info['project_name'])


    def setFolderPath(self):

        readFolderPath = self.dialog.getExistingDirectory(None, "Select Folder", "./")
        print(readFolderPath)
            # folderPath : QTextEdit
        self.newProjectDialog.folderPath.setMarkdown(readFolderPath)
        self.new_project_info['folder_path'] = readFolderPath
        print(self.new_project_info)


    def createProjectHeader(self):

        createProjectFile_name = self.new_project_info['project_name'] + ".hdr"
        print(createProjectFile_name)

        path = self.new_project_info['folder_path']
        n_row = self.setCategoryDialog.tableWidget.rowCount()

        self.new_project_info['categories'] = []

        for i in range(n_row):
            self.new_project_info['categories'].append(
                [
                    self.setCategoryDialog.tableWidget.item(i, 0).text(),
                    self.setCategoryDialog.tableWidget.item(i, 2).text()
                ]
                )
            
        with open(os.path.join(path, createProjectFile_name), 'w') as fp:
            json.dump(self.new_project_info, fp)
            self.setCategoryDialog.close()


    def mousePressEvent(self, event):
        # print("mousePressEvent")

        if self.hKey : 
            self.scrollAreaMousePress(event)

        elif self.use_brush : 
            self.brushPressOrReleasePoint(event)

        elif self.use_erase :
            self.erasePressOrReleasePoint(event)

        elif self.set_roi : 
            self.roiPressPoint(event)

        elif self.set_roi_256 :
            self.roi256PressPoint(event)

        elif self.get_points_roi and self.get_points_roi_setRec == False :
            self.getPointsRoi(event)

        elif self.get_points_roi and self.get_points_roi_setRec == True :
            self.GPRpress(event)


    def mouseMoveEvent(self, event):

        if self.hKey : 
            self.scrollAreaMouseMove(event)

        elif self.use_brush : 
            self.brushMovingPoint(event)

        elif self.use_erase :
            self.eraseMovingPoint(event)

        elif self.set_roi : 
            self.roiMovingPoint(event)

        elif self.get_points_roi and self.get_points_roi_setRec == True :
            self.GPRmove(event)

    def mouseReleaseEvent(self, event): 

        if self.hKey :
            pass

        elif self.use_brush : 
            self.brushPressOrReleasePoint(event)

        elif self.use_erase :
            self.erasePressOrReleasePoint(event)

        elif self.set_roi : 
            self.roiReleasePoint(event)
        
        elif self.get_points_roi and self.get_points_roi_setRec == True :
            self.GPRrelease(event)
    
    def showRoiMenu(self):
        self.roiAutoLabelButton.showMenu()

    def setVerticalScale(self, new_scale):
        self.ver_scale = new_scale

    def setHorizontalScale(self, new_scale):
        self.hzn_scale = new_scale

        # key press ?????? ?????? ??? ?????? ???????????? ????????? ??????
        # turn on : ????????? press 
        # turn off : ????????? press 
    def keyPressEvent(self, event):
        print(event.key())
            # zoom
        if event.key() == Qt.Key_Control:
            self.ControlKey = True
            # handMove
            # h_key ?????? press ??? ????????? ??? ?????? 
        elif event.key() == Qt.Key_H: 
            self.hKey = True
            print(QCursor().shape())
            QApplication.setOverrideCursor(Qt.OpenHandCursor)

        
        
        # get_points_roi??? 1????????? A, S ??? ?????? ?????? ?????? ????????? ?????? ??????
        elif event.key() == 65 : # A Key
            
            # get_points_roi ????????? 
            if self.get_points_roi == True :
                
                self.get_points_roi_setRec = False
                self.auto_256.setChecked(True)
                self.auto_sr.setChecked(False)
            
            
            # get_points_roi ????????????
            elif self.get_points_roi == False :

                
                self.set_roi_256 = 1-self.set_roi_256

                if self.set_roi_256 : 
                    self.roiAutoLabelButton.setChecked(True)

                else : 
                    self.roiAutoLabelButton.setChecked(False)

                if self.set_roi :
                    self.set_roi = False

                if self.get_points_roi :
                    self.get_points_roi = False
                    self.getPointsButton.setChecked(False)


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

            
        elif event.key() == 89 : # Y key 
            print(f"self.se_seconds: ")
            

        # stopwatch tools
        elif event.key() == 84 : # T Key 
            # print("stopwatch")
            # self.stopwatch = 1 - self.stopwatch
            # print(f"stopwatch: {self.stopwatch}")
            
            if self.stopwatch == True :
                self.stopwatch = False
                self.stopwatchButton.setChecked(False)
                self.openStopwatchDialog(event)
                

            elif self.stopwatch == False :
                self.stopwatch = True
                self.stopwatchButton.setChecked(True)
                self.openStopwatchDialog(event)

        
        elif event.key() == 66 : # B Key
            print("B")

            if self.use_brush == True :
                self.use_brush = False
                self.brushButton.setChecked(False)

                if hasattr(self, 'brushMenu'):
                    self.brushMenu.close()  

            else :
                self.openBrushDialog(event)
                # self.listWidget.setCurrentRow(self.label_class)
                
            if self.use_erase : 
                self.use_erase = False
                self.eraseButton.setChecked(False)
                
            if  hasattr(self, 'eraseMenu'):   
                self.eraseMenu.close()
                
            if self.set_roi_256:
                self.set_roi_256 = False
                self.roiAutoLabelButton.setChecked(False)

            if self.set_roi:
                self.set_roi = False

            if self.get_points_roi :
                self.get_points_roi = False
                self.getPointsButton.setChecked(False)

        elif event.key() == 69 : # E Key
            print("E")

            if self.use_erase == True :
                self.use_erase = False
                self.eraseButton.setChecked(False)

                if  hasattr(self, 'eraseMenu'):   
                    self.eraseMenu.close()

            else :
                self.openEraseDialog(event)

            if self.use_brush :
                self.use_brush = False
                self.brushButton.setChecked(False)

            if hasattr(self, 'brushMenu'):
                self.brushMenu.close()

            if self.set_roi_256:
                self.set_roi_256 = False
                self.roiAutoLabelButton.setChecked(False)

            if self.set_roi:
                self.set_roi = False

            if self.get_points_roi :
                self.get_points_roi = False
                self.getPointsButton.setChecked(False)
            

        elif event.key() == 70 : # f Key
            print("filling works")
            self.layers[self.label_class] = ndimage.binary_fill_holes(self.layers[self.label_class])

            for idx in reversed(range(1, len(self.layers))): 
                self.label = np.where(self.layers[idx], idx, self.label) 

            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
                
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))

            self.resize_image()
            
        # get_points_roi ?????????, ???????????? ?????? 2??? ????????? ??????
        # get_points_roi ????????? ??? ???????????? set_roi_256
        # get_points_roi ????????? ??? set_roi ?????? ??????
        elif event.key() == 71 : # G key

            print("Get points")
            self.get_points_roi = 1-self.get_points_roi

            if self.get_points_roi : 
                self.roiAutoLabelButton.setChecked(True)
                self.getPointsButton.setChecked(True)
                self.auto_256.setChecked(True)
                self.auto_sr.setChecked(False)

            else : 
                self.roiAutoLabelButton.setChecked(False)
                self.getPointsButton.setChecked(False)
                self.auto_256.setChecked(False)
                self.auto_sr.setChecked(False)
                self.get_points_roi_setRec = False

            if self.set_roi_256 :
                self.set_roi_256 = False

            if self.set_roi :
                self.set_roi = False
                

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
            
        elif event.key() == 74: # j key
            

            saveFolderName = os.path.dirname(self.imgPath)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.join(saveFolderName, "Coordinate")
            saveImgName = os.path.basename(self.imgPath)
                    
            csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
            # 23.01.16 ?????? ?????? ??? ???????????? ?????? ??? ??????
            # csvImgName = saveImgName.replace("_leftImg8bit_leftImg8bit.png", "_leftImg8bit.csv")
            
            f = open(os.path.join(saveFolderName, csvImgName), "r", encoding="cp949", newline='')
            # print(f"filepath!!: {os.path.join(saveFolderName, csvImgName)}")
            data = csv.reader(f)
            self.getPointsList = []

            for row in data:
                    
                self.getPointsList.append(row)

            print(self.getPointsList)
            print(type(self.getPointsList))
            print(len(self.getPointsList))
            print(self.getPointsList[0][0])
            print(type(self.getPointsList[0]))

            for idx in self.getPointsList:
                print(idx)
                print(f"idx[0] {idx[0]} {type(idx[0])}")
                print(f"idx[0] {int(idx[0])} {type(int(idx[0]))}")
                AutoLabelButton.pointsRoi(self, int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
                
            """
            ????????? ????????? ?????? ????????? ROI??? ?????? ?????? ????????? ??????
            """
            # ?????? ?????? ????????? ??? blendImageWithColorMap ????????? ?????? ?????????
            # cv2.rectangle ?????? ???????????? ?????????, ???????????? gt??? left ??? ????????? ????????????
            for idx in self.getPointsList:
                print("?????? ?????? ????????? ??????")
                rect_start = [int(idx[2]), int(idx[0])]
                rect_end = [int(idx[3]), int(idx[1])]

                thickness = 2    

                # ????????? ?????? ???????????? ?????? ???????????? rect??? ?????? ????????? ??????
                self.colormap = cv2.rectangle(
                    self.colormap, rect_start, rect_end, (255, 255, 255), thickness)

                # print(f"rectangle size {rect_start, rect_end}")
                self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                self.resize_image()
                
                
                # result = inference_segmentor(self.model, self.img[int(idx[0]): int(idx[1]), 
                #                                                 int(idx[2]): int(idx[3]), :])

                # idx = np.argwhere(result[0] == 1)
                # y_idx, x_idx = idx[:, 0], idx[:, 1]
                # x_idx = x_idx + int(idx[2])
                # y_idx = y_idx + int(idx[0])

                # self.label[y_idx, x_idx] = self.label_segmentation # label_palette ??? ????????? ????????? ??????
                
                # self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
                # self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                # self.resize_image()


        elif event.key() == 81: # Q key
            self.labelOpacityCheckBox.setChecked(1-self.labelOpacityCheckBox.isChecked())
            self.labelOpacityOnOff()
            # Brush
            # B_key ?????? press ??? Brush ?????? ?????? ?????? 

        # Save Image
        elif event.key() == 83 : # S key
            # Save the image
            if self.ControlKey : 
                
                imwrite(self.labelPath, self.label)

                print('Save')
                self.saveFolderName = os.path.dirname(self.imgPath)
                self.saveImgName = os.path.basename(self.imgPath)
                
                print(f"self.saveFolderName : {self.saveFolderName}")
                print(f"self.saveImgName : {self.saveImgName}")
                
                self.situationLabel.setText(self.saveImgName + "???(???) ?????????????????????.")

                ### ????????? ????????? roi???????????? csv ??? ?????? ?????? ??????.
                # open("file", "mode(default=r)", encoding="cp949", newline='')
                # mode_r = read ??????
                # mode_w = writing ??????, ?????? ????????? ?????????. ????????? ?????? ?????? ????????? ????????? ?????? ????????????.
                # mode_a(appending) = writing ??????, ?????? ????????? ?????????. ???????????? ????????? open??? ????????? ??? ?????? ??????

                # self.csvImgName = self.saveImgName.replace("_leftImg8bit.png", ".csv")

                # open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="")
                
                # getScaledPoint ????????? csv ????????? ?????? 
                # ????????? csv ???????????? ????????? ????????? autolabel?????? 
            
            # Automatic image labeling tool (set_roi)
            elif self.ControlKey == False and self.get_points_roi == False :
                self.set_roi = 1-self.set_roi

                if self.set_roi : 
                    self.roiAutoLabelButton.setChecked(True)

                else : 
                    self.roiAutoLabelButton.setChecked(False)
                
                if self.set_roi_256 :       
                    self.set_roi_256 = 1-self.set_roi_256


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
            # get_points_roi Mode Automatic image labeling tool
            elif self.ControlKey == False and self.get_points_roi == True :
                print("get_points_roi Mode")
                
                # self.get_points_roi ???????????? ????????? ?????? ????????? setRec??? 256 ????????????
                self.get_points_roi_setRec = True
                self.auto_256.setChecked(False)
                self.auto_sr.setChecked(True)
                # self.get_points_roi = False

        # Delete Image
        elif event.key() == 16777223 : # delete key
            print(event.key())
            
            
            if "png" in self.dotSplit_imgPath:
                os.remove(self.imgPath)    
                os.remove(self.labelPath)
                saveFolderName = os.path.dirname(self.imgPath)
                saveImgName = os.path.basename(self.imgPath)
                csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
                # csv ?????? ?????? ??? ?????? ??????
                # os.remove(os.path.join(saveFolderName, csvImgName))
            elif "csv" in self.dotSplit_imgPath:
                os.remove(self.imgPath)
            

                
        else :
            print(event.key())
          
    def keyReleaseEvent(self, event):

            # zoom
        if event.key() == Qt.Key_Control:
            self.ControlKey = False
            # QApplication.restoreOverrideCursor()

            # handMove
        elif event.key() == Qt.Key_H:
            self.hKey = False
            QApplication.restoreOverrideCursor()
            
        # brush ?????? ??? ?????? ???????????? ????????? ????????????
        # mousePress ??? Release def ??? ??????
    def scrollAreaMousePress(self, event):

        self.hand_last_point = QPoint(QCursor.pos().x(), QCursor.pos().y())
        print(f"scrollAreaMousePress's pos {self.hand_last_point}")
        
    def scrollAreaMouseMove(self, event):


        delta_y = self.hand_last_point.y() - QCursor.pos().y()
        delta_x = self.hand_last_point.x() -  QCursor.pos().x() 

        print(f"delta_y {delta_y}, delta_x {delta_x}")

        setvalueY = self.scrollArea.verticalScrollBar().value()
        setvalueX = self.scrollArea.horizontalScrollBar().value()
        
        self.scrollArea.verticalScrollBar().setValue(setvalueY + delta_y)
        self.scrollArea.horizontalScrollBar().setValue(setvalueX + delta_x)

        self.hand_last_point = QPoint(QCursor.pos().x(), QCursor.pos().y())

    
    def wheelEventScroll(self, event):
        """
        FIXME: zoom in & out ??? ????????? ?????? ????????? ???????????? ?????? 
        ????????? ??? ???????????? ?????? ??? ??? ????????? ????????? ???????????? ?????? ?????? ????????? ??????.
        """
        
        self.mouseWheelAngleDelta = event.angleDelta().y() # -> 1 (up), -1 (down)
        if self.ControlKey:
                
            if self.mouseWheelAngleDelta > 0: 
                self.scale *= 1.1
                width_tobe = self.mainImageViewer.geometry().width() * 1.1
                print(f"self.mainImageViewer.geometry: {self.mainImageViewer.geometry()}")
                height_tobe = self.mainImageViewer.geometry().height() * 1.1
            else : 
                self.scale /= 1.1
                width_tobe = self.mainImageViewer.geometry().width() / 1.1
                height_tobe = self.mainImageViewer.geometry().height() / 1.1

            self.resize_image()

            _width_diff = width_tobe - self.scrollArea.geometry().width()
            _height_diff = height_tobe - self.scrollArea.geometry().height() 

            x_max_img_v = self.mainImageViewer.geometry().width()
            y_max_img_v = self.mainImageViewer.geometry().height()

            set_hor_max = _width_diff + 45 if _width_diff > 0 else 0
            set_ver_max = _height_diff + 45 if _height_diff > 0 else 0

            self.scrollArea.horizontalScrollBar().setRange(0, set_hor_max) 
            self.scrollArea.verticalScrollBar().setRange(0, set_ver_max) 
            
            ver_max = self.scrollArea.verticalScrollBar().maximum()
            hor_max = self.scrollArea.horizontalScrollBar().maximum()
            
            if self.scrollArea.verticalScrollBar().maximum() > 0: 
                setvalueY = self.img_v_y/y_max_img_v*ver_max                
                self.scrollArea.verticalScrollBar().setValue(setvalueY)

            if self.scrollArea.horizontalScrollBar().maximum() > 0: 
                setvalueX = self.img_v_x/x_max_img_v*hor_max
                
                self.scrollArea.horizontalScrollBar().setValue(setvalueX)

        else : 
            scroll_value = self.scrollArea.verticalScrollBar().value()
            self.scrollArea.verticalScrollBar().setValue(scroll_value - self.mouseWheelAngleDelta)

    def resize_image(self):
        size = self.pixmap.size()
        self.scaled_pixmap = self.pixmap.scaled(self.scale * size)
        self.mainImageViewer.setPixmap(self.scaled_pixmap)

    def showHorizontalSliderValue(self):

        self.labelOpacityCheckBox.setChecked(True)

        if abs(self.alpha-(self.lableOpacitySlider.value() / 100)) > 0.03 :
            self.alpha = self.lableOpacitySlider.value() / 100
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()    

    def labelOpacityOnOff(self):
        
        if self.labelOpacityCheckBox.isChecked():
            self.alpha = self.lableOpacitySlider.value() / 100
        else : 
            self.alpha = 1 
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()    

    def getListWidgetIndex (self):

        print(f"self.listWidget.currentRow(){self.listWidget.currentRow()}")
        
        self.label_class = self.listWidget.currentRow()
        self.label_segmentation = self.listWidget.currentRow()
        
        if self.use_brush :
            print("Brush")

        elif self.set_roi or self.set_roi_256 or self.get_points_roi or self.get_points_roi_setRec :
            if self.label_segmentation == 1:
                DL_Model.crackModel(self)
            elif self.label_segmentation == 2:
                DL_Model.efflorescenceModel(self)
            elif self.label_segmentation == 3:                  # train_8 : crack, rebar, spalling, effloresence
                DL_Model.rebarExposureModel(self)               # ?????? ???????????? : crack, effloresence, rebar, spalling
            elif self.label_segmentation == 4:
                DL_Model.spallingModel(self)       
                

    def getDlListWidgetIndex(self):
        print(f"dlListidx {self.DL_listWidget.currentRow()}")
        self.DL_class = self.DL_listWidget.currentRow()
        self.model = self.model_list[self.DL_class]
        

if __name__ == "__main__" :

    # Open Chalk window
    app = QApplication(sys.argv)
    myWindow = MainWindow() 
    myWindow.show()

    # Open Stopwatch Window 
    # ClockApp().run()
    # LabelBase.register(name='Roboto',
    #                 fn_regular='./font/Roboto-Thin.ttf',
    #                 fn_bold='./font/Roboto-Medium.ttf')

    sys.exit(app.exec_())
