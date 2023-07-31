"""
저장된 좌표 자료들(좌표, OR)과 비교값 자료들(소요시간, IoU)의 활용 방안
    
    1. 좌표 파일: Coordinate Data
        저장된 좌표 갯수가 탐지 횟수임, 각 이미지에 대한 MOR(Mean Overlap Rate) 계산

    2. 비교값 파일: Comparative Data
        
"""

import os
import argparse

from glob import glob 
from tqdm import tqdm
import csv
import pandas as pd

import cv2
import numpy as np
import matplotlib.pyplot as plt

from py_script.utils.utils import *


parser = argparse.ArgumentParser()

parser.add_argument("labeler_name", help="Typing your name", type=str)
parser.add_argument("coordinateData_path", help="Folder path to Coordinate for Coordinate Data", type=str)
parser.add_argument("comparativeData_path", help="File path to comparativeData for Comparative Data", type=str)

# parser.add_argument("--target_class_num", default=1, type=int)

args = parser.parse_args()


def main():
    
    labeler_name = args.labeler_name
    coordinateData_path = args.coordinateData_path
    comparativeData_path = args.comparativeData_path

    
    coordinateData_list = glob(os.path.join(coordinateData_path, '*.csv'))
    comparativeData_list = glob(os.path.join(comparativeData_path, '*.xlsx'))

    
    MOR_list = []
    for File in coordinateData_list:
        file_name = os.path.basename(File)
        with open(File, newline="") as csvfile:
            coordinatereader = csv.reader(csvfile, delimiter=",", quotechar='|')
            OR_list = []
            for row in coordinatereader:
                
                if 'overlap rate' in row[-1].split(":"):
                    OR_list.append(row[-1].split(":")[-1])

                else :
                    OR_list.append(row[-1])
                
                
            float_OR_list = list(map(float, OR_list))
            
            MOR = sum(float_OR_list)/len(float_OR_list)
            MOR_list.append([file_name, len(float_OR_list), MOR])
    
    
    # comparative data analysis
    comparativeData_file = comparativeData_list[-1]
    comp_xl_file = pd.read_excel(
                                 comparativeData_file, 
                                 sheet_name=None, 
                                 engine="openpyxl", 
                                 header=None,
                                 names=["img", "auto", "manual", "manual_2차", "IoU", "IoU_2차", "IoU_BGR"], 
                                 index_col=None, 
                                 skiprows=2,
                                 na_values=None,
                                 usecols= "B, C, D, E, F, G, H",
                                 dtype={"img": str, "auto": float, "manual": float, "manual_2차": float, "IoU": float, "IoU_2차": float, "IoU_BGR": float}
                                 )
    xl_sheet_list=["crack", "efflorescence", "rebar-exposure", "spalling"]
    idx = 0
    
    MOR_crack_idx=[]
    MOR_efflorescence_idx=[]
    MOR_rebarexposure_idx=[]
    MOR_spalling_idx=[]
    for i in list(range(len(MOR_list))):
        img_name = MOR_list[i][0]
        img_name_split = MOR_list[i][0].split("_")
        img_name_damage = img_name_split[0]
        
        if img_name_damage == "crack":
            
            MOR_crack_idx.append(MOR_list.index(MOR_list[i]))
        
        elif img_name_damage == "efflorescence":
            
            MOR_efflorescence_idx.append(MOR_list.index(MOR_list[i]))
        
        elif img_name_damage == "rebarExposure":
            
            MOR_rebarexposure_idx.append(MOR_list.index(MOR_list[i]))
        
        elif img_name_damage == "spalling":
            
            MOR_spalling_idx.append(MOR_list.index(MOR_list[i]))

    


    for sheet_name in xl_sheet_list:
        
        comp_xl_file_to_list = comp_xl_file[sheet_name].values.tolist()

        if sheet_name == "crack":

            for idx_MOR, idx in zip(MOR_crack_idx, list(range(len(MOR_crack_idx)))):
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                manual_time_2 = comp_xl_file_to_list[idx][3]
                iou=comp_xl_file_to_list[idx][4]
                iou_2=comp_xl_file_to_list[idx][5]
                iou_bgr = comp_xl_file_to_list[idx][6]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                improvement_rate_2=((manual_time + manual_time_2)-auto_time)/(manual_time + manual_time_2)

                MOR_list[idx_MOR].extend([auto_time, manual_time, manual_time_2, improvement_rate, improvement_rate_2, iou, iou_2, iou_bgr])
        
        elif sheet_name == "efflorescence":
            
            for idx_MOR, idx in zip(MOR_efflorescence_idx, list(range(len(MOR_crack_idx)))):
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                manual_time_2 = comp_xl_file_to_list[idx][3]
                iou=comp_xl_file_to_list[idx][4]
                iou_2=comp_xl_file_to_list[idx][5]
                iou_bgr = comp_xl_file_to_list[idx][6]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                improvement_rate_2=((manual_time + manual_time_2)-auto_time)/(manual_time + manual_time_2)

                MOR_list[idx_MOR].extend([auto_time, manual_time, manual_time_2, improvement_rate, improvement_rate_2, iou, iou_2, iou_bgr])
        
        elif sheet_name == "rebar-exposure":
            
            for idx_MOR, idx in zip(MOR_rebarexposure_idx, list(range(len(MOR_crack_idx)))):
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                manual_time_2 = comp_xl_file_to_list[idx][3]
                iou=comp_xl_file_to_list[idx][4]
                iou_2=comp_xl_file_to_list[idx][5]
                iou_bgr = comp_xl_file_to_list[idx][6]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                improvement_rate_2=((manual_time + manual_time_2)-auto_time)/(manual_time + manual_time_2)

                MOR_list[idx_MOR].extend([auto_time, manual_time, manual_time_2, improvement_rate, improvement_rate_2, iou, iou_2, iou_bgr])
        
        elif sheet_name == "spalling":
            
            for idx_MOR, idx in zip(MOR_spalling_idx, list(range(len(MOR_crack_idx)))):
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                manual_time_2 = comp_xl_file_to_list[idx][3]
                iou=comp_xl_file_to_list[idx][4]
                iou_2=comp_xl_file_to_list[idx][5]
                iou_bgr = comp_xl_file_to_list[idx][6]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                improvement_rate_2=((manual_time + manual_time_2)-auto_time)/(manual_time + manual_time_2)

                MOR_list[idx_MOR].extend([auto_time, manual_time, manual_time_2, improvement_rate, improvement_rate_2, iou, iou_2, iou_bgr])
        
            
                            
            
    MOR_damage_list = [MOR_crack_idx, MOR_efflorescence_idx, MOR_rebarexposure_idx, MOR_spalling_idx]
    

    for i in MOR_damage_list:
        iou_list = []    
        iou_list_2 = []
        iou_list_bgr = []
        mir_list = []
        for i_Mdl in i:
            iou_list.append(MOR_list[i_Mdl][-3])
            iou_list_2.append(MOR_list[i_Mdl][-2])
            iou_list_bgr.append(MOR_list[i_Mdl][-1])
            if MOR_list[i_Mdl][-4] > 0:
                mir_list.append(MOR_list[i_Mdl][-4])
            else :
                mir_list.append(MOR_list[i_Mdl][-5])
        mIoU=sum(iou_list)/len(iou_list)
        mIoU_2=sum(iou_list_2)/len(iou_list_2)
        mIoU_BGR=sum(iou_list_bgr)/len(iou_list_bgr)
        mIR = sum(mir_list)/len(mir_list)
        iou_idx = i[-1]
        MOR_list[iou_idx].extend([mIoU, mIoU_2, mIoU_BGR, mIR])
        


        
    fields = ["File", "Num of detect", "MOR", "auto time (sec)", "manual time (sec)", "manual time 2차 (sec)", "IR", "IR 2차", "IoU", "IoU 2차", "IoU BGR", "mIoU", "mIoU 2차", "mIoU BGR", "mIR"]
    
    analysisData_filename = f"{labeler_name}_analysisData.csv"
    
    analysisData = os.path.join(comparativeData_path, analysisData_filename)
    

    
    with open(analysisData, "w", encoding="cp949", newline="") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(MOR_list)
        
        
            
if __name__ == '__main__' :

    main()