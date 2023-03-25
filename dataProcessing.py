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
    print("dataProcessing")
    
    labeler_name = args.labeler_name
    coordinateData_path = args.coordinateData_path
    comparativeData_path = args.comparativeData_path

    print(f"coordinate: {coordinateData_path}")
    print(f"comparative: {comparativeData_path}")

    coordinateData_list = glob(os.path.join(coordinateData_path, '*.csv'))
    comparativeData_list = glob(os.path.join(comparativeData_path, '*.xlsx'))

    # print(f"coordinate_list: {coordinateData_list}")
    print(f"comparative_file: {comparativeData_list}")

    MOR_list = []
    for File in coordinateData_list:
        file_name = os.path.basename(File)
        with open(File, newline="") as csvfile:
            coordinatereader = csv.reader(csvfile, delimiter=",", quotechar='|')
            OR_list = []
            for row in coordinatereader:
                # print(row[-1].split(":"))
                if 'overlap rate' in row[-1].split(":"):
                    OR_list.append(row[-1].split(":")[-1])

                else :
                    OR_list.append(row[-1])
                
                # print(f"coordinate_list: {coordinate_list}")
                # print(", ".join(row))
                # print(f"row: {row}")
                
                # print(f"damageClassIdx: {row[-3]}, OR: {row[-1]}")
            # print(f"OR_list: {OR_list}")
            float_OR_list = list(map(float, OR_list))
            # print(f"float_OR_list: {float_OR_list}")
            MOR = sum(float_OR_list)/len(float_OR_list)
            # print(f"MOR: {MOR}")
            MOR_list.append([file_name, len(float_OR_list), MOR])
    # print(f"MOR: {MOR_list}, len: {len(MOR_list)}")

    
    # comparative data analysis
    comparativeData_file = comparativeData_list[-1]
    comp_xl_file = pd.read_excel(
                                 comparativeData_file, 
                                 sheet_name=None, 
                                 engine="openpyxl", 
                                 header=None,
                                 names=["img", "auto", "manual", "IoU"], 
                                 index_col=None, 
                                 skiprows=2,
                                 usecols= "B, C, D, E",
                                 dtype={"img": str, "auto": float, "manual": float, "IoU": float}
                                 )
    # print(comp_xl_file)
    xl_sheet_list=["crack", "efflorescence", "rebar-exposure", "spalling"]
    idx = 0
    
    MOR_crack_idx=[]
    MOR_efflorescence_idx=[]
    MOR_rebarexposure_idx=[]
    MOR_spalling_idx=[]
    for i in list(range(len(MOR_list))):
        print(MOR_list[i])
        img_name = MOR_list[i][0]
        img_name_split = MOR_list[i][0].split("_")
        img_name_damage = img_name_split[0]
        print(img_name_damage)
        # comp_xl_file_to_list[i][0]
        if img_name_damage == "crack":
            print(MOR_list.index(MOR_list[i]))
            MOR_crack_idx.append(MOR_list.index(MOR_list[i]))
        elif img_name_damage == "efflorescence":
            print(MOR_list.index(MOR_list[i]))
            MOR_efflorescence_idx.append(MOR_list.index(MOR_list[i]))
        elif img_name_damage == "rebarExposure":
            print(MOR_list.index(MOR_list[i]))
            MOR_rebarexposure_idx.append(MOR_list.index(MOR_list[i]))
        elif img_name_damage == "spalling":
            print(MOR_list.index(MOR_list[i]))
            MOR_spalling_idx.append(MOR_list.index(MOR_list[i]))

    print(MOR_crack_idx)
    print(MOR_efflorescence_idx)
    print(MOR_rebarexposure_idx)
    print(MOR_spalling_idx)



    for sheet_name in xl_sheet_list:
        print(f"{sheet_name}")
        print(f"type: {type(comp_xl_file[sheet_name])}")
        print(f"{comp_xl_file[sheet_name]}")
        comp_xl_file_to_list = comp_xl_file[sheet_name].values.tolist()

        print(f"{comp_xl_file_to_list}")
        print(f"{type(comp_xl_file_to_list)}")
        print(f"{comp_xl_file_to_list[0][3]}")
        print(f"{type(comp_xl_file_to_list[0][3])}")
        print(f"{len(comp_xl_file_to_list)}")
        print(list(range(len(comp_xl_file_to_list))))
        if sheet_name == "crack":

            for idx_MOR, idx in zip(MOR_crack_idx, list(range(len(MOR_crack_idx)))):
                print(idx_MOR)
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                iou=comp_xl_file_to_list[idx][3]
                improvement_rate=(manual_time-auto_time)/(manual_time)

                MOR_list[idx_MOR].extend([auto_time, manual_time, improvement_rate, iou])
        elif sheet_name == "efflorescence":
            for idx_MOR, idx in zip(MOR_efflorescence_idx, list(range(len(MOR_crack_idx)))):
                print(idx_MOR)
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                iou=comp_xl_file_to_list[idx][3]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                MOR_list[idx_MOR].extend([auto_time, manual_time, improvement_rate, iou])
        elif sheet_name == "rebar-exposure":
            for idx_MOR, idx in zip(MOR_rebarexposure_idx, list(range(len(MOR_crack_idx)))):
                print(idx_MOR)
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                iou=comp_xl_file_to_list[idx][3]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                MOR_list[idx_MOR].extend([auto_time, manual_time, improvement_rate, iou])
        elif sheet_name == "spalling":
            for idx_MOR, idx in zip(MOR_spalling_idx, list(range(len(MOR_crack_idx)))):
                print(idx_MOR)
                auto_time=comp_xl_file_to_list[idx][1]
                manual_time=comp_xl_file_to_list[idx][2]
                iou=comp_xl_file_to_list[idx][3]
                improvement_rate=(manual_time-auto_time)/(manual_time)
                MOR_list[idx_MOR].extend([auto_time, manual_time, improvement_rate, iou])
        
            
                            
            
            


        
    # comparativeAnalysis = open(os.path.join(comparativeData_path, "analysisData.csv"), "w", encoding="cp949", newline="")
    fields = ["File", "Num of detect", "MOR", "auto time (sec)", "manual time (sec)", "IR", "IoU"]
    
    analysisData_filename = f"{labeler_name}_analysisData.csv"
    # analysisData_filename.format(name=labeler_name)
    analysisData = os.path.join(comparativeData_path, analysisData_filename)
    

    
    with open(analysisData, "w", encoding="cp949", newline="") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(MOR_list)
        
        
            
if __name__ == '__main__' :

    main()