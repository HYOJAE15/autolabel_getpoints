"""
SAM을 사용한 자동 이미지 라벨링의 성능 검증을 위한 비교실험 결과 분석
    
    1. 수동 라벨링: 수동 라벨링 실험 내용을 담은 엑셀 파일

    2. 자동 라벨링: SAM을 사용한 자동 라벨링 결과를 담은 엑셀 파일
        
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
parser.add_argument("manualData_path", help="File path to manual labeling Data", type=str)
parser.add_argument("automaticData_path", help="File path to automatic labeling Data", type=str)

# parser.add_argument("--target_class_num", default=1, type=int)

args = parser.parse_args()


def main():
    
    labeler_name = args.labeler_name
    manualData_path = args.manualData_path
    automaticData_path = args.automaticData_path
    
    comp_manual_xl_file = pd.read_excel(
                                 manualData_path, 
                                 sheet_name=None, 
                                 engine="openpyxl", 
                                 header=None,
                                 names=["img", "auto", "manual", "manual_2차", "IoU", "IoU_2차"], 
                                 index_col=None, 
                                 skiprows=2,
                                 na_values=None,
                                 usecols= "B, C, D, E, F, G",
                                 dtype={"img": str, "auto": float, "manual": float, "manual_2차": float, "IoU": float, "IoU_2차": float}
                                 )
    
    comp_automatic_xl_file = pd.read_excel(
                                 automaticData_path, 
                                 sheet_name=None, 
                                 engine="openpyxl", 
                                 header=None,
                                 names=["img", "auto", "IoU"], 
                                 index_col=None, 
                                 skiprows=1,
                                 na_values=None,
                                 usecols= "A, B, C",
                                 dtype={"img": int, "auto": float, "IoU": float}
                                 )
    
    xl_sheet_list = ["efflorescence", "rebar-exposure", "spalling"]
    
    efflorescence_list = []
    rebarExposure_list = []
    spalling_list = []

    for sheet_name in tqdm(xl_sheet_list) :
        
        comp_xl_file_to_list = comp_manual_xl_file[sheet_name].values.tolist()
        comp_automatic_xl_file_to_list = comp_automatic_xl_file[sheet_name].values.tolist()
        
        if sheet_name == "efflorescence":

            for i in list(range(len(comp_automatic_xl_file_to_list))):
                img = comp_xl_file_to_list[i][0]
                manual_time_1 = comp_xl_file_to_list[i][2]
                manual_time_2 = comp_xl_file_to_list[i][3]
                
                auto_time = comp_automatic_xl_file_to_list[i][1]

                iou = comp_automatic_xl_file_to_list[i][2]

                if manual_time_2 > 0:
                    manual_time = manual_time_1 + manual_time_2
                else :
                    manual_time = manual_time_1
                
                if auto_time > 0 :
                    improvement_rate = (manual_time-auto_time)/(manual_time)
                else :
                    improvement_rate = "nan"

                efflorescence_list.append([img, auto_time, manual_time, improvement_rate, iou])
    
        elif sheet_name == "rebar-exposure":

            for i in list(range(len(comp_automatic_xl_file_to_list))):
                img = comp_xl_file_to_list[i][0]
                manual_time_1 = comp_xl_file_to_list[i][2]
                manual_time_2 = comp_xl_file_to_list[i][3]
                
                auto_time = comp_automatic_xl_file_to_list[i][1]

                iou = comp_automatic_xl_file_to_list[i][2]

                if manual_time_2 > 0:
                    manual_time = manual_time_1 + manual_time_2
                else :
                    manual_time = manual_time_1
                
                if auto_time > 0 :
                    improvement_rate = (manual_time-auto_time)/(manual_time)
                else :
                    improvement_rate = "nan"

                rebarExposure_list.append([img, auto_time, manual_time, improvement_rate, iou])
    
        elif sheet_name == "spalling":

            for i in list(range(len(comp_automatic_xl_file_to_list))):
                img = comp_xl_file_to_list[i][0]
                manual_time_1 = comp_xl_file_to_list[i][2]
                manual_time_2 = comp_xl_file_to_list[i][3]
                
                auto_time = comp_automatic_xl_file_to_list[i][1]

                iou = comp_automatic_xl_file_to_list[i][2]

                if manual_time_2 > 0:
                    manual_time = manual_time_1 + manual_time_2
                else :
                    manual_time = manual_time_1
                
                if auto_time > 0 :
                    improvement_rate = (manual_time-auto_time)/(manual_time)
                else :
                    improvement_rate = "nan"

                spalling_list.append([img, auto_time, manual_time, improvement_rate, iou])


    
    total_list = efflorescence_list + rebarExposure_list + spalling_list
    


        
    fields = ["File", "auto time (sec)", "manual time (sec)", "IR", "IoU"]
    
    analysisData_filename = f"{labeler_name}_SAM_nonGetPrompt_analysisData.csv"
    analysisData = os.path.join(os.path.dirname(manualData_path), analysisData_filename)
    

    
    with open(analysisData, "w", encoding="cp949", newline="") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(total_list)
        
        
            
if __name__ == '__main__' :

    main()