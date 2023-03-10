"""

저장된 좌표 자료들(좌표, OR)과 비교값 자료들(소요시간, IoU)의 활용 방안
    
    1. 좌표 파일: Coordinate Data

    2. 비교값 파일: Comparative Data

한개의 파일로 만들어서 결과들을 한눈에 보고싶다 그지??

"""

import os
import argparse

from glob import glob 
from tqdm import tqdm
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

from py_script.utils.utils import *


parser = argparse.ArgumentParser()

parser.add_argument("coordinateData_path", help="folder path to Coordinate for Coordinate Data", type=str)
parser.add_argument("comparativeData_path", help="file path to comparativeData for Comparative Data", type=str)

# parser.add_argument("--target_class_num", default=1, type=int)

args = parser.parse_args()


def main():
    print("dataProcessing")
    
    coordinateData_path = args.coordinateData_path
    comparativeData_path = args.comparativeData_path

    print(f"coordinate: {coordinateData_path}")
    print(f"comparative: {comparativeData_path}")

    coordinateData_list = glob(os.path.join(coordinateData_path, '*.csv'))
    comparativeData_file = glob(os.path.join(comparativeData_path, '*.xlsx'))

    print(f"coordinate_list: {coordinateData_list}")
    print(f"comparative_file: {comparativeData_file}")

    MOR_list = []
    for File in coordinateData_list:
        file_name = os.path.basename(File)
        with open(File, newline="") as csvfile:
            coordinatereader = csv.reader(csvfile, delimiter=",", quotechar='|')
            OR_list = []
            for row in coordinatereader:
                OR_list.append(row[-1])
                # print(f"coordinate_list: {coordinate_list}")
                # print(", ".join(row))
                # print(f"row: {row}")
                print(f"damageClassIdx: {row[-3]}, OR: {row[-1]}")
            print(f"OR_list: {OR_list}")
            float_OR_list = list(map(float, OR_list))
            print(f"float_OR_list: {float_OR_list}")
            MOR = sum(float_OR_list)/len(float_OR_list)
            print(f"MOR: {MOR}")
            MOR_list.append([file_name, MOR])
    print(f"MOR: {MOR_list}, len: {len(MOR_list)}")

    
    # comparativeAnalysis = open(os.path.join(comparativeData_path, "analysisData.csv"), "w", encoding="cp949", newline="")
    fields = ["File", "MOR"]
    with open(os.path.join(comparativeData_path, "analysisData.csv"), "w", encoding="cp949", newline="") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(MOR_list)
        
        
            
if __name__ == '__main__' :

    main()