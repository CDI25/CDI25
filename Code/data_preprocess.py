

import pandas as pd
import numpy as np
import random
from cmath import nan
from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
from statsmodels.tsa.arima.model import ARIMA
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import torch.nn as nn
import torch
from cmath import nan
import math
from statsmodels.tsa.ar_model import AutoReg


def get_org_data(args):
    if args.dataset == "energy":
        res = pd.read_csv("Data/energy.csv")[["T1","T2","T3","T4","T5","T6","T7","T8","T9"]].to_numpy()
        return res
    if args.dataset == "AIP":
        res = np.loadtxt("Data/AIP.csv", delimiter=",")
        return res
    if args.dataset == "Microsoft_Stock":
        res = pd.read_csv("Data/Microsoft_Stock.csv")[["Open","High","Low","Close","Volume"]].to_numpy()
        return res
    if args.dataset =="Treasury":
        res = pd.read_csv("Data/Treasury.csv")[["High","Low","Open","Close","Adj Close"]].to_numpy()
        return res
    if args.dataset =="AirQualityUCI":
        res = np.loadtxt("Data/AirQualityUCI.csv", delimiter=",")
        return res
    if args.dataset =="ILINet":
        res = np.loadtxt("Data/ILINet.csv", delimiter=",")
        return res
    if args.dataset == "power":
        res = np.loadtxt("Data/power.csv", delimiter=",")
        return res
    if args.dataset == "gps":
        res = np.loadtxt("Data/location.csv", delimiter=",")
        return res
    if args.dataset == "imu":
        res, __ = load_imu()
        return res
    if args.dataset == "weer":
        res, __ = load_weer()
        return res


def canmask(i, j, len, mask_flag):
    if i-1 >= 0:
        if mask_flag[i-1, j] == 1:
            return False
    if i+len < mask_flag.shape[0]:
        if mask_flag[i+len, j] == 1:
            return False
    return True

def get_location_mask():
    res = np.loadtxt("Mask/location_mask.csv", delimiter=",")
    return res

def get_mask_flag(args, org_data):
    if args.dataset == "location":
        return get_location_mask()
    if args.dataset == "imu":
        __, mask = load_imu()
        return mask
    if args.dataset == "weer":
        res, __ = load_weer()
        return res
    random.seed(args.seed + args.iter)
    np.random.seed(args.seed + args.iter)

    mask_flag = np.zeros(org_data.shape)

    time_step_num = org_data.shape[0]
    missing_sensor_num = len(args.missing_column)
    # missing_num = int(time_step_num * missing_sensor_num * args.missing_percent) 
    missing_num_for_attr = int(time_step_num * args.missing_percent) 
  

    for attr in args.missing_column:
        while np.sum(mask_flag[:, attr]) < missing_num_for_attr:
            # if np.sum(mask_flag[:, attr]) % 1000 == 0:
                # print(np.sum(mask_flag[:, attr]))
            mask_len = random.randint(1, args.max_missing_len)
            x = random.randint(0, time_step_num-1)
    
            if canmask(x, attr, mask_len, mask_flag) == False:
                continue
            mask_flag[x, attr] = 1

            if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
            for i in range(x+1, min(time_step_num, x+mask_len)):
                mask_flag[i, attr] = 1
                if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                    break    
            if np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break

    mask_flag[np.where(org_data == -200)] = 1
    return mask_flag

def EWMA_init(norm_data, mask_flag):
    numpy2ri.activate()
    norm_data[np.where(mask_flag == 1)] = nan
    r('suppressMessages(require(imputeTS))')
    r('suppressMessages(require(forecast))')
    robjects.globalenv['missing_data'] = norm_data
    robjects.globalenv['j'] = 2

    for sensor in range(norm_data.shape[1]):
        
        robjects.globalenv['j'] = sensor + 1
        r('x <- ts(missing_data[,j])')
        r('y <- na_ma(x, k = 4, weighting = "exponential", maxgap = Inf)')
        r('missing_data[, j] <- y')

    init_data = r('missing_data')
    numpy2ri.deactivate()
    return init_data

def ARIMA_init(norm_data, mask_flag):
    missing_data = np.array(norm_data)
    missing_data[np.where(mask_flag == 1)] = np.nan
    for sensor in range(missing_data.shape[1]):
        series_with_nan = missing_data[:,sensor]
    
        model = ARIMA(series_with_nan, order=(1,0,1))
        model_fit = model.fit()
        for i in range(missing_data.shape[0]):
            if mask_flag[i,sensor] == 1:
                missing_data[i,sensor] = model_fit.predict(start=i, end=i)[0]
    return missing_data


def AR_init(norm_data, mask_flag):
    missing_data = np.array(norm_data)
    missing_data[np.where(mask_flag == 1)] = np.nan
    for sensor in range(missing_data.shape[1]):
        series = missing_data[:,sensor]
        model = AutoReg(endog=series,lags=1, exog=None, missing="drop")
        model_fit = model.fit()

        for i in range(missing_data.shape[0]):
            if mask_flag[i,sensor] == 1:
                pre = model_fit.predict(start=i, end=i)
                if pre[0] != pre[0]:
                    continue
                missing_data[i, sensor] = pre[0]
    return missing_data

def AQ_CSDI_init(args):
    #return non-normalization data
    dataname = "AirQualityInit/AirQualityUCI_" + str(args.iter) + "_" + str(args.missing_percent) + ".csv"
    res = np.loadtxt(dataname, delimiter=",")
    return res


def Interpolation_init(norm_data, mask_flag):
    missing_data = np.array(norm_data)
    missing_data[np.where(mask_flag == 1)] = np.nan
    for i in range(missing_data.shape[1]):
        nans = np.isnan(missing_data[:, i])
        missing_data[nans, i] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), missing_data[~nans, i]) 
    return missing_data

def random_init(args, norm_data, mask_flag):
    np.random.seed(args.seed + args.iter)
    missing_data = norm_data
    missing_data = missing_data * (1-mask_flag)
    impute_data = np.random.rand(norm_data.shape[0], norm_data.shape[1]) * (mask_flag)
    return missing_data + impute_data

def normalization(data, mask_flag):
    x_shape = data.shape[0]
    min_list = []
    max_list = []
    for sensor_p in range(data.shape[1]):
        minn = 1000000000
        maxx = -1000000000
        for i in range(x_shape):
            #只计算非缺失数据的min max
            if mask_flag[i,sensor_p] == 1:
                continue
            minn = min(minn, data[i,sensor_p])
            maxx = max(maxx, data[i,sensor_p])
        min_list.append(minn)
        max_list.append(maxx)
    min_sensor_p = np.array(min_list)
    max_sensor_p = np.array(max_list) 
    norm_data = (data - min_sensor_p) / (max_sensor_p - min_sensor_p)
    return norm_data, min_sensor_p, max_sensor_p

def without_normalization(data, mask_flag):
    norm_data = np.array(data)
    min_sensor_p = np.zeros(norm_data.shape[1])
    max_sensor_p = np.ones(norm_data.shape[1])
    return norm_data, min_sensor_p, max_sensor_p

def get_MAR_mask_flag(args, org_data):
    random.seed(args.seed + args.iter)
    np.random.seed(args.seed + args.iter)

    mask_flag = np.zeros(org_data.shape)

    time_step_num = org_data.shape[0]
    missing_sensor_num = len(args.missing_column)
    missing_num_for_attr = int(time_step_num * args.missing_percent) 

    attribute_data = org_data[:,0]
    index = np.argsort(attribute_data) 
    rank = np.argsort(index) + 1 
    rank_sum = np.sum(rank)
    probability = rank / rank_sum

    for attr in args.missing_column:
        while np.sum(mask_flag[:, attr]) < missing_num_for_attr:
            # if np.sum(mask_flag[:, attr]) % 1000 == 0:
                # print(np.sum(mask_flag[:, attr]))
            mask_len = random.randint(1, args.max_missing_len)
            x = np.random.choice(range(time_step_num), p = probability.ravel())
    
            if canmask(x, attr, mask_len, mask_flag) == False:
                continue
            mask_flag[x, attr] = 1

            if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
            for i in range(x+1, min(time_step_num, x+mask_len)):
                mask_flag[i, attr] = 1
                if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                    break    
            if np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
    mask_flag[np.where(org_data == -200)] = 1

    return mask_flag

def get_MNAR_mask_flag(args, org_data):
    random.seed(args.seed + args.iter)
    np.random.seed(args.seed + args.iter)

    mask_flag = np.zeros(org_data.shape)

    time_step_num = org_data.shape[0]
    missing_sensor_num = len(args.missing_column)
    missing_num_for_attr = int(time_step_num * args.missing_percent) 

    attribute_data = org_data[:,0]
    index = np.argsort(attribute_data) 
    rank = np.argsort(index) + 1 
    rank_sum = np.sum(rank)
    probability = rank / rank_sum

    for attr in args.missing_column:
        while np.sum(mask_flag[:, attr]) < missing_num_for_attr:
            # if np.sum(mask_flag[:, attr]) % 1000 == 0:
                # print(np.sum(mask_flag[:, attr]))
            mask_len = random.randint(1, args.max_missing_len)

            attribute_data = org_data[:,attr]
            index = np.argsort(attribute_data) 
            rank = np.argsort(index) + 1 
            rank_sum = np.sum(rank)
            probability = rank / rank_sum

            x = np.random.choice(range(time_step_num), p = probability.ravel())
    
            if canmask(x, attr, mask_len, mask_flag) == False:
                continue
            mask_flag[x, attr] = 1

            if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
            for i in range(x+1, min(time_step_num, x+mask_len)):
                mask_flag[i, attr] = 1
                if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                    break    
            if np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
    mask_flag[np.where(org_data == -200)] = 1

    return mask_flag


def load_imu():
    data_org = pd.read_csv("Data/data_processed.csv", delimiter=";", index_col = False)

    reference = data_org.iloc[:, 1:7].to_numpy()
    data = data_org.iloc[:, 67:73].to_numpy()
    missing_p = np.where(data == 0)
    data[missing_p] = reference[missing_p]
    miss_flag = np.zeros_like(data)
    miss_flag[missing_p] = 1
    return data, miss_flag

def load_weer():
    datanamelist = ["uurgeg_260_1951-1960.txt", "uurgeg_260_1961-1970.txt", "uurgeg_260_1971-1980.txt", "uurgeg_260_1981-1990.txt", 
                    "uurgeg_260_1991-2000.txt", "uurgeg_260_2001-2010.txt", "uurgeg_260_2011-2020.txt","uurgeg_260_2021-2030.txt"]
    datanamelist += ["uurgeg_265_1951-1960.txt", "uurgeg_265_1961-1970.txt", "uurgeg_265_1971-1980.txt", "uurgeg_265_1981-1990.txt", 
                    "uurgeg_265_1991-2000.txt", "uurgeg_265_2001-2010.txt"]

    for name in datanamelist:
        data = []
        with open("Data/weer/" + name, 'r') as f:
            lines = f.readlines()[33:]
            for line in lines:
                line = line.strip().split(',')

                line = line[1:8]
                data.append(line)
        data = np.array(data)

        newname = name[0:-3] + "csv"
        np.savetxt("Data/weer/" + newname, data, delimiter=",", fmt='%s')


    datanamelist_260 = ["uurgeg_260_1951-1960.csv", "uurgeg_260_1961-1970.csv", "uurgeg_260_1971-1980.csv", "uurgeg_260_1981-1990.csv", 
                    "uurgeg_260_1991-2000.csv", "uurgeg_260_2001-2010.csv","uurgeg_260_2011-2020.csv","uurgeg_260_2021-2030.csv"]
    data_all_260 = []
    for dataname in datanamelist_260:
        data = pd.read_csv("Data/weer/" + dataname, header=None)
        data[0] =  data[0].astype(str)
        data[1] = data[1].apply(lambda x: f'{int(x):02}')
        data[1] = data[1].astype(str)
        time = data[0] + data[1]
        data["time"] = time
        data = data.drop(data.columns[[0, 1]], axis=1)
        cols = ['time'] + [col for col in data.columns if col != 'time']
        data = data[cols]
        data = data.to_numpy()

        data_all_260.append(data)

    data_all_260 = np.vstack(data_all_260)
    data_all_260 = np.where(data_all_260 == '     ', np.nan, data_all_260)
    data_all_260.astype(float)

    datanamelist_265 = ["uurgeg_265_1951-1960.csv", "uurgeg_265_1961-1970.csv", "uurgeg_265_1971-1980.csv", "uurgeg_265_1981-1990.csv", 
                    "uurgeg_265_1991-2000.csv", "uurgeg_265_2001-2010.csv"]
    data_all_265 = []
    for dataname in datanamelist_265:
        data = pd.read_csv("Data/weer/" + dataname, header=None)
        data[0] =  data[0].astype(str)
        data[1] = data[1].apply(lambda x: f'{int(x):02}')
        data[1] = data[1].astype(str)
        time = data[0] + data[1]
        data["time"] = time
        data = data.drop(data.columns[[0, 1]], axis=1)
        cols = ['time'] + [col for col in data.columns if col != 'time']
        data = data[cols]
        data = data.to_numpy()
        
        data_all_265.append(data)

    data_all_265 = np.vstack(data_all_265)
    data_all_265 = np.where(data_all_265 == '     ', np.nan, data_all_265)
    data_all_265.astype(float)

    data_get_gt = []
    data_mask_flag = []
    startindex = 0
    for i in range(data_all_265.shape[0]):
        if int(data_all_265[i, 0]) < 1970123008:
            continue

        mask_flag = np.zeros_like(data_all_265[i])
        data_gt = np.array(data_all_265[i])
        for j in range(data_all_265.shape[1]):
            if data_all_265[i,j] != data_all_265[i,j]:
                mask_flag[j] = 1
                for index_260 in range(startindex, data_all_260.shape[0]):
                    if data_all_265[i,0] == data_all_260[index_260, 0]:
                        data_gt[j] = data_all_260[index_260, j]
                        startindex = index_260
                        break
        data_get_gt.append(data_gt)
        data_mask_flag.append(mask_flag)

    data_get_gt = np.array(data_get_gt)
    data_mask_flag = np.array(data_mask_flag)
    return data_get_gt, data_mask_flag
