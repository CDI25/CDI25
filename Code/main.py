
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from config import REGRESSION, IMPUTER, cfg 
import utils
import argparse 
import torch.multiprocessing as mp
import data_preprocess
from train import test, train

parser = argparse.ArgumentParser(description="models setting")
#global parameter
parser.add_argument('--dataset', dest='dataset', type=str, help='dataset name', default=cfg["dataset"])
parser.add_argument('--window_size', dest='window_size', type=int, help='temporal window size context', default=cfg["window_size"])
parser.add_argument('--max_missing_len', dest='max_missing_len', type=int, help='max missing length for missing position generation', default=cfg["max_missing_len"])
parser.add_argument('--missing_percent', dest='missing_percent', type=float, help='missing percent of generate missing data', default=cfg["missing_percent"])
parser.add_argument('--missing_column', nargs='+', type=int, default=cfg["missing_column"])
parser.add_argument('--used_regression_index', nargs='+', type=int, default=cfg["used_regression_index"])
parser.add_argument('--init', dest='init', help="fast init imputation method", type=str, default=cfg["init"])

#test parameter
parser.add_argument('--seed', dest='seed', type=int, help='base seed', default=cfg["base_seed"])
parser.add_argument('--iter', dest='iter', type=int, help='iter of test', default=cfg["iter"])


#regression pretrain parameter
parser.add_argument('--regression_learning_rate', dest='regression_learning_rate', type=float, help='learning rate of pretrain regression models', default=cfg[REGRESSION]["learning_rate"])
parser.add_argument('--regression_learning_epoch', dest='regression_learning_epoch', type=int, help='learning epoch of pretrain regression models', default=cfg[REGRESSION]["epoch"])

#imputation learing parameter
parser.add_argument('--impute_model', dest='impute_model', type=str, help='model type: ["FEW","HUGE"]', default=cfg[IMPUTER]["model"])
parser.add_argument('--impute_num_processes', dest='impute_num_processes', type=int, help='num of parallel processes', default=cfg[IMPUTER]["num_processes"])
parser.add_argument('--impute_learning_epoch', dest='impute_learning_epoch', type=int, help='impute training epoch', default=cfg[IMPUTER]["epoch"])
parser.add_argument('--impute_learning_rate', dest='impute_learning_rate', type=float, help='learning rate of impute model', default=cfg[IMPUTER]["learning_rate"])
parser.add_argument('--impute_reg_learning_rate', dest='impute_reg_learning_rate', type=float, help='learning rate of regression models in impute model', default=cfg[IMPUTER]["reg_learning_rate"])
parser.add_argument('--impute_skip_training', dest='impute_skip_training', type=float, help='skip training loss difference for training missing values', default=cfg[IMPUTER]["skip_training"])


if __name__ == '__main__':
    args = parser.parse_args()
    # print(args.impute_model)

    torch.manual_seed(args.seed + args.iter)
    np.random.seed(args.seed + args.iter)

    org_data = data_preprocess.get_org_data(args)

    # print(org_data.shape)
    mask_flag = data_preprocess.get_mask_flag(args, org_data)
    org_miss_postion = np.where(org_data == -200)


    norm_data, min_sensor_p, max_sensor_p = data_preprocess.normalization(org_data, mask_flag)
    init_data = data_preprocess.EWMA_init(norm_data, mask_flag)

    #####################################
    #           count time              #
    st = time.time()                    #
    #                                   #
    #####################################

    regression_list = utils.Regression_models(args, norm_data, mask_flag).regression_list
    cmdi_data = utils.CMDI_DATA(args, norm_data, mask_flag)
    cmdi_model = utils.CMDI(args, init_data, mask_flag)
    cmdi_model.share_memory()
    cmdi_model.train()
    if args.impute_model == "HUGE":
        for sensor in range(norm_data.shape[1]):
            if sensor not in args.used_regression_index:
                continue
            regression_list[sensor].train()
            regression_list[sensor].share_memory()
    processes = []
    #####################################
    #           count time              #
    st = time.time()                    #
    #                                   #
    #####################################

    for rank in range(args.impute_num_processes):
        p = mp.Process(target=train, args=(args, cmdi_model, regression_list, norm_data, cmdi_data.input_of_processor[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    #############################################################
    #                        count time                         #
    training_cost = time.time()-st
    print("ALL Training Time:", training_cost)                 #
    #                                                           #
    #############################################################

    f1 = open("result_time.txt","a")
    print(training_cost, file=f1)

    mask_flag[org_miss_postion] = 0
    eval_result = test(args, cmdi_model, min_sensor_p, max_sensor_p, org_data, mask_flag)


    f2 = open("result.txt","a")
    print(eval_result)
    print(eval_result, file=f2)