
import torch
import torch.nn as nn
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os

def train(args, cmdi_model, regression_list, norm_data, contexts_set):
    torch.manual_seed(args.seed + args.iter)
    np.random.seed(args.seed + args.iter)

    getpid_X = os.getpid()

    criterion = nn.MSELoss(reduction="mean")

    if args.impute_model == "HUGE":
        reg_model_parameterlist = None 
        for reg_model in regression_list:
            if reg_model == None:
                continue
            if reg_model_parameterlist == None:
                reg_model_parameterlist = list(reg_model.parameters())
            else:
                reg_model_parameterlist = reg_model_parameterlist + list(reg_model.parameters())
        optimizer_reg = torch.optim.Adam(reg_model_parameterlist, lr = args.impute_reg_learning_rate)    
    optimizer_impute = torch.optim.Adam(cmdi_model.parameters(), lr = args.impute_learning_rate)

    contexts_for_p = []
    missing_flags_for_p = []
    missing_poss_for_p = []

    for sensor in range(norm_data.shape[1]):
        if sensor not in args.used_regression_index:
            contexts_for_p.append(None)
            missing_flags_for_p.append(None)
            missing_poss_for_p.append(None)
            continue
        contexts_for_p.append(torch.tensor(contexts_set[sensor][0]))
        missing_flags_for_p.append(np.array(contexts_set[sensor][1]))
        missing_poss_for_p.append(contexts_set[sensor][2])

    # print("Already get context!")
    #####################################
    #           count time              #
    st = time.time()                    #
    #                                   #
    #####################################
    
    lastloss = 1000000.0
    __losslist = []

    outepoch = 0
    for epoch in range(args.impute_learning_epoch):
        if args.impute_model == "HUGE":
            optimizer_reg.zero_grad()
        optimizer_impute.zero_grad()
        filled_contexts, __, __ = cmdi_model(args, contexts_for_p, missing_flags_for_p, missing_poss_for_p)
        total_loss = torch.tensor(0).float()
        for sensor_p in range(norm_data.shape[1]):
            if sensor_p not in args.used_regression_index:
                continue
            filled_context = filled_contexts[sensor_p].float()
            label = filled_context[:,-1].unsqueeze(1)
            input = filled_context[:,:-1]
            predict = regression_list[sensor_p](input)
        
            total_loss += criterion(predict,label)
    
        # if epoch % 20 == 0 or epoch == args.impute_learning_epoch-1:
        #     print(epoch, total_loss)
        # print(epoch, total_loss)
        
        total_loss = total_loss / norm_data.shape[1]
        total_loss.backward()
        __losslist.append(total_loss.item())

        
        if args.impute_model == "HUGE":
            optimizer_reg.step()
        optimizer_impute.step()

        
        # print(epoch, getpid_X, total_loss.item(), abs(total_loss.item() - lastloss))
        outepoch = epoch


        if abs(total_loss.item() - lastloss) < args.impute_skip_training:
            break
        lastloss = total_loss.item()
        
    # print(outepoch)
    #############################################################
    #                        count time                         #
    # print("Training Time of one porcessor:", time.time()-st)    #
    #                                                           #
    #############################################################

def test(args, cmdi_model, min_sensor_p, max_sensor_p, org_data, mask_flag):
    cmdi_model.eval()
    learning_cell = cmdi_model.learning_cell.detach().numpy()
    cell_id_to_missing_pos = cmdi_model.cell_id_to_missing_pos
    predict, target = [],[]

    impute_data = org_data.copy()
    for id in range(learning_cell.shape[0]):
        missing_pos = cell_id_to_missing_pos[id]
        if mask_flag[missing_pos[0], missing_pos[1]] == 1:
            target.append(org_data[missing_pos[0], missing_pos[1]])
            predict.append(learning_cell[id] * (max_sensor_p[missing_pos[1]] - min_sensor_p[missing_pos[1]]) + min_sensor_p[missing_pos[1]])
        impute_data[missing_pos[0], missing_pos[1]] = learning_cell[id] * (max_sensor_p[missing_pos[1]] - min_sensor_p[missing_pos[1]]) + min_sensor_p[missing_pos[1]]

    res = np.sqrt(((np.array(target) - np.array(predict)) ** 2).mean())
    return res
