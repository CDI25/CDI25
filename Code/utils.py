
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#Regression参数组成：先Left-Hand，再不同timestep
class Regression(nn.Module):
    def __init__(self, args, sensor_num, window_size):
        torch.manual_seed(args.seed + args.iter)

        super(Regression, self).__init__()
        self.fc = nn.Linear(sensor_num-1 + 2*window_size, 1)
        # self.sm = nn.Sigmoid()
    def forward(self, input):
        output = self.fc(input)
        return output

class CMDI(nn.Module):
    def __init__(self, args, init_data, mask_flag):
        torch.manual_seed(args.seed + args.iter)
        
        super(CMDI, self).__init__()
        __missing_num = np.sum(mask_flag)

        __initcell = []
        __missing_pos = []
        for i in range(mask_flag.shape[0]):
            for sensor_p in range(mask_flag.shape[1]):
                if mask_flag[i,sensor_p] == 1:
                    __initcell.append((init_data[i, sensor_p]))
                    __missing_pos.append((i,sensor_p))

        self.learning_cell = torch.nn.Parameter(torch.FloatTensor(__initcell), requires_grad=True)
        self.cell_id_to_missing_pos = dict(zip(range(int(__missing_num)), __missing_pos))
        self.missing_pos_to_cell_id = dict(zip(__missing_pos, range(int(__missing_num))))
        
    def forward(self, args, contexts_for_p, missing_flags_for_p, missing_poss_for_p):
        filled_contexts = []
        for sensor_p in range(len(contexts_for_p)):
            if sensor_p not in args.used_regression_index:
                filled_contexts.append(None)
                continue
            context = contexts_for_p[sensor_p].float()
            missing_flag = missing_flags_for_p[sensor_p]
            missing_pos = missing_poss_for_p[sensor_p]
            for i in range(context.shape[0]):
                missing_pos_of_i = missing_pos[i]
                for j in range(context.shape[1]):
                    if missing_flag[i,j] == 1:
                        context[i,j] = self.learning_cell[self.missing_pos_to_cell_id[missing_pos_of_i[j]]]
            filled_contexts.append(context)
        return filled_contexts, self.learning_cell, self.cell_id_to_missing_pos

class Regression_models():
    def __init__(self, args, norm_data, mask_flag):
        torch.manual_seed(args.seed + args.iter)

        self.regression_list = []
        for i in range(norm_data.shape[1]):
            if i in args.used_regression_index:
                self.regression_list.append(Regression(args, norm_data.shape[1], args.window_size))
            else:
                self.regression_list.append(None)

        # self.__lasso_training(args, norm_data, mask_flag)
        self.__training(args, norm_data, mask_flag)

    def __have_missing(self, i:int, j:int, args, mask_flag):
        for sensor in range(0, mask_flag.shape[1]):
            if mask_flag[i, sensor] == 1:
                return True
        for time in range(i - args.window_size, i + args.window_size + 1):
            if mask_flag[time, j] == 1:
                return True
        return False
    
    def __get_input_for_postion(self, i:int, j:int, args, norm_data):
        list = []
        #先Left-Hand，再不同timestep
        for sensor in range(0, norm_data.shape[1]):
            if sensor == j:
                continue
            list.append(norm_data[i,sensor])
        for time in range(i - args.window_size, i + args.window_size + 1):
            if time == i:
                continue
            list.append(norm_data[time, j])
        return np.array(list)

    def __get_training_data_for_gp(self, sensor_p:int, args, norm_data, mask_flag):
        training_data = []
        label = []
        for i in range(args.window_size, norm_data.shape[0] - args.window_size):
            if self.__have_missing(i, sensor_p, args, mask_flag) == False:
                training_data.append(self.__get_input_for_postion(i,sensor_p, args, norm_data))
                label.append(norm_data[i, sensor_p])
        return torch.tensor(training_data).float(), torch.tensor(label).float().unsqueeze(1)
    
    def __training(self, args, norm_data, mask_flag):
        for sensor_p in range(norm_data.shape[1]):
            if sensor_p not in args.used_regression_index:
                continue

            training_data, label = self.__get_training_data_for_gp(sensor_p, args, norm_data, mask_flag)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.regression_list[sensor_p].parameters(), lr = args.regression_learning_rate, weight_decay=1e-6)
            self.regression_list[sensor_p].train()
            for epoch in range(args.regression_learning_epoch):
                optimizer.zero_grad()
                predict = self.regression_list[sensor_p](training_data)
                loss = criterion(predict, label)
                loss.backward()
                optimizer.step()
                # if epoch % 100 == 0 or epoch == args.regression_learning_epoch-1:
                #     print(loss.item())
            self.regression_list[sensor_p].eval()
    def __lasso_training(self, args, norm_data, mask_flag):
        for sensor_p in range(norm_data.shape[1]):
            if sensor_p not in args.used_regression_index:
                continue

            training_data, label = self.__get_training_data_for_gp(sensor_p, args, norm_data, mask_flag)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.regression_list[sensor_p].parameters(), lr = args.regression_learning_rate, weight_decay=1e-6)
            self.regression_list[sensor_p].train()
            for epoch in range(args.regression_learning_epoch):
                optimizer.zero_grad()
                predict = self.regression_list[sensor_p](training_data)
                fn = criterion(predict, label)
                
                l1_penalty = 1 * sum([p.abs().sum() for p in self.regression_list[sensor_p].fc.parameters()])
                loss = fn + l1_penalty
                loss.backward()
                optimizer.step()
                # if epoch % 100 == 0 or epoch == args.regression_learning_epoch-1:
                #     print(loss.item())
            self.regression_list[sensor_p].eval()

class CMDI_DATA():
    def __init__(self, args, norm_data, mask_flag):
        self.input_of_processor = None
        if args.impute_model == "FEW":
            self.input_of_processor = self.__distribute_data(args, norm_data, mask_flag)
        else:
            self.input_of_processor = self.__distribute_data(args, norm_data, mask_flag)

    def __get_missing_pos_id(self, mask_flag):
        missing_pos_id_list = []
        for i in range(mask_flag.shape[0]):
            for j in range(mask_flag.shape[1]):
                if mask_flag[i,j] == 1:
                    missing_pos_id_list.append(i * mask_flag.shape[1] + j)
        return missing_pos_id_list

    def __is_connect(self, ida, idb, args, norm_data):
        ax = ida // norm_data.shape[1]
        ay = ida - ax * norm_data.shape[1]
        bx = idb // norm_data.shape[1]
        by = idb - bx * norm_data.shape[1]
        
        if ax == bx:
            return True
        if ay == by:
            if ay in args.used_regression_index:
                if abs(ax - bx) <= 2 * args.window_size:
                    return True
        if (ay in args.used_regression_index) or (by in args.used_regression_index):
            if abs(ax - bx) <= args.window_size:
                return True
        return False

    def __get_context_of_pos(self, i, sensor_p, args, norm_data, mask_flag):
        #(sensor_num-1):LHS  + windowsize*2: temporal relation + label
        pos_num = norm_data.shape[1] + 2 * args.window_size
        
        context = np.zeros(pos_num)
        missing_flag = np.zeros(pos_num)
        missing_pos = []
        
        cnt = 0

        for sensor in range(0, norm_data.shape[1]):
            if sensor == sensor_p:
                continue
            context[cnt] = norm_data[i, sensor]
            if mask_flag[i, sensor] == 1:
                missing_flag[cnt] = 1      
                missing_pos.append((i,sensor))
            else:
                missing_pos.append((-1,-1))
            cnt += 1
        for time in range(i - args.window_size, i+args.window_size + 1):
            if time == i:
                continue
            context[cnt] = norm_data[time, sensor_p]
            if mask_flag[time, sensor_p] == 1:
                missing_flag[cnt] = 1
                missing_pos.append((time, sensor_p))
            else:
                missing_pos.append((-1,-1))
            cnt += 1
        context[cnt] = norm_data[i, sensor_p]
        if mask_flag[i, sensor_p] == 1:
            missing_flag[cnt] = 1
            missing_pos.append((i, sensor_p)) 
        else:
            missing_pos.append((-1,-1))
        return context, missing_flag, missing_pos
    
    def __get_contexts_for_missing_pos(self, takeout, pos_id, args, norm_data, mask_flag):
        x = pos_id // norm_data.shape[1]
        y = pos_id - x * norm_data.shape[1]
        res = []
        #res中共sensor_num块信息，分别代表了以每个sensor为RHS的contexts

        for sensor in range(norm_data.shape[1]):
            if sensor not in args.used_regression_index:
                res.append(None)
                continue

            info_of_sensor = [[],[],[]]
            #每个sensor 3个信息: context, missing_flag, absolute missing position
            res.append(info_of_sensor)   
        
        #第y个column上有回归模型，寻找以x时刻为中心 上下时刻的context
        if y in args.used_regression_index:
            for i in range(x - args.window_size, x + args.window_size+1):
                if i >= args.window_size and i < (norm_data.shape[0] - args.window_size) and takeout[i,y]==0:
                    context, missing_flag, missing_pos = self.__get_context_of_pos(i, y, args, norm_data, mask_flag)
                    res[y][0].append(context)
                    res[y][1].append(missing_flag)
                    res[y][2].append(missing_pos)
                    takeout[i,y] = 1
        #在x时刻寻找所有context
        for sensor in range(0, norm_data.shape[1]):
            if sensor not in args.used_regression_index:
                continue

            if takeout[x,sensor]==0 and sensor != y :
                context, missing_flag, missing_pos = self.__get_context_of_pos(x, sensor, args, norm_data, mask_flag)
                res[sensor][0].append(context)
                res[sensor][1].append(missing_flag)
                res[sensor][2].append(missing_pos)
                takeout[x,sensor] = 1
        return res

    #union-find disjoint sets
    def __find(self, x, father_of_missing_pos):
        if x == father_of_missing_pos[x]:
            return x
        else:
            father_of_missing_pos[x] = self.__find(father_of_missing_pos[x], father_of_missing_pos)
            return father_of_missing_pos[x]
    #union-find disjoint sets
    def __unite(self, x, y, father_of_missing_pos,Rank):
        x = self.__find(x, father_of_missing_pos)
        y = self.__find(y, father_of_missing_pos)
        if x == y:
            return
        if Rank[x] <= Rank[y]:
            father_of_missing_pos[x] = y
        else:
            father_of_missing_pos[y] = x
        if Rank[x] == Rank[y]:
            Rank[x] += 1

    def __get_maximum_connected_contexts_set(self, args, norm_data, mask_flag):
        missing_pos_id_list = self.__get_missing_pos_id(mask_flag)

        #union-find disjoint sets
        father_of_missing_pos = [i for i in range(len(missing_pos_id_list))]
        Rank = [1 for i in range(len(missing_pos_id_list))]
        for i in range(len(missing_pos_id_list)):
            posid = missing_pos_id_list[i]
            for j in range(max(0, i - 2 * args.window_size * norm_data.shape[1]), i):
                preid = missing_pos_id_list[j]
                if self.__is_connect(posid, preid, args, norm_data):
                    self.__unite(i, j, father_of_missing_pos, Rank)
                    self.__find(i, father_of_missing_pos)
                    self.__find(j, father_of_missing_pos)            
        
        father_set = set(father_of_missing_pos)
        father2maxid = dict(zip(list(father_set), range(len(father_set))))

        #flag of take out
        takeout = np.zeros(norm_data.shape)
        takeout[0: args.window_size] = 1
        takeout[norm_data.shape[0] - args.window_size : norm_data.shape[0]] = 1
        #maximum_connected_context: item number == number of maximum connected sets
        #maximum_connected_context[i]: item number==sensor_num   
        #maximum_connected_context[i][sensor_num]: [[contexts list],[mask flag list],[pos list]]
        maximum_connected_contexts_set = []
        size_of_maximum_sets = []
        for i in range(len(father_set)):
            one_set = []
            for sensor in range(norm_data.shape[1]):
                if sensor not in args.used_regression_index:
                    one_set.append(None)
                    continue
                one_set.append([[], [], []])
            maximum_connected_contexts_set.append(one_set)
            size_of_maximum_sets.append(0)

        for i in range(len(father_of_missing_pos)):
            missing_pos_id = missing_pos_id_list[i]
            father_of_missing_pos_id = father_of_missing_pos[i]
            maxmial_set_id = father2maxid[father_of_missing_pos_id]

            ax = missing_pos_id // norm_data.shape[1]
            ay = missing_pos_id - ax * norm_data.shape[1]
            if ay in args.used_regression_index:
                size_of_maximum_sets[maxmial_set_id] += 2 * args.window_size + len(args.used_regression_index)-1
            else:
                size_of_maximum_sets[maxmial_set_id] += len(args.used_regression_index)


            contexsinfor_for_missing_pos = self.__get_contexts_for_missing_pos(takeout, missing_pos_id,  args, norm_data, mask_flag)
            for sensor in range(norm_data.shape[1]):
                if sensor not in args.used_regression_index:
                    continue
                maximum_connected_contexts_set[maxmial_set_id][sensor][0] += contexsinfor_for_missing_pos[sensor][0]
                maximum_connected_contexts_set[maxmial_set_id][sensor][1] += contexsinfor_for_missing_pos[sensor][1]
                maximum_connected_contexts_set[maxmial_set_id][sensor][2] += contexsinfor_for_missing_pos[sensor][2]

        return maximum_connected_contexts_set, size_of_maximum_sets

    def __distribute_data(self, args, norm_data, mask_flag):
        maximum_connected_contexts_set, size_of_maximum_sets = self.__get_maximum_connected_contexts_set(args, norm_data, mask_flag)
        # print("size_of_maximum_sets:", size_of_maximum_sets)
        
        # makespan by greedy strategy

        input_of_processor = []
        sample_num_of_processor = []
        avg = np.sum(size_of_maximum_sets) / args.impute_num_processes
        remain_processor = args.impute_num_processes
        get_flag = np.zeros(len(maximum_connected_contexts_set))
        for set_id in range(len(maximum_connected_contexts_set)):
            if size_of_maximum_sets[set_id] >= avg:
                input_of_processor.append(maximum_connected_contexts_set[set_id])
                sample_num_of_processor.append(size_of_maximum_sets[set_id])
                get_flag[set_id] = 1
                remain_processor -= 1
        for i in range(remain_processor):
            for set_id in range(len(maximum_connected_contexts_set)):
                if get_flag[set_id] == 0:
                    input_of_processor.append(maximum_connected_contexts_set[set_id])
                    sample_num_of_processor.append(size_of_maximum_sets[set_id])
                    get_flag[set_id] = 1
                    break
        for set_id in range(len(maximum_connected_contexts_set)):
            if get_flag[set_id] == 0:
                minn = 100000000
                min_processor = None
                for i in range(args.impute_num_processes):
                    if minn > sample_num_of_processor[i]:
                        minn = sample_num_of_processor[i]
                        min_processor = i
                for sensor in range(norm_data.shape[1]):
                    if sensor not in args.used_regression_index:
                        continue
                    input_of_processor[min_processor][sensor][0] += maximum_connected_contexts_set[set_id][sensor][0]
                    input_of_processor[min_processor][sensor][1] += maximum_connected_contexts_set[set_id][sensor][1]
                    input_of_processor[min_processor][sensor][2] += maximum_connected_contexts_set[set_id][sensor][2]
                sample_num_of_processor[min_processor] += size_of_maximum_sets[set_id]

        # print(sample_num_of_processor)
        return input_of_processor