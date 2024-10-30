
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

random.seed(3407)
label_csv = pd.read_csv("in-hospital-mortality/train/listfile.csv", delimiter=",")

true_csv = []
false_csv = []
for index, row in label_csv.iterrows():
    if row.y_true:
        true_csv.append(row.stay)
    else:
        false_csv.append(row.stay)

true_subset = []
false_subset = []

for name in true_csv:
    df = pd.read_csv("in-hospital-mortality/train/" + name , delimiter=",")
    res = df[["Hours", "Heart Rate", "Mean blood pressure", "Oxygen saturation", "Respiratory rate", "Systolic blood pressure"]].to_numpy()

    missing_num = len(np.where(res != res)[0])
    missing_p = (missing_num / (res.shape[0] * res.shape[1]))
    if res.shape[0] > 100 and missing_p < 0.4:
        true_subset.append(name)

for name in false_csv:
    df = pd.read_csv("in-hospital-mortality/train/" + name , delimiter=",")
    res = df[["Hours", "Heart Rate", "Mean blood pressure", "Oxygen saturation", "Respiratory rate", "Systolic blood pressure"]].to_numpy()

    missing_num = len(np.where(res != res)[0])
    missing_p = (missing_num / (res.shape[0] * res.shape[1]))
    if res.shape[0] > 100 and missing_p < 0.2:
        false_subset.append(name)

print(len(true_subset)) #399
print(len(false_subset)) #403


temp_data = []
for name in true_subset:
    # print(name)
    df = pd.read_csv("in-hospital-mortality/train/" + name , delimiter=",")
    res = df[["Hours", "Heart Rate", "Mean blood pressure", "Oxygen saturation", "Respiratory rate", "Systolic blood pressure"]].to_numpy()[0:100]
    res[np.isnan(res)] = -200
    temp_data.append(res)

for name in false_subset:
    # print(name)
    df = pd.read_csv("in-hospital-mortality/train/" + name , delimiter=",")
    res = df[["Hours", "Heart Rate", "Mean blood pressure", "Oxygen saturation", "Respiratory rate", "Systolic blood pressure"]].to_numpy()[0:100]
    res[np.isnan(res)] = -200
    # np.savetxt("mimic/" + str(id) + ".csv", res, delimiter=",", fmt="%.6f")
    temp_data.append(res)

temp_data = np.vstack(temp_data)

print(temp_data.shape)

missing_pos = np.where(temp_data == -200)
temp_data = StandardScaler().fit_transform(temp_data)

print(len(missing_pos[0]) / (temp_data.shape[0]*temp_data.shape[1]))

temp_data[missing_pos] = -200

temp_data = temp_data.reshape(-1, 100, 6)

# 81 224 
# 478 595 675
# True: 0-396 
# False: 397-796

id = 0
for i in range(temp_data.shape[0]):
    if i ==81 or i == 224 or i==478 or i==595 or i==675:
        continue

    res = temp_data[i]
    np.savetxt("mimic/" + str(id) + ".csv", res, delimiter=",", fmt="%.6f")

    id += 1