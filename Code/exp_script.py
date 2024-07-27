
import os
python_dir = "C:/Users/DELL/AppData/Local/Programs/Python/Python38/python.exe "
file_dir = "d:/cmdi/main.py "

missing_percents = ["0.1 ", "0.2 ", "0.3 ", "0.4 "]

for missing_percent in missing_percents:
    for iter in range(5):                                
        parameter = "--iter " + str(iter) + " "

        parameter += "--missing_percent " + missing_percent + " "
        parameter += "--impute_model " + "FEW" + " "
        cmd = python_dir + file_dir + parameter
        print(cmd)
        os.system(cmd)

    print("FEW")
    res = 0.0
    with open("result.txt", "r") as f:
        lines = f.readlines()
        for line in lines[-5:]:
            line = float(line.strip("\n"))
            res += line
    print("result: ", res/5.0)
    print("*"* 100)

    res_time = 0.0
    with open("result_time.txt", "r") as f:
        lines = f.readlines()
        for line in lines[-5:]:
            line = float(line.strip("\n"))
            res_time += line

    print("time: ", res_time/5.0)
    print("*"* 100)


    for iter in range(5):                                
        parameter = "--iter " + str(iter) + " "
        parameter += "--missing_percent " + missing_percent + " "
        parameter += "--impute_model " + "HUGE" + " "
        cmd = python_dir + file_dir + parameter
        print(cmd)
        os.system(cmd)

    print("HUGE")
    res = 0.0
    with open("result.txt", "r") as f:
        lines = f.readlines()
        for line in lines[-5:]:
            line = float(line.strip("\n"))
            res += line
    print("result: ", res/5.0)
    print("*"* 100)

    res_time = 0.0
    with open("result_time.txt", "r") as f:
        lines = f.readlines()
        for line in lines[-5:]:
            line = float(line.strip("\n"))
            res_time += line

    print("time: ", res_time/5.0)
    print("*"* 100)
