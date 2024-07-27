
cfg = {}

REGRESSION = "reg"
cfg[REGRESSION] = {}
cfg[REGRESSION]["learning_rate"] = 1e-1
cfg[REGRESSION]["epoch"] = 1000

#energy few: learning_rate=1e-2 huge:learning_rate=1e-2 reg_learning_rate=5e-4 epoch=50  skip_training=1e-7
#location few: learning_rate=5e-6 skip_training=1e-12 ; huge: learning_rate=5e-6 reg_learning_rate=0.0001  epoch=100  skip_training=1e-12
#AirQualityUCI few: dlearning_rate=1e-2 skip-traing 1e-4 epoch30; huge: learning_rate=1e-2 reg_learning_rate=1e-2 skip-traing 1e-4 epoch=30   init=ARIMA
#Ethanol few: learning_rate=1e-3 skip_training=1e-12 ; huge: learning_rate=1e-3 reg_learning_rate=1e-3  epoch=50  skip_training=1e-12

IMPUTER = "cmdi"
cfg[IMPUTER] = {}
cfg[IMPUTER]["model"] = "FEW"
cfg[IMPUTER]["num_processes"] = 1
cfg[IMPUTER]["epoch"] = 30
cfg[IMPUTER]["learning_rate"] = 1e-3
cfg[IMPUTER]["reg_learning_rate"] = 1e-3
cfg[IMPUTER]["skip_training"] = 1e-4
cfg["dataset"] = "AirQualityUCI" 

cfg["init"] = "EWMA"
cfg["iter"] = 0
cfg["missing_percent"] = 0.2
cfg["base_seed"] = 3407  #final seed == base_seed + iter

cfg["window_size"] = 1
cfg["max_missing_len"] = 8

cfg["missing_column"] = [i for i in range(13)]
cfg["used_regression_index"] = [i for i in range(13)] 