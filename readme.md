# Collaborative Imputation for Multivariate Time Series with Convergence Guarantee


## File Structure

+ Code: source code of our implementation
+ Data: some source files of datasets used in experiments
+ Appendix.pdf: 
  - complete proofs for all the theoretical results in the manuscript, including Propositions 1,2,3,4,6,7, and Lemma 5; 
  - empirical evidence of the L-smooth assumption; 
  - memory consumption of our comparison experiments;
  - averaged maximal self-connected contexts number of various scenarios；
  - averaged imputation rounds of various scenarios；
  - intratemporal and intertemporal patterns in each dataset
+ Full.pdf: the full version of our manuscript
+ requirments.txt: python version and libraries requirements

## Demo Script Running
```
python main.py --dataset AirQualityUCI ---iter 0 --missing_percent 0.1  --impute_model HUGE --missing_column 0 1 2 3 4 5 6 7 8 9 10 11 12 --used_regression_index 0 1 2 3 4 5 6 7 8 9 10 11 12 --impute_learning_rate 1e-2 --impute_reg_learning_rate 1e-2 --impute_skip_training 1e-4 --impute_learning_epoch 30 --init IPL
```

```
python main.py --dataset energy ---iter 0 --missing_percent 0.1  --impute_model HUGE --missing_column 0 1 2 3 4 5 6 7 8 --used_regression_index 0 1 2 3 4 5 6 7 8 --impute_learning_rate 1e-2 --impute_reg_learning_rate 5e-4 --impute_skip_training 1e-7 --impute_learning_epoch 50 --init IPL
```

## Running a Dataset for All Cases:
```
# change the parameters in config.py
# change the python path and file path in exp_script.py
# runnning the script
python exp_script.py
```

## Dataset Source
* Energy: https://archive.ics.uci.edu/ml/datasets/Appliances%20energy%20prediction
* Ethanol: http://www.timeseriesclassification.com/description.php?Dataset=EthanolConcentration
* AirQuality: https://archive.ics.uci.edu/ml/datasets/Air+Quality
* MIMIC-III: https://physionet.org/content/mimiciii/1.4/
* GPS: GPS time series dataset collected by us
* IMU: https://github.com/dusan-nemec/mems-calib/tree/master
* Weer: https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens/

# Ground truth of real missing values
* GPS: We have repaired the original GPS data according to the actual location. The corrected file is "location_ground_truth.csv", while the original file with missing data is "location_missing.csv". In this context, the value “-200” indicates that the observation is missing.
* IMU: The original data is contained in "data_processed.csv". We utilized readings from higher-precision sensors to establish the ground truth. The processing details can be found in "data_preprocess.py".
* Weer: The original data is located in the "weer" folder. We used readings from the nearest station to determine the ground truth. The related processing information is also available in "data_preprocess.py".



### Preprocessing the MIMIC-III dataset
*Notably, due to the copyright issue, we cannot directly release the MIMIC-III dataset here. In order to get the form of the MIMIC-III dataset we are using, the authors need to*
1. Download the MIMIC-III dataset from the source (https://physionet.org/content/mimiciii/1.4/).
2. Process the dataset according to the existing work (https://github.com/YerevaNN/mimic3-benchmarks).
3. Run MIMICIII_process.py in the  ''Code'' folder to get the data that we actually use.
