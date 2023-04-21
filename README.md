# Traffic Flow Forecasting
This is a mini-project for SC1015 (Introduction to Data Science & Artificial Intelligence) AY22/23 Semester 2.
## Introduction
This focuses on predicting hourly traffic flow using the [`PEMS-08 Dataset`](https://doi.org/10.1609/aaai.v33i01.3301922). This dataset contains the traffic data of 170 locations in San Bernardino from July 2016 to August in 2016, recorded using a detector every 5 minutes.

The features that were given in the dataset are:
1. `flow`: number of vehicles pass through the detector during the 5 minute interval.
2. `occupancy`: proportion of the time interval that the road was occupied by vehicle(s).
3. `speed`: average speed of the vehicles passing through the detector during the time interval.

From this, we deduced a question: which locations are congested at a particular hour and day? Our main goal is to create the rank of the locations in terms of traffic congestion. To be more specific, we believe that `occupy` encaptures the estimate traffic at a specific time, so this would be the `feature` we would predict.

Formally, we formulated the following problem statement:
For $N$ prediction of the `occupy` variable and $K$ locations, make a predictive model to minimize $\sum_{t=0}^{N-1} \rho_t$, where $\rho_t$ is defined as the [Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) of the prediction values and the actual values.

## Exploratory Data Analysis (EDA) and Data Preparation
For the EDA, we performed several analysis on the dataset which includes:
- Data Visualization
- Correlation
- Periodogram

From the correlation analysis (between $feature_t$ and $feature_{t+1}$), there is a significant correlation between the cuurrent and future features.
![image](https://user-images.githubusercontent.com/26087840/233602261-8622f2cd-b8bc-4964-a301-a95ccdda6f2f.png)


From the analysis using periodogram, we found out that the there is a significant pattern in the `occupy` variable in a daily interval, and a less significant pattern in a weekly interval. 
![image](https://user-images.githubusercontent.com/26087840/233600486-30586112-de96-4e0a-a605-eee9bf70cf77.png)

From the EDA, we appended features to indicate the hour and the day of which the data is recorded in format of one-hot encoding to improve our data. As an additional improvement, we scaled the input data for each of the features using `MinMaxScaler`.


## Models Training and Evaluation
Our model utilizes `LSTM` layer, due to its capability to remember information from earlier timesteps and gain information from their relation. We used `Mean Squared Error (MSE)` for the loss and `Root Mean Squared Error (RMSE)` as the metric (which would not be used for backpropagation and only serves a purpose of measuring how well the prediction in that iteration).

(to be uploaded model image once final)

Using the test set, our model achieves a `RMSE` of 

## Contributors
- Bryan Atista Kiely (@Brytista)
- Clayton Fernalo (@sanstzu)
- Joshua Adrian Cahyono (@JvThunder)

## References
