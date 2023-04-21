# Traffic Flow Forecasting
This is a mini-project for SC1015 (Introduction to Data Science & Artificial Intelligence) AY22/23 Semester 2.
## Introduction
This focuses on predicting hourly traffic flow using the [`PEMS-08 Dataset`](https://doi.org/10.1609/aaai.v33i01.3301922). This dataset contains the traffic data of 170 locations in San Bernardino from July 2016 to August in 2016, recorded using a detector every 5-minute interval. The given dataset is in a dimension of $(17856, 170, 3)$:
1. The first dimension ($17856$) refers to the number of 5-minute intervals data collected.
2. The second dimension ($170$) refers to the location of the data.
3. The third dimension ($3$) corresponds to the `flow`, `occupy`, and `speed`.

The features that were given in the dataset are:
1. `flow`: number of vehicles pass through the detector in a time interval.
2. `occupy`: proportion of the time interval that the road was occupied by vehicle(s).
3. `speed`: average speed of the vehicles passing through the detector in a time interval.

From this, we deduced a question: which locations are congested at a particular hour and day? Our main goal is to create the rank of the locations in terms of traffic congestion. To be more specific, we believe that `occupy` encaptures the estimate traffic at a specific time, so `occupy` would be the feature we predict.

Formally, we formulated the following problem statement:
For $N$ prediction of the `occupy` variable and $K$ locations, make a predictive model to minimize $\sum_{t=0}^{N-1} \rho_t$, where $\rho_t$ is defined as the [Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) of the prediction values and the actual values.

## Exploratory Data Analysis (EDA) and Data Preparation
For the EDA, we performed several analysis on the dataset which includes:
- Data Visualization
- Correlation
- Periodogram


### Data Visualization
Judging from the distribution of `flow`, `occupy`, and `speed`, it seems that they follow a slightly-skewed normal distribution. Due to the difference in the order of magnitude of each individual features, we normalized the input data for each of the features using `MinMaxScaler`.

In addition to normalized distribution, the wave pattern on `occupy` and `flow` signifies that there is a seasonality in those features. To find the exact seasonality in `occupy`, which we would predict, we can use a periodogram later on.

### Correlation
From the correlation analysis, there is a significant correlation between the features. Of course in any time series prediction, lag features are used as the main features. Specifically, $k$ lag features means that we use the last $k$ values to predict the next value. So, it is a good measure to check the correlation of present $(t_n)$ and future $(t_{n+k})$ values.
The below shows the correlation between the current features and the future $(k=1)$
![image](https://user-images.githubusercontent.com/26087840/233602261-8622f2cd-b8bc-4964-a301-a95ccdda6f2f.png)

### Periodogram
The analysis using periodogram proves that the existence of a significant recurring pattern in `occupy`, in which it recurs daily.
![image](https://user-images.githubusercontent.com/26087840/233600486-30586112-de96-4e0a-a605-eee9bf70cf77.png)



From the EDA, we appended features to indicate the hour of which the data is recorded in format of one-hot encoding to improve our model.


## Methodology
To predict the time-series data, we uses Recurrent Neural Network (RNN). Our model utilizes `LSTM` layer, due to its capability to remember information from earlier timesteps and gain information from their relation. In addition to `LSTM`, we also used the standard `Dense` layer, as well as `Dropout` layer to introduce noise to the model and reduce the chance of overfitting.

Here is the details of the model (arranged from input to output):
| Layer Type  | Input Shape | Output Shape|
| - | - | - |
| LSTM (input) | **(24, 4590)** | (24, 256) |
| LSTM | (24, 256) | (256) |
| Dropout | (256) | (256)|
| Dense | (256) | (256) |
| Dropout | (256) | (256) |
| Dropout (output) | (256) | **(170)** |

We used `Mean Squared Error (MSE)` for the loss function of the training and `Root Mean Squared Error (RMSE)` as an additional metric (which would not be used for the training). For the optimizer, we used `Adam`.




### Hyperparameters
The training process used $150$ epochs and the standard batch size of $32$.

One thing to note is that `val_loss` began to plateau around $120$ epochs whilst `loss` kept decreasing, thus we used a value close to that ($150$ epochs) to avoid the risk of overfitting.

(to be uploaded model image once final)

## Evaluation
To test whether our model is better than a random guessing, we compared it with a baseline, that is the Moving Average. The comparison was performed using these metrics:
1. RMSE (lower means better)
2. Spearman Correlation (higher means better)

### Training Set
|Metric|Baseline  Value|Prediction Value|
|-|-|-|
|RMSE|0.02385027817581988|0.01451811135755047|
|Spearman Correlation|0.7151104327649883|0.80599578910884|

Using the training dataset, the Spearman correlation and the RMSE of the model is better than the baseline. So we can conclude that our model managed to learn from the training set and not underfit. However, the true test lies on the test evaluation (since Neural Network models might overfit).

### Test Set
|Metric|Baseline  Value|Prediction Value|
|-|-|-|
|RMSE|0.026519227318993185|0.019782170402918516|
|Spearman Correlation|0.7208934552099153|0.8191585132042268|

The same pattern also exhibits when the test data is used instead, meaning that it does not overfit to the training set. Therefore, it is suffice to say that our model is neither underfitting nor overfitting.
## Limitations and Improvements
Despite that our model better than the baseline, there are several improvement that can be done to improve the performance:
1. **Lower the time interval**
    - The data that is used to train the model was converted from 5-minute interval to a 60-minute interval, sacrificing specific details.
    - The former one was not used because of hardware limitations ($17000 \text{ timesteps }\times 24 \text{ lag steps } \times 27 \text { features } \times 170 \text{ locations } \approx 14.98 \text { GB}$), as using the latter one reduces the memory requirement by the factor of 12 ($\approx 1.25 \text{ GB}$).
2. **Larger lag steps**
    - By including more previous values, we could improve the accuracy of the `LSTM` model.
    - Identical to the previous one, hardware limitations prevented us from doing so. 
3. **Vanishing Gradient Problem**
    - As the model uses more hidden layers, the weight adjustment towards the earlier layers might become very small during the backpropagation, hindering the model in achieving higher performance.
    - This is an issue in `LSTM` layer.
    - This issue is addressed using Attention Layers.

## Contributors
- Bryan Atista Kiely (@Brytista)
- Clayton Fernalo (@sanstzu)
- Joshua Adrian Cahyono (@JvThunder)

## References
