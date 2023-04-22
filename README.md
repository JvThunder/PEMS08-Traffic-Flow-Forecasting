# PEMS08 Traffic Flow Forecasting 

[![stars - PEMS08-Traffic-Flow-Forecasting](https://img.shields.io/github/stars/JvThunder/PEMS08-Traffic-Flow-Forecasting?style=social)](https://github.com/JvThunder/PEMS08-Traffic-Flow-Forecasting)
[![forks - PEMS08-Traffic-Flow-Forecasting](https://img.shields.io/github/forks/JvThunder/PEMS08-Traffic-Flow-Forecasting?style=social)](https://github.com/JvThunder/PEMS08-Traffic-Flow-Forecasting)
[![GitHub - Brytista](https://img.shields.io/static/v1?label=GitHub&message=Brytista&color=lightgrey&logo=github)](https://github.com/Brytista)
[![GitHub - JvThunder](https://img.shields.io/static/v1?label=GitHub&message=JvThunder&color=lightgrey&logo=github)](https://github.com/JvThunder)
[![GitHub - sanstzu](https://img.shields.io/static/v1?label=GitHub&message=sanstzu&color=lightgrey&logo=github)](https://github.com/sanstzu)

*This is a mini-project for SC1015 (Introduction to Data Science & Artificial Intelligence) AY22/23 Semester 2.*

<p align="center">
    <img src="https://user-images.githubusercontent.com/26087840/233774521-bd770aef-d777-48f7-b098-825e1e837068.png" width="100%x"></img>
</p>


The contributors for this project are:
- **Bryan Atista Kiely** ([@Brytista](https://github.com/Brytista))
- **Clayton Fernalo** ([@sanstzu](https://github.com/sanstzu))
- **Joshua Adrian Cahyono** ([@JvThunder](https://github.com/JvThunder))

---

**Table of Contents**
1. **[Introduction](#introduction)**
2. **[Exploratory Data Analysis (EDA) and Data Preparation](#exploratory-data-analysis-eda-and-data-preparation)**
3. **[Methodology](#methodology)**
4. **[Evaluation](#evaluation)**
5. **[Conclusion](#conclusion)**
6. **[Limitations and Improvements](#limitations-and-improvements)**
7. **[References](#references)**

## Introduction
This project is focused on predicting the hourly traffic flow using the PEMS-08 Dataset (see [References](#references)). This dataset contains the traffic data of 170 locations in San Bernardino from July 2016 to August in 2016, recorded using a detector in a 5-minute interval. The given dataset is in a dimension of $(17856, 170, 3)$:
1. The first dimension ($17856$) refers to the number of 5-minute intervals data collected.
2. The second dimension ($170$) refers to the location of the data.
3. The third dimension ($3$) corresponds to three spatiotemporal features.

In total, there are $3\times 3035520 =  9106560$ cells.

The features that were given in the dataset are:
1. `flow`: number of vehicles pass through the detector in a time interval.
2. `occupy`: proportion of the time interval that the road was occupied by vehicle(s).
3. `speed`: average speed of the vehicles passing through the detector in a time interval.

From this, we deduced a question: which locations are congested at a particular hour and day? Our main goal is to create the rank of the locations in terms of traffic congestion. To be more specific, we believe that `occupy` encaptures the estimate traffic at a specific time, so `occupy` would be the feature we predict.

Formally, we formulated the following problem statement:
For $N$ prediction of the `occupy` variable and $K$ locations, make a predictive model to minimize $\sum_{t=0}^{N-1} \rho_t$, where $\rho_t$ is defined as the [Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) of the prediction values and the actual values.

There are real-life applications for this model, specifically for managing traffic. Traffic polices could consider their locations to patrol on a specific time based on the prediction that our model made, improving efficiency by allocating more resources on a accident-prone congested road. This model could also assist governments in making key policies such as road expansion to mitigated the heavily congested road.

## Exploratory Data Analysis (EDA) and Data Preparation
For the EDA, we performed several analysis on the dataset which includes:
- [Data Visualization](#data-visualization)
- [Correlation](#correlation)
- [Periodogram](#periodogram)


### Data Visualization
Judging from the distribution of `flow`, `occupy`, and `speed`, it seems that they follow a slightly-skewed normal distribution (the image is respectively shown below). In addition to normalized distribution, the wave pattern on `occupy` and `flow` signifies that there is a seasonality in those features. To find the exact seasonality in `occupy`, the feature we were trying predict, we would use a periodogram later on.

<img src="https://user-images.githubusercontent.com/26087840/233772662-f884bf73-9cd0-4522-b260-d04be554800a.png" width="480px"></img>
<img src="jttps://user-images.githubusercontent.com/26087840/233772665-1b323de8-c203-4820-ad50-43a87efa3565.png" width="480px"></img>
<img src="https://user-images.githubusercontent.com/26087840/233772669-0e4fc2cc-a906-4cf3-adb6-35d886e3efec.png" width="480px"></img>




### Correlation
From the correlation table, there is a strong correlation (above $0.7$) between the features. The correlation between the current and the future features $(t+1)$ is significant as well. 

<img src="https://user-images.githubusercontent.com/26087840/233602261-8622f2cd-b8bc-4964-a301-a95ccdda6f2f.png" width="720px"></img>


### Periodogram
Periodogram uses Discrete-time Fourier transform to examine the frequency/periodicity of a time series. This analysis proves the existence of a significant recurring pattern in `occupy`, in which it recurs daily. This make sense because of the number of people driving depends on the hour of the day, since people work and commute on a fixed schedule (e.g. busy hours after work will have higher traffic)

<img src="https://user-images.githubusercontent.com/26087840/233600486-30586112-de96-4e0a-a605-eee9bf70cf77.png" width="720px"></img>


### Data Preparation
- Due to the difference in the order of magnitude of each features, the input data for each of the features is normalized using `MinMaxScaler`.
- From the result of periodogram, an additional feature is appended to indicate the hour of which the data is recorded in format of one-hot encodings.
- The 5-minute interval data is averaged to form a 60-minute interval data (see [Limitations and Improvements](#limitations-and-improvements)).
- A lag step of $24$ previous values is appended as the input for the Neural Network.

In the end, we created $(1464, 24, 27, 170)$-array as an input, which will be converted into $(1462, 24, 4590)$-array, and $(1464, 170)-array as an output. Then, we splitted these data into a contiguous training data (72%), validation data (8%) and test data (20%).


## Methodology
To predict the time series, we used Neural Network from Keras library. Our model mainly utilizes the `Long-Short Term Memory (LSTM)` layer, due to its capability to remember information from earlier timesteps and gain information from their relation. In addition to `LSTM`, we also used the standard `Dense` layer, as well as `Dropout` layer to introduce noise to the model and reduce the chance of overfitting.

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

One thing to note is that `val_root_mean_squared_error` (validation RMSE) began to plateau around $120$ epochs whilst `root_mean_squared_error` (training RMSE) kept decreasing, thus we used a value close to that ($150$ epochs) to avoid the risk of overfitting.

<img src="https://user-images.githubusercontent.com/26087840/233721404-abb860b9-07f2-40ef-b0b5-1be5257377ab.png" width="480px"><img src="https://user-images.githubusercontent.com/26087840/233774627-bb93bbde-6c06-4a9e-b9d6-5070c4b5330d.png" width="480px">





## Evaluation

<img src="https://user-images.githubusercontent.com/26087840/233722022-2137727e-6200-4222-874d-802a7281892c.png" width="720px"></img>

Visually, we can see that our model managed to pick up the pattern on the dataset. To truly know whether our model is better than a random guessing, we quantitatively compared the model with a baseline, which would be Moving Average. The comparison was performed using these metrics:
1. RMSE (lower means better)
2. Spearman Correlation (higher means better)

### Training Data
|Metric|Baseline  Value|Prediction Value|
|-|-|-|
|RMSE|0.02385027817581988|0.01451811135755047|
|Spearman|0.7151104327649883|0.80599578910884|

Using the training dataset, the Spearman correlation and the RMSE of the model is better than the baseline. So we can conclude that our model managed to learn from the training set and not underfit. However, the true test lies on the test evaluation (since Neural Network models might overfit).

### Test Data
|Metric|Baseline  Value|Prediction Value|
|-|-|-|
|RMSE|0.026519227318993185|0.019782170402918516|
|Spearman|0.7208934552099153|0.8191585132042268|

The same pattern also exhibits when the test data is used instead, meaning that it does not overfit to the training set. Therefore, it is suffice to say that our model is neither underfitting nor overfitting.

## Conclusion
To conclude, we have made a predictive model using an LSTM neural network. We included hour and lag features, and also carefully scale and split the data. In the end, our model was able to predict both the train and test data better than our baseline which is moving average of previous values. We believe that this model can be further optimized and tested to help solve one of the real world issues, which is traffic management.

## Limitations and Improvements
Despite that our model is better than the baseline, there are several improvement that can be done to improve the performance:
1. **Lower the time interval**
    - The data that is used to train the model was converted from 5-minute interval to a 60-minute interval, sacrificing specific details.
    - The former interval one was not used due to hardware limitations ($17000 \text{ timesteps }\times 24 \text{ lag steps } \times 27 \text { features } \times 170 \text{ locations } \approx 14.98 \text { GB}$); our notebook platform (Kaggle) were only given 13GB of memory.
    - Using the latter one reduces the memory requirement by the factor of 12 ($\approx 1.25 \text{ GB}$).
2. **Larger lag steps**
    - By including more lag steps, there is a potential improvement on the accuracy of the `LSTM` network.
    - Similar to the previous issue, memory limitations prevented us from using higher timesteps ($>100$).
    - Although we were only using a timestep of $24$, the accuracy of the model is suffice in this scope.
3. **Vanishing Gradient Problem**
    - As the model uses more hidden layers and lag steps, the weight adjustment towards the earlier layers might become insignificant during the backpropagation, hindering the model in achieving higher performance.
    - This is an issue in `LSTM` layer, which is addressed in Attention Layers.



## References
- Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019). Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 922-929. https://doi.org/10.1609/aaai.v33i01.3301922.
- Black, S. (2020, November). What's happening in my LSTM layer? Towards Data Science. https://towardsdatascience.com/whats-happening-in-my-lstm-layer-dd8110ecc52f
- ChatGPT is used for debugging, clarification of concepts, and the writing of comments and descriptions.
- DALL-E 2 is used for generating the banner image in this README document.
- Special thanks to our Lab TA Ng Wen Zheng Terence for providing valuable feedbacks for our project.
    <img src="https://user-images.githubusercontent.com/26087840/233735987-30b47e33-90ab-4a2d-870a-8e5546578256.png" width="480px"></img>

