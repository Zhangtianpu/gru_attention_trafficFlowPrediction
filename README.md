# Gru_Attention_TrafficFlowPrediction
Several methods including GRU Dual Stage Attention Netwrok, GRU, LSTM, TCN were used in this work to predict traffic flow, share and speed on PeMS04 data.

# Directory and File Description
**Directory:** <br>
* data:Raw data <br>
* processed: Stored data which have been processed. The process includes data standardization through Z-score and historical features extraction. <br>
* *_model: A various of models to predict traffic task that have been trained and persisted in the way of pkl. <br>
* re: It stored prediction result in the way of csv, which given by these models. <br>

**File:**<br>
* main: The entry pint of the  program
* *_network: The structure of various models
* my_parameter: Some of settings about our models, such as various path definition, history window which decided how long period historical data I used

# Prediction Result Display

| Model | MSE| R Square |
|-------|----|-----|
|LSTM|27.928|0.875|
|GRU|25.579|0.911|
|GRU_Attention|22.626|0.934|
|GRU_Dual_Stage_Attention|21.913|0.937|
|TCN_Attention|26.530|0.909|

![image](https://github.com/Zhangtianpu/gru_attention_trafficFlowPrediction/blob/master/fig/traffic%20flow%20prediction%20with%20different%20models.jpg?raw=true)
# Reference
- [1] [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](http://arxiv.org/abs/1803.01271)
- [2] [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](http://arxiv.org/abs/1704.02971)
