import data_process
import numpy as np
import my_parameter
import pandas as pd
import torch
import lstm_network
import evaluation
import gru_network
import GRU_attention_network
import GRU_dual_stage_attention_network
import TCN_network
import matplotlib.pyplot as plt




if __name__ == '__main__':
    """数据读取，及数据处理，标准化"""
    # data=np.load(my_parameter.RAW_PATH)['data'][:,0,:]
    # data=data_process.scal_data(data)
    # data_df=pd.DataFrame(data,columns=['traffic_flow','share','speed'])
    # data_df=data_process.get_history_data(data_df,columns=['traffic_flow','share','speed'])

    data_df=pd.read_csv(my_parameter.PROCESS_PATH)
    print(data_df.shape)

    """
    划分训练集，测试集
    """
    train_X,train_y,test_X,test_y=data_process.split_train_test(data_df)


    """
    处理成lstm可读的格式
    """

    lstm_train_X=data_process.process_lstm_data(train_X)
    lstm_test_X=data_process.process_lstm_data(test_X)
    test_y=data_process.scal_data_inverse(test_y)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    print(device)

    lstm_train_X_torch, lstm_train_y_torch, lstm_test_X_torch, lstm_test_y_torch = data_process.lstm_data_to_device(
        lstm_train_X, train_y, lstm_test_X, test_y, device)

    """
    train lstm model
    """
    lstm_model=lstm_network.lstm(input_size=3,hidden_size=16,output_size=3).to(device)
    # lstm_model.training_lstm(lstm_train_X_torch,lstm_train_y_torch,lstm_test_X_torch,lstm_test_y_torch)

    """
    load lstm model and predict
    """
    evaluation.load_model(my_parameter.LSTM_MODEL_PATH,lstm_model)
    lstm_pred = lstm_model.predict(lstm_test_X_torch)

    lstm_val_mse, lstm_val_r2 = evaluation.evaluate(lstm_test_y_torch, lstm_pred)
    print("-----------lstm---------\nmse:%s\tr2:%s\n--------------------------------\n" % (
        lstm_val_mse, lstm_val_r2))
    #
    # lstm_re=np.hstack([test_y, lstm_pred])
    # re=pd.DataFrame(lstm_re,columns=['traffic_flow','share','speed','lstm_traffic_flow_pred','lstm_share_pred','lstm_speed_pred'])
    # re.to_csv(my_parameter.LSTM_RE_PATH,index=False)

    """
    train GRU model
    """
    GRU_model=gru_network.gru(input_size=3,hidden_size=16,output_size=3).to(device)
    # GRU_model.training_GRU(lstm_train_X_torch,lstm_train_y_torch,lstm_test_X_torch,lstm_test_y_torch)

    """
        load GRU model and predict
    """
    evaluation.load_model(my_parameter.GRU_MODEL_PATH,GRU_model)
    GRU_pred = GRU_model.predict(lstm_test_X_torch)
    # gru_val_mse, gru_val_r2 = evaluation.evaluate(lstm_test_y_torch, GRU_pred)
    # print("-----------GRU---------\nmse:%s\tr2:%s\n--------------------------------\n" % (
    # gru_val_mse, gru_val_r2))
    #
    # GRU_re=np.hstack([test_y, GRU_pred])
    # re=pd.DataFrame(GRU_re,columns=['traffic_flow','share','speed','GRU_traffic_flow_pred','GRU_share_pred','GRU_speed_pred'])
    # re.to_csv(my_parameter.GRU_RE_PATH,index=False)
    """
      train GRU+attention model
    """
    GRU_attention_model=GRU_attention_network.gru_attention(device,attention_size=1,input_size=3,hidden_size=16,output_size=3).to(device)
    # GRU_attention_model.training_GRU_attention(lstm_train_X_torch,lstm_train_y_torch,lstm_test_X_torch,lstm_test_y_torch)

    """
        load GRU+attention model and predict
    """
    evaluation.load_model(my_parameter.GRU_ATTENTION_PATH,GRU_attention_model)
    GRU_attention_pred = GRU_attention_model.predict(lstm_test_X_torch)
    # alpha=GRU_attention_model.get_alpha()
    # gru_attention_val_mse, gru_attention_val_r2 =evaluation.evaluate(lstm_test_y_torch,GRU_attention_pred)
    # print("-----------GRU_attention---------\nmse:%s\tr2:%s\n--------------------------------\n" % (gru_attention_val_mse, gru_attention_val_r2))
    # GRU_attention_re=np.hstack([test_y, GRU_attention_pred])
    # re=pd.DataFrame(GRU_attention_re,columns=['traffic_flow','share','speed','GRU_attention_traffic_flow_pred','GRU_attention_share_pred','GRU_attention_speed_pred'])
    # re.to_csv(my_parameter.GRU_ATTENTION_RE_PATH,index=False)
   # print(alpha[0])

    """
          train GRU+dual_stage_attention model
    """

    # print(my_parameter.HISTORY_WINDOW)
    GRU_dual_stage_attention_model = GRU_dual_stage_attention_network.gru_dual_stage_attention(device,
                                                                                               input_size=3,
                                                                                                hidden_size=14,).to(device)
    # GRU_dual_stage_attention_model.training_gru_dual_stage_attention(lstm_train_X_torch,
    #                                                                  lstm_train_y_torch,
    #                                                                  lstm_test_X_torch,
    #                                                                  lstm_test_y_torch)

    """
         load GRU+dual_stage_attention model and predict
     """
    evaluation.load_model(my_parameter.GRU_DUAL_STAGE_ATTENTION_PATH,GRU_dual_stage_attention_model)
    GRU_dual_stage_attention_pred = GRU_dual_stage_attention_model.predict(lstm_test_X_torch)
    # alpha=GRU_dual_stage_attention_model.get_alpha()
    # GRU_dual_stage_attention_val_mse, GRU_dual_stage_attention_val_r2 =evaluation.evaluate(lstm_test_y_torch,GRU_dual_stage_attention_pred)
    # print("-----------GRU_dual_stage_attention---------\nmse:%s\tr2:%s\n--------------------------------\n" % (GRU_dual_stage_attention_val_mse, GRU_dual_stage_attention_val_r2))
    # GRU_dual_stage_attention_re=np.hstack([test_y, GRU_dual_stage_attention_pred])
    # re=pd.DataFrame(GRU_dual_stage_attention_re,columns=['traffic_flow','share','speed','GRU_dual_stage_attention_traffic_flow_pred','GRU_dual_stage_attention_share_pred','GRU_dual_stage_attention_speed_pred'])
    # re.to_csv(my_parameter.GRU_DUAL_STAGE_ATTENTION_RE_PATH,index=False)
   # print(alpha[0])



    """
        Train TCN+attention network
    """
    args = { 'drop_out': 0.2,'attention_size':16}
    #args=nni.get_next_parameter()

    #[52,52,52,72,72]
    TCN_attention_model=TCN_network.TemporalConvNet(num_inputs=3,
                                                    num_channels=[52,52,52,72,72],
                                                    kernel_size=3,
                                                    drop_out=args['drop_out'],
                                                    attention_size=args['attention_size']).to(device)
    # TCN_attention_model.training_TCN_model(lstm_train_X_torch,
    #     #                                      lstm_train_y_torch,
    #     #                                      lstm_test_X_torch,
    #     #                                      lstm_test_y_torch)
    """
        load TCN+attention network
    """
    evaluation.load_model(my_parameter.TCN_MODEL_PATH,TCN_attention_model)
    TCN_attention_pred = TCN_attention_model.predict(lstm_test_X_torch)
    print(TCN_attention_pred)
    # TCN_val_mse,TCN_val_r2 =evaluation.evaluate(lstm_test_y_torch,TCN_attention_pred)
    # print("-----------TCN_attention---------\nrmse:%s\tr2:%s\n--------------------------------\n" % (TCN_val_mse, TCN_val_r2))
    # TCN_attention_re=np.hstack([test_y, TCN_attention_pred])
    #re=pd.DataFrame(TCN_attention_re,columns=['traffic_flow','share','speed','TCN_attention_traffic_flow_pred','TCN_attention_share_pred','TCN_attention_speed_pred'])
    #re.to_csv(my_parameter.TCN_ATTENTION_RE_PATH,index=False)



    """
    各个模型对比图
    """
    # plt.rcParams['font.sans-serif'] = ['SimSun']
    # plt.rcParams['font.family'] = 'SimHei'
    # time_range=30
    # show_feature=0
    # time=np.arange(0,time_range).astype('str')
    # plt.plot(time,lstm_pred[:time_range,show_feature],label='lstm_pred')
    # plt.plot(time,test_y[:time_range,show_feature],label='ground_truth',color='black')
    # plt.plot(time,GRU_pred[:time_range,show_feature],label='gru_pred')
    # plt.plot(time,GRU_dual_stage_attention_pred[:time_range,show_feature],label='gru_dual_stage_attention')
    # plt.plot(time, GRU_attention_pred[:time_range, show_feature], label='gru_attention')
    # plt.legend()
    # plt.xlabel('time')
    # plt.ylabel('traffic flow')
    # plt.show()

    """
    损失函数
    """
    loss=GRU_dual_stage_attention_model.load_loss()
    loss_x=np.arange(0,len(loss))
    plt.plot(loss_x,loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()