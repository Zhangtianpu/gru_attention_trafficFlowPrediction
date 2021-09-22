import numpy as np
import pandas as pd
import my_parameter
from sklearn.preprocessing import StandardScaler
import pickle
import torch

def scal_data(data,is_train=True):
    if is_train:
        scal=StandardScaler()
        x=scal.fit_transform(data)

        with open(my_parameter.SCAL_PATH,'wb') as f:
            pickle.dump(scal,f)
    else:
        with open(my_parameter.SCAL_PATH,'r') as f:
            scal=pickle.load(f)
            x=scal.transform(data)
    return x

def get_history_data(data_df,columns):

    data_df = data_df.sort_index(ascending=False).reset_index(drop=True)

    for index in range(my_parameter.HISTORY_WINDOW):
        history_traffic_flow = data_df[columns][index + 1:]
        history_data=history_traffic_flow.reset_index()[columns]
        columns_dict={}
        for column in columns:
            columns_dict[column]='before_%s_%s'%(column,index+1)
        history_data.rename(columns=columns_dict,inplace=True)
        data_df=pd.concat([data_df,history_data],axis=1)

    data_df.dropna(inplace=True)
    data_df.sort_index(ascending=False,inplace=True)
    data_df.reset_index(drop=True,inplace=True)
    data_df.to_csv(my_parameter.PROCESS_PATH, index=False)

    return data_df

def split_train_test(data,train_pro=0.7):
    total_num=data.shape[0]
    train_num=int(total_num*train_pro)
    train_X=data.iloc[:train_num,3:]
    train_y=data.iloc[:train_num,0:3]
    test_X=data.iloc[train_num:,3:]
    test_y=data.iloc[train_num:,0:3]
    return train_X,train_y,test_X,test_y


def process_lstm_data(data_df):
    lstm_data=[]
    row_num = data_df.shape[0]
    for batch in range(row_num):
        data = np.array(data_df.iloc[batch]).reshape(-1, 3)
        lstm_data.append(data)
    return np.array(lstm_data)

def scal_data_inverse(data):
    with open(my_parameter.SCAL_PATH,'rb') as f:
        scal=pickle.load(f)
        inverse_data=scal.inverse_transform(data)
    return inverse_data


def lstm_data_to_device(train_X,train_y,test_X,test_y,device):
    train_X_torch = torch.tensor(np.array(train_X)).float().to(device)
    train_y_torch = torch.tensor(np.array(train_y)).float().to(device)
    test_X_torch = torch.tensor(np.array(test_X)).float().to(device)
    test_y_torch = torch.tensor(np.array(test_y)).float().to(device)
    return train_X_torch,train_y_torch,test_X_torch,test_y_torch

if __name__ == '__main__':
    data=np.load(my_parameter.RAW_PATH)['data'][:,0,:]
    data=scal_data(data)
    data_df=pd.DataFrame(data,columns=['traffic_flow','share','speed'])
    data_df=get_history_data(data_df,columns=['traffic_flow','share','speed'])
    print(data_df.head())


