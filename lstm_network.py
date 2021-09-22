import torch
import torch.nn as nn
import  evaluation
from tqdm import tqdm
import data_process
import torch.nn.functional as F
import my_parameter

class lstm(nn.Module):
    def __init__(self,input_size=1,hidden_size=16,output_size=1,num_layer=2):
        super(lstm,self).__init__()
        #LSTM层
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True)
        #全连接层
        self.layer2=nn.Linear(hidden_size,output_size)


    def forward(self,x):
        #得到lstm层输出结果
        x,_ = self.layer1(x)
        x=x[:,-1,:]
        #得到全连接层结果
        x=self.layer2(x)
        return x


    def _train(self,train_x, train_y):

        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        out = self(train_x)

        out=torch.squeeze(out)
        loss = criterion(out, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def training_lstm(self,train_x, train_y,test_x,test_y):
        #model = lstm(input_size=1,hidden_size=32).to(device)
        for e in tqdm(range(3000)):
            loss = self._train(train_x, train_y)
            if (e + 1) % 600 == 0:
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
                pred=self.predict(test_x)
                val_mse, val_r2 = evaluation.evaluate(test_y,pred)
                print("mse:%s\tr2:%s\n" % (val_mse, val_r2))

        evaluation.save_model(self,save_path=my_parameter.LSTM_MODEL_PATH)

    def predict(self,test_x):
        self.eval()
        out=self(test_x).detach().cpu().numpy()
        out=data_process.scal_data_inverse(out)
        return out
