import torch
import torch.nn as nn
import  evaluation
from tqdm import tqdm
import data_process
import torch.nn.functional as F
import my_parameter

class gru_attention(nn.Module):
    def __init__(self,device,attention_size,input_size=1,hidden_size=16,output_size=1,num_layer=2):
        super(gru_attention,self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.device = device
        #LSTM层
        self.layer1 = nn.GRU(input_size,hidden_size,num_layer,batch_first=True)
        #全连接层
        self.layer2=nn.Linear(hidden_size,output_size)


    def forward(self,x):
        #得到GRU层输出结果
        x,_ = self.layer1(x)
        x,alpha=self._attention(x)
        #得到全连接层结果
        x=self.layer2(x)
        self.alpha=alpha
        return x

    def _attention(self, lstm_output):
        self.batch_size = lstm_output.size()[0]
        self.seq_len = lstm_output.size()[1]
        w_empty = torch.empty(self.hidden_size, self.attention_size).to(self.device)
        W = torch.nn.init.xavier_normal_(w_empty).to(self.device)

        lstm_output_1 = torch.tanh(lstm_output)
        lstm_output_1 = torch.reshape(lstm_output_1, [-1, self.hidden_size])
        """
        linear
            self.attention_layer=nn.Sequential(nn.Linear(hidden_size,attention_size),
                                               nn.Tanh(),
                                               nn.Linear(attention_size,1)
                                               torch.softmax())
        """
        # [batch_size*seq_len,attention_size]
        M = torch.matmul(lstm_output_1, W)
        u_empty = torch.empty(self.attention_size, 1).to(self.device)
        U = torch.nn.init.xavier_normal_(u_empty).to(self.device)

        # [batch_size*seq_len,1]
        exp = torch.matmul(M, U)
        exp = torch.reshape(exp, [-1, self.seq_len])

        # [batch_size*seq_len,1]
        alpha = torch.softmax(exp, dim=1)
        # [batch_size,seq_len,1]
        alpha = torch.reshape(alpha, [-1, self.seq_len, 1])

        # [batch_size,hidden_size,seq_len]
        lstm_output_per = lstm_output.permute(0, 2, 1)
        # [batch_size,hidden_size,seq_len]*[batch_size,seq_len,1]=[batch_size,hidden_size,1]
        out = torch.matmul(lstm_output_per, alpha)

        # [batch_size,hidden_size]
        out = torch.tanh(out.reshape([-1, self.hidden_size]))

        return out, alpha

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

    def training_GRU_attention(self,train_x, train_y,test_x,test_y):
        for e in tqdm(range(3000)):
            loss = self._train(train_x, train_y)
            if (e + 1) % 600 == 0:
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
                pred=self.predict(test_x)
                val_mse, val_r2 = evaluation.evaluate(test_y,pred)
                print("mse:%s\tr2:%s\n" % (val_mse, val_r2))

        evaluation.save_model(self,save_path=my_parameter.GRU_ATTENTION_PATH)

    def predict(self,test_x):
        self.eval()
        out=self(test_x).detach().cpu().numpy()
        out=data_process.scal_data_inverse(out)
        return out

    def get_alpha(self):
        return self.alpha
