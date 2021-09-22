import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from tqdm import tqdm
import evaluation
import my_parameter
import data_process
"""
残差块
"""
class TemporalBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding,dilation,dropout):
        super(TemporalBlock, self).__init__()
        #in_channel:embedding的维度
        #out_channel:输出embedding的维度
        #kernel_size:time slop的个数
        self.conv_1=weight_norm(nn.Conv1d(in_channels=in_channel,
                  out_channels=out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation))
        self.relu_1=nn.ReLU()
        self.dropout_1=nn.Dropout(dropout)


        self.conv_2=weight_norm(nn.Conv1d(in_channels=out_channel,
                                          out_channels=out_channel,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation))
        self.relu_2=nn.ReLU()
        self.dropout_2=nn.Dropout(dropout)

        self.block=nn.Sequential(self.conv_1,self.relu_1,self.dropout_1,
                                 self.conv_2,self.relu_2,self.dropout_2)

        self.reshape_conv=nn.Conv1d(in_channels=in_channel,
                                    out_channels=out_channel,
                                     kernel_size=1) if in_channel!=out_channel else None

        self.relu_3=nn.ReLU()
        self.init_weights()

    def forward(self,x):
        temporal_re=self.block(x)
        reshape_x= x if self.reshape_conv == None else self.reshape_conv(x)
        re=self.relu_3(temporal_re+reshape_x)
        return re

    def init_weights(self):
        self.conv_1.weight.data.normal_(0, 0.01)
        self.conv_2.weight.data.normal_(0, 0.01)
        if self.reshape_conv is not None:
            self.reshape_conv.weight.data.normal_(0, 0.01)


class TemporalConvNet(nn.Module):
    def __init__(self,num_inputs,num_channels,kernel_size,drop_out,attention_size):
        super(TemporalConvNet, self).__init__()

        """
        存储每一个残差块网络结构，做堆叠
        """
        layers=[]
        length=len(num_channels)



        for i in range(length):
            dilation=2**i
            in_channel=num_inputs if i==0 else num_channels[i-1]
            out_channel=num_channels[i]
            re=TemporalBlock(in_channel=in_channel,
                              out_channel=out_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=int(dilation*(kernel_size-1)/2),
                              dilation=dilation,
                              dropout=drop_out)
            layers.append(re)

        self.network=nn.Sequential(*layers)

        """
        attention 模块
        """
        self.attention_layer=nn.Sequential(nn.Linear(in_features=num_channels[-1],out_features=attention_size),
                                      nn.Tanh(),
                                      nn.Linear(in_features=attention_size,out_features=1))

        self.fc=nn.Linear(in_features=num_channels[-1],out_features=3)
    def forward(self,x):
        #[batch_size,embedding_size,length]
        x=x.permute(0,2,1)
        re=self.network(x)
        embedding_size=re.size()[1]
        re_1=re.permute(0,2,1)
        #re=torch.reshape(re,[-1,embedding_size])
        alpha=self.attention_layer(re_1)
        out=torch.matmul(re,alpha)
        out=torch.reshape(out,[-1,embedding_size])
        out=self.fc(out)

        return out

    def _train(self, train_x, train_y):

        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

        out = self(train_x)

        out = torch.squeeze(out)
        loss = criterion(out, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def training_TCN_model(self, train_x, train_y, test_x, test_y):
        for e in tqdm(range(3000)):
            loss = self._train(train_x, train_y)
            if (e + 1) % 600 == 0:
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
                pred = self.predict(test_x)
                val_mse, val_r2 = evaluation.evaluate(test_y, pred)
                print("mse:%s\tr2:%s\n" % (val_mse, val_r2))


        evaluation.save_model(self, save_path=my_parameter.TCN_MODEL_PATH)


    def predict(self, test_x):
        self.eval()
        out = self(test_x).detach().cpu().numpy()
        out = data_process.scal_data_inverse(out)
        return out

    def get_alpha(self):
        return self.alpha



