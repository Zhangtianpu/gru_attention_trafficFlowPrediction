import torch
import torch.nn as nn
import  evaluation
from tqdm import tqdm
import data_process
import torch.nn.functional as F
import my_parameter
import pickle



class Encoder(nn.Module):
    def __init__(self,device,input_size,hidden_size,num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.T = my_parameter.HISTORY_WINDOW
        self.device=device

        self.gru = nn.GRU(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)
        self.attn1 = nn.Linear(in_features=self.hidden_size, out_features=self.input_size)
        self.attn2 = nn.Linear(in_features=self.input_size, out_features=self.input_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.input_size, out_features=self.input_size)

    def forward(self,x):
        batch_size=x.size()[0]

        final_output=torch.empty(batch_size,self.T,self.hidden_size).to(self.device)


        #初试化c0,h0 (num_layers * num_directions, batch, hidden_size)
        # c_empty = torch.empty(self.num_layers, batch_size,self.hidden_size).to(device)
        # c_0 = torch.nn.init.xavier_normal_(c_empty).to(device)

        h_empty=torch.empty(self.num_layers,batch_size,self.hidden_size).to(self.device)
        h_0=torch.nn.init.xavier_normal_(h_empty).to(self.device)
        for step in range(self.T):
            #[batch_size,hidden_size]
            c_h_merge=h_0[-1,:,:]

            #[batch_size,self.input_size]
            w_1=self.attn1(c_h_merge)

            #[batch_size,self.input_size]

            w_2=self.attn2(x[:,step,:])

            #[batch_szie,self.input_size]
            v=self.attn3(self.tanh(w_1+w_2))

            #[batch_size,self.input_size]
            alpha=F.softmax(v,dim=1)

            #[batch_size,self.input_size]
            weighted_x=torch.mul(x[:,step,:],alpha)
            weighted_x=weighted_x.reshape([-1,1,self.input_size])

            output,h_next=self.gru(weighted_x,h_0)

            final_output[:,step,:]=output[:,-1,:]
            h_0=h_next


        return final_output



class Decoder(nn.Module):
    def __init__(self,device,input_size,hidden_size,output_size=1,num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.num_layers=num_layers
        self.T_1=my_parameter.HISTORY_WINDOW
        self.output_size=output_size
        self.device=device

        self.layer1=nn.Linear(self.hidden_size,self.hidden_size)
        self.layer2=nn.Linear(self.input_size,self.hidden_size)
        self.tanh=nn.Tanh()
        self.layer3 = nn.Linear(self.hidden_size, 1)
        self.layer4=nn.Linear(self.input_size+3,self.hidden_size)
        self.gru=nn.GRU(self.hidden_size,self.hidden_size,num_layers=1,batch_first=True)
        self.fc=nn.Linear(self.hidden_size,self.output_size)

    def forward(self,decoder_x,x):
        batch_size=decoder_x.size()[0]

        final_output=torch.empty(batch_size,self.T_1,self.hidden_size).to(self.device)


        # 初试化c0,h0 (num_layers * num_directions, batch, hidden_size)
        # c_empty = torch.empty(self.num_layers, batch_size, self.hidden_size).to(device)
        # c_0 = torch.nn.init.xavier_normal_(c_empty).to(device)

        h_empty = torch.empty(self.num_layers, batch_size, self.hidden_size).to(self.device)
        h_0 = torch.nn.init.xavier_normal_(h_empty).to(self.device)

        for step in range(self.T_1):
            #[batch,hidden_size]
            c_h_merge=h_0[-1,:,:]


            #[batch_size,hidden_size]
            w_1=self.layer1(c_h_merge)
            #[batch_size,1,hidden_size]
            w_1_reshape=w_1.reshape([batch_size,1,-1])
            #[batch_size*self.T,input_size]
            decoder_x_reshape=decoder_x.reshape([batch_size*self.T_1,-1])
            #[batch_size*self.T,hidden_size]
            w_2=self.layer2(decoder_x_reshape)
            #[batch_size,self.T,hidden_size]
            w_2_reshape=w_2.reshape([batch_size,-1,self.hidden_size])



            #[batch_size,self.T,1]
            alpha=self.layer3(self.tanh(w_1_reshape+w_2_reshape))
            alpha_reshape=alpha.reshape([batch_size,-1])
            #[batch_size,self.T]
            weighted_alpha=F.softmax(alpha_reshape,dim=1)

            #[batch_size,self.T,1]
            weighted_alpha=weighted_alpha.reshape([batch_size,self.T_1,1])

            #[batch_size,input_size,self.T]
            decoder_x_per = decoder_x.permute(0, 2, 1)
            #[batch_size,input_size]
            weighted_c=torch.matmul(decoder_x_per, weighted_alpha).reshape([batch_size,self.input_size])

            #final_alpha[:, step, :] = alpha

            #[batch_size,input_size+raw_input_size]
            y_0=torch.cat([weighted_c,x[:,step,:]],dim=1)
            y_0=self.layer4(y_0)
            y_0=y_0.reshape([-1,1,self.hidden_size])

            out,h_next=self.gru(y_0,h_0)
            h_0=h_next

            final_output[:,step,:]=out.squeeze(1)

        re=self.fc(final_output[:, -1, :])
        return re,weighted_alpha


class gru_dual_stage_attention(nn.Module):
    def __init__(self,device,input_size=1,hidden_size=16):
        super(gru_dual_stage_attention,self).__init__()

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        # LSTM encoder层
        self.layer1 = Encoder(device=device,
                              input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=1).to(device)

        # decoder
        self.layer2 = Decoder(device=device,
                              input_size=self.hidden_size,
                              hidden_size=self.hidden_size * 2,
                              num_layers=1,
                              output_size=3).to(device)


    def forward(self,x):
        #get encoder result
        output = self.layer1(x)

        #get decoder result
        output,alpha=self.layer2(output,x)
        self.alpha=alpha
        return output


    def _train(self,train_x, train_y):

        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

        out = self(train_x)

        out=torch.squeeze(out)
        loss = criterion(out, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def training_gru_dual_stage_attention(self,train_x, train_y,test_x,test_y):
        total_loss=[]
        for e in tqdm(range(3000)):
            loss = self._train(train_x, train_y)
            total_loss.append(loss)
            if (e + 1) % 600 == 0:
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
                pred=self.predict(test_x)
                val_mse, val_r2 = evaluation.evaluate(test_y,pred)
                print("mse:%s\tr2:%s\n" % (val_mse, val_r2))

        evaluation.save_model(self,save_path=my_parameter.GRU_DUAL_STAGE_ATTENTION_PATH)
        self._save_loss(total_loss)

    def predict(self,test_x):
        self.eval()
        out=self(test_x).detach().cpu().numpy()
        out=data_process.scal_data_inverse(out)
        return out

    def get_alpha(self):
        return self.alpha

    def _save_loss(self,total_loss):
        with open(my_parameter.LOSS_PATH,'wb') as f:
            pickle.dump(total_loss,f)

    def load_loss(self):
        with open(my_parameter.LOSS_PATH,'rb') as f:
            total_loss=pickle.load(f)
        return total_loss