import torch
import data_process
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np


def evaluate(label,pred):
    label=label.cpu().numpy()
    print(type(label),label.shape)
    print(type(pred),pred.shape)
    return np.sqrt(mean_squared_error(label, pred)),r2_score(label,pred)

def save_model(model,save_path):
    torch.save(model.state_dict(), save_path)

def load_model(save_path,model):
    model.load_state_dict(torch.load(save_path))

