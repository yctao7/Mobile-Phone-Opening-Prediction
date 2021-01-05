import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from new_model import GruPredictor

torch.manual_seed(0)
NUM_EPOCHS = 600
BATCH_SIZE = 128
HIDDEN_SIZE = 64
LR = 5e-2

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def dataloader(x_dict, length=10):
    x, y = [], []
    for i in range(len(x_dict)):
        dic = x_dict[i]
        y_all = dic['time before bright'].values.tolist()
        dic1 = dic.drop(columns='time before bright')
        x_all = dic1.values.tolist()
        for j in range(len(dic) - length):
            x.append(x_all[j:(j+length)])
            y.append(y_all[j+length-1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)
    x_test, y_test = torch.tensor(x_test), torch.tensor(y_test)
    
    contin_range = [0, 1] + list(range(7, 25))
    dow_idx = 25
    h_idx = 26
    m_idx = 27
    s_idx = 28
    
    # normalize x_train
    s1 = x_train.shape
    x_train = x_train.reshape(-1, x_train.shape[-1])
    ma, _ = x_train[:, contin_range].max(dim=0)
    mi, _ = x_train[:, contin_range].min(dim=0)
    x_train[:, contin_range] = (x_train[:, contin_range] - mi) / (ma - mi + 1e-8)
    x_train[:, dow_idx] /= 7 # day of week
    x_train[:, h_idx] /= 24 # hour
    x_train[:, m_idx] /= 60 # minute
    x_train[:, s_idx] /= 60 # second
    x_train = x_train.reshape(s1)
    
    # normalize x_test
    s2 = x_test.shape
    x_test = x_test.reshape(-1, x_test.shape[-1])
    x_test[:, contin_range] = (x_test[:, contin_range] - mi) / (ma - mi + 1e-8)
    x_test[:, dow_idx] /= 7 # day of week
    x_test[:, h_idx] /= 24 # hour
    x_test[:, m_idx] /= 60 # minute
    x_test[:, s_idx] /= 60 # second
    x_test = x_test.reshape(s2)
    
    # normalize y
    y_train //= 30
    y_test //= 30
    
    train_set = MyDataset(x_train, y_train)
    test_set = MyDataset(x_test, y_test)
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=test_set.__len__())
    return loader_train, loader_test

x_dict = np.load('x_dict_10d_NOH000041.npy', allow_pickle=True).item()
loader_train, loader_test = dataloader(x_dict)

# import xgboost as xgb
# xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xg_reg.fit(x_train.reshape(-1, x_train.shape[-1]).numpy(), y_train.reshape(-1).numpy())
# preds = xg_reg.predict(x_test.reshape(-1, x_test.shape[-1]).numpy())

input_size = len(x_dict[0].columns) - 1
model = GruPredictor(input_size, HIDDEN_SIZE)
optimizer = Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, 200, gamma=0.4)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 150], gamma=0.1)
criterion = nn.MSELoss()

for i in range(NUM_EPOCHS):
    for t, (x, y) in enumerate(loader_train):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        print('[%d, %d] loss: %.3f' % (i + 1, t + 1, loss))
    scheduler.step()   
    
    model.eval()
    with torch.no_grad():
        for (x_, y_) in loader_test:
            loss_ = criterion(model(x_), y_)
            print('[%d] loss: %.3f' % (i + 1, loss_))
    model.train()