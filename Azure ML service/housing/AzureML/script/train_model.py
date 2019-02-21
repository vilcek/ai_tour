import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
#import pickle

# Azure ML specific
from azureml.core.run import Run
run = Run.get_context()

# data folder in Azure ML service is passed through an argument
parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point")
parser.add_argument("--num_hidden_layers", type=int, dest="num_hidden_layers", help="number of hidden leayers")
parser.add_argument("--hidden_layer_size", type=int, dest="hidden_layer_size", help="hidden layer size")
parser.add_argument("--dropout_rate", type=float, dest="dropout_rate", help="dropout rate")
parser.add_argument("--learning_rate", type=float, dest="learning_rate", help="learning rate")
args = parser.parse_args()
data_folder = os.path.join(args.data_folder, "pytorch-dl-regression")
print('Data folder:', data_folder)

# read data
df_housing = pd.read_csv(os.path.join(data_folder, "train.csv"))

# prepare data
df_housing.drop("Id", axis = 1, inplace = True)
fill_none = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtExposure", "BsmtCond", "BsmtQual", "MasVnrType"]
for var in fill_none:
    df_housing[var] = df_housing[var].fillna("None")   
fill_zero = ["GarageYrBlt", "BsmtFinType2", "BsmtFinType1", "MasVnrArea"]
for var in fill_zero:
    df_housing[var] = df_housing[var].fillna(0)
df_housing["LotFrontage"] = df_housing.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df_housing['Electrical'] = df_housing['Electrical'].fillna(df_housing['Electrical'].mode()[0])
numeric_vars = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
               "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
               "ScreenPorch", "PoolArea", "MiscVal", "SalePrice"]
categorical_vars = [v for v in df_housing.columns if v not in numeric_vars]
df_housing[numeric_vars] = df_housing[numeric_vars].astype(float)
df_housing[categorical_vars] = df_housing[categorical_vars].astype(str)
response = ["SalePrice"]
df_housing_numeric = df_housing[[var for var in numeric_vars if var not in response]]
df_housing_categotical = df_housing[categorical_vars]
df_housing_response = df_housing[response]
label_encoders = {}
for var in categorical_vars:
    label_encoders[var] = LabelEncoder()
    df_housing_categotical[var] = label_encoders[var].fit_transform(df_housing_categotical[var])

idx = list(range(df_housing_response.shape[0]))
np.random.seed(123)
np.random.shuffle(idx)
train_idx = idx[0:round(len(idx)*0.8)]
test_idx = idx[round(len(idx)*0.8):round(len(idx)*0.9)]
val_idx = idx[round(len(idx)*0.9):]
df_housing_num_train = df_housing_numeric.iloc[train_idx]
df_housing_cat_train = df_housing_categotical.iloc[train_idx]
df_housing_resp_train = df_housing_response.iloc[train_idx]
df_housing_num_test = df_housing_numeric.iloc[test_idx]
df_housing_cat_test = df_housing_categotical.iloc[test_idx]
df_housing_resp_test = df_housing_response.iloc[test_idx]
df_housing_num_val = df_housing_numeric.iloc[val_idx]
df_housing_cat_val = df_housing_categotical.iloc[val_idx]
df_housing_resp_val = df_housing_response.iloc[val_idx]

# create model
numeric_vars = [var for var in numeric_vars if var not in response]

cat_dims = [int(df_housing_categotical[var].nunique()) for var in categorical_vars]
emb_dims = [min(50, (x + 1) // 2) for x in cat_dims]
hidden_layers = [args.hidden_layer_size] * args.num_hidden_layers
drop_rates = [args.dropout_rate] * 2
learning_rate = args.learning_rate
lr_steps = [250]
lr_gamma = 0.1
mb_size = 32
num_iter = 250

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.embeddings = nn.ModuleList()
        for i in range(len(categorical_vars)):
            self.embeddings.append(nn.Embedding(cat_dims[i], emb_dims[i]))
        
        self.cat_dropout = nn.Dropout(drop_rates[0])
        
        self.num_batchnorm = nn.BatchNorm1d(len(numeric_vars))
        
        self.dense_block = nn.ModuleList()
        for l in range(len(hidden_layers)):
            if l == 0:
                self.dense_block.append(nn.Linear(sum(emb_dims)+len(numeric_vars), hidden_layers[l]))
                self.dense_block.append(nn.ReLU6())
                self.dense_block.append(nn.BatchNorm1d(hidden_layers[l]))
                self.dense_block.append(nn.Dropout(drop_rates[1]))
            if l > 0:
                self.dense_block.append(nn.Linear(hidden_layers[l-1], hidden_layers[l]))
                self.dense_block.append(nn.ReLU6())
                self.dense_block.append(nn.BatchNorm1d(hidden_layers[l]))
                self.dense_block.append(nn.Dropout(drop_rates[1]))
        
        self.dense = nn.Sequential(*self.dense_block)
        
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, Xc_list, Xn):
        Xc_list = [self.embeddings[i](Xc_list[i]) for i in range(len(Xc_list))]
        Xc = torch.cat(Xc_list, dim=1)
        Xc = self.cat_dropout(Xc)
        Xn = self.num_batchnorm(Xn)
        X = torch.cat([Xc, Xn], dim=1)
        X = self.dense(X)
        X = self.output(X)
        return X
      
class MAE_Loss(nn.Module):
    def __init__(self):
        super(MAE_Loss, self).__init__()
    
    def forward(self, X, Y):
        error = torch.abs(Y - X)
        loss = torch.mean(error)
        return loss
      
gpu_available = torch.cuda.is_available()

model = Model().cuda() if gpu_available else Model()

model_loss = MAE_Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

# train model
def get_batch(data_num, data_cat, data_resp, mb_size, scale_num):
    array_num = data_num.values.astype(np.float32)
    if scale_num:
        scaler = Normalizer()
        scaler.fit(array_num)
        array_num = scaler.transform(array_num)
    array_cat = data_cat.values.astype(np.long)
    array_resp = data_resp.values.astype(np.float32)
    shuffle = np.random.permutation(len(data_resp))
    start = 0
    array_num = array_num[shuffle]
    array_cat = array_cat[shuffle]
    array_resp = array_resp[shuffle]
    while start + mb_size <= len(data_resp):
        yield array_num[start:start+mb_size], array_cat[start:start+mb_size], array_resp[start:start+mb_size]
        start += mb_size

num_mb = int(len(df_housing_num_train)/mb_size)

for n in range(num_iter):
    train_loss = 0
    val_loss = 0
    
    for k, (mb_Xn, mb_Xc, mb_Y) in enumerate(get_batch(df_housing_num_train, df_housing_cat_train, df_housing_resp_train, mb_size, True)):
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # convert data from numpy arrays to torch tensors and put them on GPU, if we have one available
        mb_Xn = torch.from_numpy(mb_Xn).cuda() if gpu_available else torch.from_numpy(mb_Xn)
        mb_Xc = [torch.from_numpy(mb_Xc[:,i]).long().cuda() if gpu_available else torch.from_numpy(mb_Xc[:,i]).long() for i in range(len(categorical_vars))]
        mb_Y = torch.from_numpy(mb_Y).cuda() if gpu_available else torch.from_numpy(mb_Y)
        
        # run forward + backward + optimize on a mini-batch
        outputs = model(mb_Xc, mb_Xn)
        loss = model_loss(outputs, mb_Y)
        loss.backward()
        optimizer.step()
        
        # update cost for each iteration
        train_loss += loss.data / num_mb
    
    # print and save the cost for each itaration
    run.log("train loss", int(train_loss.cpu().numpy()))
    
    # print actual and predicted values
    with torch.no_grad():
        model.eval()
        
        for k, (mb_Xn, mb_Xc, mb_Y) in enumerate(get_batch(df_housing_num_val, df_housing_cat_val, df_housing_resp_val, mb_size, True)):
            mb_Xn = torch.from_numpy(mb_Xn).cuda() if gpu_available else torch.from_numpy(mb_Xn)
            mb_Xc = [torch.from_numpy(mb_Xc[:,i]).long().cuda() if gpu_available else torch.from_numpy(mb_Xc[:,i]).long() for i in range(len(categorical_vars))]
            mb_Y = torch.from_numpy(mb_Y).cuda() if gpu_available else torch.from_numpy(mb_Y)
            outputs = model(mb_Xc, mb_Xn)
            loss = model_loss(outputs, mb_Y)
            val_loss += loss.data / num_mb
        run.log("validation loss", int(val_loss.cpu().numpy()))
        
        model.train()
        
    scheduler.step()
    
torch.save(model.state_dict(), './outputs/housing_model_state_dict.pt')

# evaluate model
model.load_state_dict(torch.load('./outputs/housing_model_state_dict.pt'))

def MAE(data_num, data_cat, data_resp):
    with torch.no_grad():
        errors = []
        for k, (mb_Xn, mb_Xc, mb_Y) in enumerate(get_batch(data_num, data_cat, data_resp, mb_size, True)):
            Xn = torch.from_numpy(mb_Xn).cuda() if gpu_available else torch.from_numpy(mb_Xn)
            Xc = [torch.from_numpy(mb_Xc[:,i]).long().cuda() if gpu_available else torch.from_numpy(mb_Xc[:,i]).long() for i in range(len(categorical_vars))]
            actual = torch.from_numpy(mb_Y).cuda() if gpu_available else torch.from_numpy(mb_Y)
            predicted = model(Xc, Xn)
            actual = actual.cpu().numpy().flatten()
            predicted = predicted.cpu().numpy().flatten()
            errors.append(np.abs(actual - predicted))
        return np.mean(errors)

run.log("MAE (Train)", MAE(df_housing_num_train, df_housing_cat_train, df_housing_resp_train))
run.log("MAE (Validation)", MAE(df_housing_num_val, df_housing_cat_val, df_housing_resp_val))
run.log("MAE (Test)", MAE(df_housing_num_test, df_housing_cat_test, df_housing_resp_test))