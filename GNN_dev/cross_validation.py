from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
import sys
from tqdm import tqdm
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from GNN_architecture import GNN
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data, DataLoader
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow.pytorch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
device = torch.device("cuda")
def cross_val_score(dataset, X_train, y_train, trial, device ='cpu'):
    #Cross_validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,random_state=42)
    params = trial
    epochs = 300
    History ={"F1_record":[],"AP_record":[]}
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X_train,y_train))):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
        train_loader = DataLoader(dataset, batch_size = int(params["batch_size"]), sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=int(params["batch_size"]), sampler=valid_sampler)
        model = GNN(feature_size = 30, model_params=params).to(device)
        #reset model
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=params["sgd_momentum"],weight_decay=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
        for epoch in range(epochs):
            model.train()
            trainning_loss = 0
            all_labels = []
            probas = []
            all_probas = []
            all_preds = []
            for _, batch in enumerate(train_loader):
                batch.to(device)
                optimizer.zero_grad()
                output = model(batch.node_features.float(),batch.edge_attr.float(),
                               batch.edge_index,batch.batch,params).to(device)
        
                # print(output)
                loss = criterion(output[:,0], batch.y.float()) 
                loss.backward()
                optimizer.step()
                trainning_loss += loss.item()
                all_probas.append(torch.sigmoid(output).cpu().detach().numpy())
                all_preds.append(np.rint(torch.sigmoid(output).cpu().detach().numpy()))
                all_labels.append(batch.y.cpu().detach().numpy())
            # print(all_preds)    
            all_preds = np.concatenate(all_preds).ravel()
            all_labels = np.concatenate(all_labels).ravel()
            all_probas = np.concatenate(all_probas).ravel()
            loss = trainning_loss/len(train_loader)
            f1 = f1_score(all_labels,all_preds)
            ap = average_precision_score(all_labels,all_probas)

            mlflow.log_metric(key="f1-train_cross", value=float(f1), step=epoch)
            mlflow.log_metric(key="AP-train_cross", value=float(ap), step=epoch)
            mlflow.log_metric(key="Train loss_cross", value=float(loss), step=epoch)
            lr_scheduler.step()
            
        #Evaluate:
        model.eval()
        validation_loss = 0
        all_labels = []
        probas_val = []
        all_probas = []
        all_preds = []

        with torch.no_grad():
            for _, batch in enumerate(valid_loader):
                batch.to(device)
                output_val = model(batch.node_features.float(),batch.edge_attr.float(),
                                   batch.edge_index,batch.batch,params).to(device)
                loss = criterion(output_val[:,0], batch.y.float())
                validation_loss += loss.item()
                all_probas.append(torch.sigmoid(output_val).cpu().detach().numpy())
                all_preds.append(np.rint(torch.sigmoid(output_val).cpu().detach().numpy()))
                all_labels.append(batch.y.cpu().detach().numpy())

        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        all_probas = np.concatenate(all_probas).ravel()        
        loss_val = validation_loss/len(valid_loader)
        f1_val = f1_score(all_labels,all_preds)
        ap_val = average_precision_score(all_labels,all_probas)
        mlflow.log_metric(key="f1-val_cross", value=float(f1_val), step=epoch)
        mlflow.log_metric(key="AP-val_cross", value=float(ap_val), step=epoch)
        mlflow.log_metric(key="Val loss_cross", value=float(loss_val), step=epoch)
        print("Fold: {}.. ".format(fold_idx+1),
        "validation f1_score: {:.3f}.. ".format(f1_val),
              "validation average precision: {:.3f}.. ".format(ap_val),
         )
        History["F1_record"].append(f1_val)
        History["AP_record"].append(ap_val)
    #if scoring == "average_precision": 
    mean_scores_ap = sum(History["AP_record"]) / len(History["AP_record"])
    print(f"Overall AP score = {mean_scores_ap :.4f}")
    #if scoring == "f1":
    mean_scores_f1 = sum(History["F1_record"]) / len(History["F1_record"])
    print(f"Overall F1 score = {mean_scores_f1:.4f}")
    return History