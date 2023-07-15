from sklearn.metrics import f1_score, average_precision_score
import sys
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
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
from model_seed import *
seed_everything(42)


def suggest_hyperparameters(trial):
    params = {
    "batch_size": trial.suggest_categorical("batch_size",[128,64]),
    "sgd_momentum": trial.suggest_uniform("sgd_momentum",0.7, 0.9),
    "scheduler_gamma": trial.suggest_uniform("scheduler_gamma",0.9,1),
    "model_embedding_size": trial.suggest_int('model_embedding_size', 150,300),
    "model_layers": trial.suggest_categorical("model_layers",[3,4,5]),
    "model_dropout_rate": trial.suggest_uniform('model_dropout_rate', 0.1, 0.5),
    "model_top_k_ratio": trial.suggest_uniform('model_top_k_ratio', 0.4, 0.9),
    "model_dense_neurons": trial.suggest_int('model_dense_neurons', 150, 256),
    "mlp_dropout": trial.suggest_uniform("mlp_dropout",0.2,0.9)}
    return params





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(epochs, model, optimizer, criterion, name):
    #print("Saving...")
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, "./Model1104/" +name)

    
def objective(trial, train_dataset, valid_dataset,device ='cuda', epochs = 20):
    with mlflow.start_run():
        mlflow.set_tag("model","GNN")
        params = suggest_hyperparameters(trial)
        mlflow.log_params(trial.params)
        train_loader = DataLoader(train_dataset, batch_size=int(params["batch_size"]),shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=int(params["batch_size"]), shuffle=True)
        model = GNN(feature_size=30, model_params=params)
        model.to(device)
        seed_everything(42)
        mlflow.log_param("num_params", count_parameters(model))
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=0.01,
                                    momentum=params["sgd_momentum"],
                                    weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        #Start training:
        #print(len(study.trials)-1)
        #id = len(study.trials)-1
        name_model = "GNN"+str(id)+".pth"
        mlflow.log_param("trial_number", id)
        epochs = epochs
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
                # if np.isnan(output.detach().numpy()).any():
                #     mlflow.log_metric(key="f1-train", value=float(0.1), step=epoch)
                #     mlflow.log_metric(key="AP-train", value=float(0.1), step=epoch)
                #     mlflow.log_metric(key="Train loss", value=float(1e6), step=epoch)
                #     return {'loss': 1e6, 'status': STATUS_OK}
        
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

            mlflow.log_metric(key="f1-train", value=float(f1), step=epoch)
            mlflow.log_metric(key="AP-train", value=float(ap), step=epoch)
            mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
            
            #val1_loss,val1_f1, val1_ap,validation_loss = evaluate(model, criterion,valid_loader)
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
            mlflow.log_metric(key="f1-val", value=float(f1_val), step=epoch)
            mlflow.log_metric(key="AP-val", value=float(ap_val), step=epoch)
            mlflow.log_metric(key="Val loss", value=float(loss_val), step=epoch)
            trial.report(f1_val, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
            scheduler.step()
            #print(study.trial)
            #if len(study.trials)>1 and loss_val <study.best_trial.value:
        save_model(epochs, model, optimizer, criterion,name_model)
    return loss_val


