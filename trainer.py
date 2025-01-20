from utils import *
import torch
import random
from model import EADTN
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
import os
from torch import nn
import torch.nn.functional as F
import itertools
class Trainer():
    def __init__(self,args) -> None:
        self.args=args
        self.set_seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_seed(self):
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
    

    def read_random_data(self):
        dir_input = f'./data/{self.args.dataset}/random/{self.args.dataset}.txt'
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")
        np.random.seed(self.args.seed)
        np.random.shuffle(data_list)
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        self.train_data = data_list[0:split_pos]
        self.test_data = data_list[split_pos:-1]
        
        
    def read_cluster_data(self):
        self.cluster_list=['012','123','234','340','401']
        self.cluster_data=[]
        for i in self.cluster_list:
            data=pd.read_csv(f'./data/{self.args.dataset}/Clustering/{self.args.Clustering_basis}/cluster{i}.csv').values.tolist()
            self.cluster_data.append(data)
        self.train_data=pd.read_csv(f'./data/{self.args.dataset}/Clustering/{self.args.Clustering_basis}/train_cluster.csv').values.tolist()
        self.test_data=pd.read_csv(f'./data/{self.args.dataset}/Clustering/{self.args.Clustering_basis}/test_cluster.csv').values.tolist()

    def init_optimizer(self,model,train_size):
        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': self.args.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.args.learning_rate, max_lr=self.args.learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // self.args.batch_size)
        loss_fun = PolyLoss(weight_loss=None,
                            DEVICE=self.device, epsilon=self.args.loss_epsilon)
        return optimizer,scheduler,loss_fun
    
    def init_cluster_optimizer(self,model,train_size):

        """create optimizer and scheduler"""
        optimizer = optim.Adam(params=model.parameters(),weight_decay=self.args.weight_decay, lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.args.learning_rate, max_lr=self.args.learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // self.args.batch_size)
        loss_fun = PolyLoss(weight_loss=None,
                            DEVICE=self.device, epsilon=self.args.loss_epsilon)
        return optimizer,scheduler,loss_fun
    
    
    
    def get_cluster_dataloader(self,fold):
        train_list=self.train_data
        random.shuffle(train_list)
        
        val_list=train_list[:int(len(train_list)*0.1)]
        train_list=train_list[int(len(train_list)*0.1):]
        
        fine_tuning_dataset=DTIDataSet(self.cluster_data[fold])
        train_dataset=DTIDataSet(train_list)
        val_dataset=DTIDataSet(val_list)
        
        train_dataloader=DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=cluster_collate_fn_normal_train, drop_last=True)
        val_dataloader=DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=cluster_collate_fn_normal_train, drop_last=True)
        fine_tuning_dataloader=DataLoader(fine_tuning_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=cluster_collate_fn, drop_last=True)
        
        return (train_dataloader,val_dataloader,fine_tuning_dataloader)
    
    def get_k_fold_dataloader(self,fold):
        
        train_list=self.train_data
        train_list, val_list = get_kfold_data(
            fold, train_list, k=self.args.k_fold)
        train_dataset=DTIDataSet(train_list)
        val_dataset=DTIDataSet(val_list)

        train_dataloader=DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn, drop_last=True)
        val_dataloader=DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn, drop_last=True)
        
        return (train_dataloader,val_dataloader)
    
    def get_combined_val_dataloader(self):
        val_datasets = []
        for fold in range(5):
            _, val_dataloader, _ = self.get_cluster_dataloader(fold)
            val_datasets.append(val_dataloader.dataset)

        combined_val_dataset = DTIDataSet(val_datasets)
        combined_val_dataloader = DataLoader(combined_val_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0, collate_fn=cluster_collate_fn_normal_train, drop_last=True)

        return combined_val_dataloader
    
    def fine_tuning(self,train_dataloader,fold):
        print('*'*10+'fine_tuning'+'*'*10)
        model=EADTN(self.args).to(self.device)
        model.load_state_dict(torch.load(f'./check_points/{fold}/valid_best_checkpoint.pth'))
        model.train()
        optimizer,scheduler,loss_fun=self.init_cluster_optimizer(model,len(train_dataloader))
        
        for epoch in range(self.args.fine_tuning_epochs):
            for drugs,proteins,labels,_ in tqdm(train_dataloader):
                drugs,proteins,labels=drugs.to(self.device),proteins.to(self.device),labels.to(self.device)
                optimizer.zero_grad()
                outputs=model(drugs,proteins)
                loss_value=loss_fun(outputs,labels)
                loss_value.backward()
                optimizer.step()
                scheduler.step()
        torch.save(model.state_dict(),f'./check_points/fine_tuning/{self.cluster_list[fold]}/valid_best_checkpoint.pth')
        
    def train(self,loader_tuple,fold):
        print('*'*10+'training'+'*'*10)
        early_stopping = EarlyStopping(
            savepath=f'./check_points/{fold}', patience=self.args.patience, verbose=True, delta=0)
        train_dataloader,val_dataloader=loader_tuple
        model=EADTN(self.args).to(self.device)
        optimizer,scheduler,loss_fun=self.init_optimizer(model,len(train_dataloader))
        for epoch in range(1, self.args.epoch + 1):
            if early_stopping.early_stop == True:
                break
            #train
            model.train()
            train_loss_list=[]
            for drugs,proteins,labels in tqdm(train_dataloader):
                drugs,proteins,labels=drugs.to(self.device),proteins.to(self.device),labels.to(self.device)
                optimizer.zero_grad()
                outputs=model(drugs,proteins)
                loss_value=loss_fun(outputs,labels)
                loss_value.backward()
                optimizer.step()
                scheduler.step()
                train_loss_list.append(loss_value.cpu().detach())
            avg_train_loss=np.average(train_loss_list)
            #eval
            model.eval()
            val_loss_list=[]
            Y, P, S = [], [], []
            with torch.no_grad():
                for drugs,proteins,labels in tqdm(val_dataloader):
                    drugs,proteins,labels=drugs.to(self.device),proteins.to(self.device),labels.to(self.device)
                    outputs=model(drugs,proteins)
                    loss_value=loss_fun(outputs,labels)
                    val_loss_list.append(loss_value.cpu().detach())
                    labels = labels.to('cpu').data.numpy()
                    outputs = F.softmax(
                        outputs, 1).to('cpu').data.numpy()
                    predictions = np.argmax(outputs, axis=1)
                    outputs = outputs[:, 1]
                    Y.extend(labels)
                    P.extend(predictions)
                    S.extend(outputs)
            avg_val_loss=(np.average(val_loss_list)/self.args.batch_size)
            result=compute_metrics(Y, S, P)
            print(f'avg_loss:{avg_val_loss}')
            for key,value in result.items():
                print(f'{key}:{value}')
            AUC_value=result['AUC']
            early_stopping(AUC_value, model)
            

            
            
            
    def test_random(self):
        print('*'*10+'testing'+'*'*10)
        output_list=[]
        label_list=[]
        Y, P, S = [], [], []
        test_dataset=DTIDataSet(self.test_data)
        test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=True,collate_fn=collate_fn)
        model_list=[]
        for i in [f'./check_points/{j}/valid_best_checkpoint.pth'for j in range(self.args.k_fold)]:
            model=EADTN(args=self.args)
            model.load_state_dict(torch.load(i))
            model.to(self.device)
            model.eval()
            model_list.append(model)

        with torch.no_grad():
            for drugs,proteins,labels in tqdm(test_dataloader):
                drugs,proteins,labels=drugs.to(self.device),proteins.to(self.device),labels.to(self.device)
                label_list.append(int(labels))
                out_sum=0
                for index,model in enumerate(model_list):
                    out=model(drugs,proteins)
                    out_sum+=out
                out_sum/=len(model_list)
                output_list.append(out_sum.cpu().detach())
        labels = np.array(label_list)
        outputs = torch.nn.functional.softmax(torch.cat(output_list,dim=0), 1).to('cpu').data.numpy()
        predictions = np.argmax(outputs, axis=1)
        outputs = outputs[:, 1]
        Y.extend(labels)
        P.extend(predictions)
        S.extend(outputs)

        result=compute_metrics(Y, S, P)
        for key,value in result.items():
                print(f'{key}:{value}')
    
    
    def test_cluster(self):
        print('*'*10+'testing_cluster(not fine_tuning)'+'*'*10)
        output_list=[]
        label_list=[]
        Y, P, S = [], [], []
        test_dataset=DTIDataSet(self.test_data)
        test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=True,collate_fn=cluster_collate_fn)
        model_list=[]
        for i in range(5):
            model=EADTN(args=self.args)
            model.load_state_dict(torch.load(f'./check_points/{i}/valid_best_checkpoint.pth'))
            model.eval()
            model.to(self.device)
            model_list.append(model)
        with torch.no_grad():
            for drugs,proteins,labels,clusters in tqdm(test_dataloader):
                drugs,proteins,labels=drugs.to(self.device),proteins.to(self.device),labels.to(self.device)
                label_list.append(int(labels))
                out_sum=0
                for index,model in enumerate(model_list):
                    out=model(drugs,proteins)
                    out_sum+=out
                out_sum/=5
                output_list.append(out_sum.cpu().detach())
        labels =np.array(label_list)
        outputs = torch.nn.functional.softmax(torch.cat(output_list,dim=0), 1).to('cpu').data.numpy()
        predictions = np.argmax(outputs, axis=1)
        outputs = outputs[:, 1]
        Y.extend(labels)
        P.extend(predictions)
        S.extend(outputs)
        result=compute_metrics(Y, S, P)
        for key,value in result.items():
                print(f'{key}:{value}')
    
    
    def grid_search(self, val_dataloader, model_list, clusters_dict):
        best_weights = None
        best_score = float('-inf')
        weight_combinations = list(itertools.product([0.1, 0.15, 0.2, 0.25, 0.3, 0.35], repeat=len(clusters_dict[0])))

        
        for weights in weight_combinations:
            Y, S, P = [], [], []
            with torch.no_grad():
                for drugs, proteins, labels, clusters in tqdm(val_dataloader):
                    drugs, proteins, labels = drugs.to(self.device), proteins.to(self.device), labels.to(self.device)
                    out_sum = 0
                    for index, model in enumerate(model_list):
                        out = model(drugs, proteins)
                        if index in clusters_dict[int(clusters)]:
                            out *= weights[clusters_dict[int(clusters)].index(index)]
                        else:
                            out *= 0.1
                        out_sum += out
                    out_sum /= sum(weights) + 0.2
                    Y.append(int(labels))
                    S.append(out_sum.cpu().detach().numpy())
            result = compute_metrics(Y, S)
            score = result['desired_metric']  # Replace 'desired_metric' with the metric you are optimizing for
            if score > best_score:
                best_score = score
                best_weights = weights

        return best_weights

    def test_cluster_fine_tuning(self,use_default_weights=True):
        print('*' * 10 + 'testing(fine_tuning)' + '*' * 10)
        output_list = []
        label_list = []
        Y, P, S = [], [], []
        test_dataset = DTIDataSet(self.test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, collate_fn=cluster_collate_fn)
        model_list = []
        clusters_dict = {0: [0, 3, 4], 1: [4, 0, 1], 2: [0, 1, 2], 3: [1, 2, 3], 4: [2, 3, 4]}
        
        for i in [f'./check_points/fine_tuning/{j}/valid_best_checkpoint.pth' for j in self.cluster_list]:
            model = EADTN(args=self.args)
            model.load_state_dict(torch.load(i))
            model.eval()
            model.to(self.device)
            model_list.append(model)
        
        combined_val_dataloader = self.get_combined_val_dataloader()
        
        if not use_default_weights:
            best_weights = self.grid_search(combined_val_dataloader, model_list, clusters_dict)
        else:
            best_weights = [0.14, 0.14, 0.14]  # Default weights
        
        with torch.no_grad():
            for drugs, proteins, labels, clusters in tqdm(test_dataloader):
                drugs, proteins, labels = drugs.to(self.device), proteins.to(self.device), labels.to(self.device)
                label_list.append(int(labels))
                out_sum = 0
                for index, model in enumerate(model_list):
                    out = model(drugs, proteins)
                    if index in clusters_dict[int(clusters)]:
                        out *= best_weights[clusters_dict[int(clusters)].index(index)]
                    else:
                        out *= 0.1
                    out_sum += out
                out_sum /= sum(best_weights) + 0.2
                output_list.append(out_sum.cpu().detach())
        
        labels = np.array(label_list)
        outputs = torch.nn.functional.softmax(torch.cat(output_list, dim=0), 1).to('cpu').data.numpy()
        predictions = np.argmax(outputs, axis=1)
        outputs = outputs[:, 1]
        Y.extend(labels)
        P.extend(predictions)
        S.extend(outputs)
        result = compute_metrics(Y, S, P)
        for key, value in result.items():
            print(f'{key}: {value}')

                
                
