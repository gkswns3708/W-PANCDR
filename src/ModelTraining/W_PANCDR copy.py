from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import wandb
import numpy as np
import pandas as pd

import torch

import os
import random
import itertools
from tqdm import tqdm
from itertools import cycle


# TODO: Variational Encoder를 추가 혹은 다른 버전의 Encoder를 추가. 현재까지는 FC Layer만 존재
from utils import scores 
from ModelTraining.model import Encoder_FC, GCN, ADV, gradient_penalty, Critic

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class train_W_PANCDR():
    def __init__(self, train_data, test_data, outer_fold=None, project=None):
        self.train_data = train_data
        self.test_data = test_data
        self.outer_fold = outer_fold
        self.project = project

    def train(self, params, weight_path='../checkpoint/model.pt'):
        run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_batch_size_{params['batch_size'][0]}_lam_{params['lam']}"
    
        # 각 실험(run)을 새로 시작
        run = wandb.init(
            project=self.project,
            config=params,
            name=run_name,
            reinit=True
        )

            
        nz, d_dim, lr, lr_critic, lam, batch_size = params.values()
        print("Hyperparameters:")
        print(f"nz: {nz}")
        print(f"d_dim: {d_dim}")
        print(f"lr: {lr}")
        print(f"lr_critic: {lr_critic}")
        print(f"lam: {lam}")
        print(f"batch_size: {batch_size}")
        print(f"device: {device}")

        # 데이터 분할 및 Tensor 변환
        X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = self.train_data
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = self.test_data
        X_t_train, X_t_val = train_test_split(t_gexpr_feature.T.values, test_size=0.05, random_state=0)
        X_drug_feat_data_train, X_drug_feat_data_val, X_drug_adj_data_train, X_drug_adj_data_val, \
        X_gexpr_data_train, X_gexpr_data_val, Y_train, Y_val = train_test_split(
            X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, test_size=0.05, random_state=0)
        
        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train).to(device)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train).to(device)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train).to(device)
        X_t_gexpr_train = torch.FloatTensor(X_t_train).to(device)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)
        
        TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
        TX_drug_adj_data_test  = torch.FloatTensor(TX_drug_adj_data_test).to(device)
        TX_gexpr_data_test     = torch.FloatTensor(TX_gexpr_data_test).to(device)
        TY_test                = torch.FloatTensor(TY_test).to(device)

        print(f"TX_drug_feat_data_test : {TX_drug_feat_data_test.device}")
        print(f"TX_drug_adj_data_test  : {TX_drug_adj_data_test.device}")
        print(f"TX_gexpr_data_test     : {TX_gexpr_data_test.device}")
        print(f"TY_test                : {TY_test.device}")

        # DataLoader 생성
        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size=batch_size[0], shuffle=True, drop_last=True)
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size=batch_size[1], shuffle=True, drop_last=True)

        wait, best_auc = 0, 0
        # 모델 생성
        EN_model = Encoder_FC(X_gexpr_train.shape[1], nz)
        GCN_model = GCN(X_drug_feat_train.shape[2], [256,256,256], h_dims=[d_dim, nz+d_dim], use_dropout=False)
        Critic_model = Critic(nz)
        EN_model.to(device)
        GCN_model.to(device)
        Critic_model.to(device)

        optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(), GCN_model.parameters()), lr=lr)
        optimizer_critic = torch.optim.Adam(Critic_model.parameters(), lr=lr_critic)
        loss_fn = torch.nn.BCELoss()

        current_epoch = -1
        for epoch in tqdm(range(1000), desc="Epoch", leave=True):
            total_critic_loss = 0.0
            total_gen_loss = 0.0
            num_batches = 0
            # training metric 저장을 위한 리스트
            train_y_true_list = []
            train_y_pred_list = []
            
            # 각 배치마다 학습 (원래 방식 그대로)
            for i, data in enumerate(tqdm(zip(GDSC_Loader, cycle(E_TEST_Loader)), desc=f"Batch (Epoch {epoch})", leave=False, total=len(GDSC_Loader))):
                DataG = data[0]
                t_gexpr = data[1][0]
                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1, 1).to(device)
                t_gexpr = t_gexpr.to(device)

                EN_model.train()
                GCN_model.train()
                Critic_model.train()

                # Critic 업데이트 (Encoder 고정)
                optimizer_critic.zero_grad()
                with torch.no_grad():
                    F_gexpr = EN_model(gexpr)      # fake (GDSC)
                    F_t_gexpr = EN_model(t_gexpr)    # real (TCGA)
                real_validity = Critic_model(F_t_gexpr)  # D(real)
                fake_validity = Critic_model(F_gexpr)      # D(fake)
                gp = gradient_penalty(Critic_model, F_gexpr, F_t_gexpr, device, gp_weight=10.0)
                critic_loss = (fake_validity.mean() - real_validity.mean()) + gp
                critic_loss.backward()
                optimizer_critic.step()

                # Encoder + GCN 업데이트 (Generator 역할)
                optimizer.zero_grad()
                # latent 추출 (tuple이면 첫번째 요소 사용)
                F_gexpr = EN_model(gexpr)[0] if isinstance(EN_model(gexpr), (list, tuple)) else EN_model(gexpr)
                F_t_gexpr = EN_model(t_gexpr)[0] if isinstance(EN_model(t_gexpr), (list, tuple)) else EN_model(t_gexpr)
                fake_validity_ = Critic_model(F_gexpr)
                adv_loss = -fake_validity_.mean()
                y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
                cdr_loss = loss_fn(y_pred, y_true)
                gen_loss = cdr_loss + lam * adv_loss
                gen_loss.backward()
                optimizer.step()
                
                total_critic_loss += critic_loss.item()
                total_gen_loss += gen_loss.item()
                num_batches += 1
                
                # 학습 중 매 배치의 y_true와 y_pred를 저장 (평가를 위해)
                train_y_true_list.append(y_true.cpu().detach().numpy().flatten())
                train_y_pred_list.append(y_pred.cpu().detach().numpy().flatten())
            
            # Epoch마다 평균 training loss 계산
            avg_critic_loss = total_critic_loss / num_batches
            avg_gen_loss = total_gen_loss / num_batches
            
            # training metric 계산 (전체 training 데이터에 대해)
            train_y_true = np.concatenate(train_y_true_list)
            train_y_pred = np.concatenate(train_y_pred_list)
            train_auc, train_acc, train_precision, train_recall, train_f1 = scores(train_y_true, train_y_pred)
            
            # Epoch 단위 평가: 전체 Validation 및 Test 데이터를 이용하여 한 번에 예측
            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                # Validation 평가
                F_val = EN_model(X_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_val)
                y_true_val = Y_val.cpu().detach().numpy().flatten()
                y_pred_val_np = y_pred_val.cpu().detach().numpy().flatten()
                val_auc, val_acc, val_precision, val_recall, val_f1 = scores(y_true_val, y_pred_val_np)
                
                # Test 평가
                F_test = EN_model(TX_gexpr_data_test)
                y_pred_test = GCN_model(TX_drug_feat_data_test, TX_drug_adj_data_test, F_test)
                y_true_test = TY_test.cpu().detach().numpy().flatten()
                y_pred_test_np = y_pred_test.cpu().detach().numpy().flatten()
                test_auc, test_acc, test_precision, test_recall, test_f1 = scores(y_true_test, y_pred_test_np)

                # wandb에 epoch 단위로 로깅 (평균 training loss 및 training metric 포함)
                wandb.log({
                    "epoch": epoch,
                    "avg_critic_loss": avg_critic_loss,
                    "avg_gen_loss": avg_gen_loss,
                    "train_auc": train_auc,
                    "train_acc": train_acc,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "test_auc": test_auc,
                    "test_acc": test_acc,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                    "loss_val": gen_loss.item()  # 마지막 배치의 Loss (선택 사항)
                })
                
                # print(f"Epoch {epoch} - TRAIN: AUC: {train_auc:.4f}, ACC: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
                # print(f"Epoch {epoch} - VAL: AUC: {val_auc:.4f}, ACC: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
                # print(f"Epoch {epoch} - TEST: AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
            
            if val_auc >= best_auc:
                wait = 0
                best_auc = val_auc
                torch.save({
                    'EN_model': EN_model.state_dict(), 
                    'GCN_model': GCN_model.state_dict(), 
                    'Critic_model': Critic_model.state_dict()
                }, weight_path)
                ret_metric = {
                    'train_auc': train_auc,
                    'train_acc': train_acc,
                    'train_precision': train_precision,
                    'train_recall': train_recall,
                    'train_f1': train_f1,
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'test_auc': test_auc,
                    'test_acc': test_acc,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1
                }
            else:
                wait += 1
                if wait >= 10: 
                    break
            current_epoch = epoch
        
        run.finish()  # wandb run 종료
        return test_auc, current_epoch

    def predict(self, data, params, weight_path):
        nz, d_dim, _, _, _, _ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder_FC(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2], [256,256,256], h_dims=[d_dim, nz+d_dim], use_dropout=False)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()
            F_gexpr = EN_model(gexpr)[0] if isinstance(EN_model(gexpr), (list, tuple)) else EN_model(gexpr)
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
        return y_pred.cpu()




def train_W_PANCDR_nested(n_outer_splits, data, best_params_file, 
                          result_file='GDSC_nested_all_results.csv', 
                          best_file='GDSC_nested_best_results.csv'):
    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    outer_splits = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=0)
    print(n_outer_splits, "- n_outer_splits")
    outer_folds = outer_splits.split(X_drug_feat_data, Y)
    
    # Load all best params for all folds and all random samples
    best_params_df = pd.read_csv(best_params_file)
    
    # Initialize result DataFrames
    auc_test_df = pd.DataFrame(columns=['Fold', 'Iteration', 'Test_AUC', 'Best_params', 'Best_epoch'])
    best_params_all_folds = pd.DataFrame(columns=['Fold', 'Test_AUC', 'Best_params', 'Best_epoch'])

    for outer_fold, (idx, test_idx) in enumerate(outer_folds):
        fold_params = best_params_df[best_params_df['Fold'] == f"Fold_{outer_fold}"]
        
        X_drug_feat_data_ = X_drug_feat_data[idx]
        X_drug_adj_data_ = X_drug_adj_data[idx]
        X_gexpr_data_ = X_gexpr_data[idx]
        Y_ = Y[idx]

        X_drug_feat_data_test = X_drug_feat_data[test_idx]
        X_drug_adj_data_test = X_drug_adj_data[test_idx]
        X_gexpr_data_test = X_gexpr_data[test_idx]
        Y_test = Y[test_idx]

        X_drug_feat_test = torch.FloatTensor(X_drug_feat_data_test)
        X_drug_adj_test = torch.FloatTensor(X_drug_adj_data_test)
        X_gexpr_test = torch.FloatTensor(X_gexpr_data_test)
        Y_test = torch.FloatTensor(Y_test)

        train_data = [X_drug_feat_data_, X_drug_adj_data_, X_gexpr_data_, Y_, t_gexpr_feature]
        test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

        best_auc = -1
        best_params = None
        best_epoch = -1

        for i, row in fold_params.iterrows():
            current_params = eval(row['Best_params'])
            model = train_W_PANCDR(train_data, test_data, outer_fold=outer_fold)
            
            epoch_counter = 0  # 추가: Epoch 번호 기록용
            
            while True:
                auc_TEST, current_epoch = model.train(current_params, weight_path=f'../checkpoint/GDSC_kfold/model_best_outerfold_{outer_fold}_{i}.pt')
                epoch_counter += 1  # Epoch 증가
                
                if auc_TEST != -1:
                    break
            
            # Save each result immediately to the all results file
            auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[f"Fold_{outer_fold}", i, auc_TEST, current_params, current_epoch]],
                                                               columns=['Fold', 'Iteration', 'Test_AUC', 'Best_params', 'Best_epoch'])])
            auc_test_df.to_csv(result_file, index=False)  # 모든 학습 결과 저장

            if auc_TEST > best_auc:
                best_auc = auc_TEST
                best_params = current_params
                best_epoch = current_epoch  # 최적 AUC를 기록한 Epoch 저장

        # Save the best result for the current fold
        best_params_all_folds = pd.concat([best_params_all_folds, pd.DataFrame([[f"Fold_{outer_fold}", best_auc, best_params, best_epoch]],
                                                                               columns=['Fold', 'Test_AUC', 'Best_params', 'Best_epoch'])])
        best_params_all_folds.to_csv(best_file, index=False)  # Best 결과 저장

        print(f'Fold {outer_fold} - Best AUC: {best_auc:.4f} (Epoch: {best_epoch})')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print(f'Mean test AUC - {auc_test_df["Test_AUC"].mean():.4f}\n')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[None, None, auc_test_df['Test_AUC'].mean(), None, None]],
                                                       columns=['Fold', 'Iteration', 'Test_AUC', 'Best_params', 'Best_epoch'])])
    auc_test_df.to_csv(result_file, index=False)
    best_params_all_folds.to_csv(best_file, index=False)

    return auc_test_df

def train_W_PANCDR_full_cv(n_splits, data, best_params_file, 
                           result_file='./logs/full_CV/GDSC_results.csv',
                           param_summary_file='./logs/full_CV/GDSC_param_summary.csv'):

    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_list = pd.read_csv(best_params_file)['Best_params'].tolist()

    auc_test_df = pd.DataFrame(columns=['Fold', 'Test_AUC', 'params', 'end_epoch'])
    best_params_all_folds = pd.DataFrame(columns=['Fold', 'Test_AUC', 'params', 'end_epoch'])
    for i, param_str in enumerate(param_list):
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_drug_feat_data, Y)):
            print(f"Fold {fold} - Iteration {i} - Current params: {param_str}")
            X_drug_feat_train = X_drug_feat_data[train_idx]
            X_drug_adj_train = X_drug_adj_data[train_idx]
            X_gexpr_train = X_gexpr_data[train_idx]
            Y_train = Y[train_idx]

            X_drug_feat_test = X_drug_feat_data[test_idx]
            X_drug_adj_test = X_drug_adj_data[test_idx]
            X_gexpr_test = X_gexpr_data[test_idx]
            Y_test = Y[test_idx]

            X_drug_feat_test = torch.FloatTensor(X_drug_feat_test)
            X_drug_adj_test = torch.FloatTensor(X_drug_adj_test)
            X_gexpr_test = torch.FloatTensor(X_gexpr_test)
            Y_test = torch.FloatTensor(Y_test)

            train_data = [X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train, t_gexpr_feature]
            test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

            current_params = eval(param_str)
            model = train_W_PANCDR(train_data, test_data, outer_fold=fold, project='W-PANCDR_fullCV')

            while True:
                auc_TEST, end_epoch = model.train(current_params, weight_path=f'../checkpoint/GDSC_fullCV/model_fold_{fold}_{i}.pt')
                if auc_TEST != -1:
                    break

            auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[fold, auc_TEST, current_params, end_epoch]],
                                                            columns=['Fold','Test_AUC', 'Best_params', 'Best_epoch'])])
            auc_test_df.to_csv(result_file, index=False)

    summary_df = (
        auc_test_df
        .groupby("params")["Test_AUC"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    # ▶ 전체 Fold 평균 AUC 별도 row 추가 (원하면)
    overall_row = pd.DataFrame([{
        "params":             "Mean_AUC_over_all_folds",
        "mean":               auc_test_df["Test_AUC"].mean(),
        "std":                auc_test_df["Test_AUC"].std(),
        "count":              10  # n_splits
    }])
    summary_df = pd.concat([summary_df, overall_row], ignore_index=True)

    # ▶ CSV로 저장
    summary_df.to_csv("./logs/full_CV/GDSC_param_summary.csv", index=False)

    print("✅ Summary-version saved:")
    print(summary_df)

    return auc_test_df



def IsNaN(pred):
    return torch.isnan(pred).sum()>0
