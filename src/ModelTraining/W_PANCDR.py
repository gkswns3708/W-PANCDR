import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import pandas as pd
import itertools
israndom=False
from itertools import cycle
from sklearn.model_selection import StratifiedKFold,KFold

from ModelTraining.model import Encoder, GCN, ADV, gradient_penalty, Critic
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class train_W_PANCDR():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self,params,weight_path='../checkpoint/model.pt'):
        nz,d_dim,lr,lr_adv,lam,batch_size = params.values()

        X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = self.train_data
        TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test = self.test_data
        X_t_train, X_t_val = train_test_split(t_gexpr_feature.T.values, test_size=0.05, random_state=0)
        X_drug_feat_data_train,X_drug_feat_data_val,X_drug_adj_data_train,X_drug_adj_data_val,X_gexpr_data_train,X_gexpr_data_val,Y_train,Y_val= train_test_split(X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,test_size=0.05, random_state=0)
        
        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train)
        X_t_gexpr_train = torch.FloatTensor(X_t_train)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)
        
        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size = batch_size[0], shuffle=True, drop_last=True)
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size = batch_size[1], shuffle=True, drop_last=True)

        wait, best_auc = 0, 0
        EN_model = Encoder(X_gexpr_train.shape[1], nz, device)
        GCN_model = GCN(X_drug_feat_train.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=False)
        Critic_model = Critic(nz)
        EN_model.to(device)
        GCN_model.to(device)
        Critic_model.to(device)

        optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(),GCN_model.parameters()), lr=lr)
        optimizer_adv = torch.optim.Adam(Critic_model.parameters(), lr=lr_adv)
        loss = torch.nn.BCELoss()

        # F_gexpr: GDSC의 Latent Vector
        # F_t_gexpr: TCGA의 Latent Vector
        # F_g_t_gexpr: GDSC, TCGA Latent Vector를 Concate한 Vector
        for epoch in range(1000):
            for i,data in enumerate(zip(GDSC_Loader, cycle(E_TEST_Loader))):

                DataG = data[0]
                t_gexpr = data[1][0]
                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1,1).to(device)
                t_gexpr = t_gexpr.to(device)
                EN_model.train()
                GCN_model.train()
                Critic_model.train()

                optimizer_adv.zero_grad()
                with torch.no_grad():
                    print(gexpr.shape, "- W_PANCDR.train_W_PANCDR-gexpr.shape")
                    print(t_gexpr.shape, "- W_PANCDR.train_W_PANCDR-t_gexpr.shape")
                    F_gexpr,_,_ = EN_model(gexpr)      # real (GDSC 라고 가정)
                    F_t_gexpr,_,_ = EN_model(t_gexpr)  # fake (TCGA 라고 가정)
                print(F_gexpr.shape, "- W_PANCDR.train_W_PANCDR-F_gexpr.shape")
                print(F_t_gexpr.shape, "- W_PANCDR.train_W_PANCDR-F_t_gexpr.shape")
                # 2) Critic forward(Score 계산)
                real_validity = Critic_model(F_gexpr)     # D(real)
                fake_validity = Critic_model(F_t_gexpr)   # D(fake)

                # real_validity vs fake_validity 사이의 차이가 Wasserstein Distnace의 근사치가 됨.
                gp = gradient_penalty(Critic_model, F_gexpr, F_t_gexpr, device, gp_weight=10.0)
                critic_loss = (fake_validity.mean() - real_validity.mean()) + gp

                critic_loss.backward()
                optimizer_adv.step()
                
                # TODO: 수정 예정(빼야할지 말아야 할 지.)
                # F_g_t_gexpr = torch.cat((F_gexpr,F_t_gexpr)) 
                # z_true = torch.cat((torch.zeros(F_gexpr.shape[0], device=device), torch.ones(F_t_gexpr.shape[0], device=device)))
                # z_true = z_true.view(-1,1)
                # z_pred = Critic_model(F_g_t_gexpr)
                # if IsNaN(z_pred): return -1
                # adv_loss = loss(z_pred, z_true)
                # adv_loss.backward()
                # optimizer_adv.step()

                # 위쪽 Critic Update 
                # 아래쪽 Encoder + GCN Update 


                optimizer.zero_grad()

                # 1) latent 추출 (Encoder)
                g_latents = EN_model(gexpr)[0]      # real sample
                t_latents = EN_model(t_gexpr)[0]  # fake sample

                # 2) WGAN generator loss = - D(fake)
                fake_validity_ = Critic_model(F_t_gexpr)
                adv_loss_ = -fake_validity_.mean()

                # 3) CDR 분류(또는 회귀) 손실
                # 여기서는 IC50을 binary로 예측함.
                y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
                print(y_pred.shape, y_true.shape, "-W-PANCDR.train_W_PANCDR-y_pred.shape, y_true.shape")
                print(type(y_pred), type(y_true), "-W-PANCDR.train_W_PANCDR-type(y_pred), type(y_true)")
                print(y_pred.min().item(), y_pred.max().item(), "-W-PANCDR.train_W_PANCDR-y_pred.min().item(), y_pred.max().item()")
                cdr_loss = loss(y_pred, y_true)  # (classification인 경우)

                # 4) 최종 Loss = CDR + λ*WGAN
                Loss = cdr_loss + lam * adv_loss_

                Loss.backward()
                optimizer.step()
            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                Critic_model.eval()

                F_gexpr_val,_,_ = EN_model(X_gexpr_val)
                F_t_gexpr_val,_,_ = EN_model(X_t_gexpr_val)

                F_g_t_gexpr_val = torch.cat((F_gexpr_val, F_t_gexpr_val))
                z_pred_val = Critic_model(F_g_t_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
                loss_val = loss(y_pred_val, Y_val.view(-1,1)) + lam*loss(z_pred_val, torch.ones(z_pred_val.shape, device=device))
                auc_val = roc_auc_score(Y_val.cpu().detach().numpy(), y_pred_val.cpu().detach().numpy())
                
                F_TEST_gexpr,_,_ = EN_model(TX_gexpr_data_test)
                y_pred_TEST = GCN_model(TX_drug_feat_data_test, TX_drug_adj_data_test, F_TEST_gexpr)
                auc_TEST = roc_auc_score(TY_test.cpu().detach().numpy(), y_pred_TEST.cpu().detach().numpy())
                
            if auc_val >= best_auc:
                wait = 0
                best_auc = auc_val
                best_auc_TEST = auc_TEST
                torch.save({'EN_model': EN_model.state_dict(), 'GCN_model':GCN_model.state_dict(), 
                            'Critic_model':Critic_model.state_dict()}, weight_path)
                
            else:
                wait += 1
                if wait >= 10: break
        
        return best_auc_TEST

    def predict(self, data, params, weight_path):
        nz,d_dim,_,_,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=
        False)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()

            F_gexpr = EN_model(gexpr)[0]
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)

        return y_pred.cpu()

def train_W_PANCDR_nested(
    data,
    n_outer_splits=10,
    n_inner_splits=5,
    n_random_search=20,
    random_state=0,
    device='cuda'
):
    """
    Nested Cross Validation을 수행하여 outer fold별 best hyperparameter와 test AUC를 구합니다.

    Parameters:
    -----------
    data: (X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature)
    n_outer_splits: 외부 K-fold 개수
    n_inner_splits: 내부 K-fold 개수
    n_random_search: 내부 loop에서 random search를 수행할 횟수
    random_state: 랜덤 시드
    device: cuda 또는 cpu

    Returns:
    --------
    auc_test_df: pd.DataFrame
        - 각 outer fold의 Test_AUC, Best_params,
        - 마지막 row로 'mean' AUC를 추가
    """

    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data

    # 반복 재현성을 위해 random seed 고정
    np.random.seed(random_state)
    random.seed(random_state)

    # StratifiedKFold를 이용하여 outer fold를 정의
    outer_skf = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    auc_test_df = pd.DataFrame(columns=['Test_AUC', 'Best_params'])

    for outer_fold, (train_idx, test_idx) in enumerate(outer_skf.split(X_drug_feat_data, Y)):
        print(f"\n[Outer Fold {outer_fold}] Start")

        #------------------------------
        # 1) Outer fold train / test 분할
        #------------------------------
        X_drug_feat_train = X_drug_feat_data[train_idx]
        X_drug_adj_train  = X_drug_adj_data[train_idx]
        X_gexpr_train     = X_gexpr_data[train_idx]
        Y_train           = Y[train_idx]

        X_drug_feat_test = X_drug_feat_data[test_idx]
        X_drug_adj_test  = X_drug_adj_data[test_idx]
        X_gexpr_test     = X_gexpr_data[test_idx]
        Y_test           = Y[test_idx]

        #------------------------------
        # 2) Inner loop을 위한 K-fold 정의
        #------------------------------
        inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

        # best_params와 best_auc를 추적할 변수
        best_inner_auc = -1.0
        best_params = None

        #------------------------------
        # 3) n_random_search 만큼 랜덤 하이퍼파라미터 샘플링
        #------------------------------
        for _ in range(n_random_search):
            # 각 파라미터를 무작위로 선택
            trial_params = {
                'nz': random.choice(nz_ls),
                'd_dim': random.choice(h_dims_ls),
                'lr': random.choice(lr_ls),
                'lr_adv': random.choice(lr_adv_ls),
                'lam': random.choice(lam_ls),
                'batch_size': random.choice(batch_size_ls)
            }

            #------------------------------
            # 4) Inner 5-fold CV로 trial_params 평가
            #------------------------------
            auc_scores = []
            for inner_train_idx, inner_val_idx in inner_skf.split(X_drug_feat_train, Y_train):
                # inner train/val 분할
                X_drug_feat_inner_train = X_drug_feat_train[inner_train_idx]
                X_drug_adj_inner_train  = X_drug_adj_train[inner_train_idx]
                X_gexpr_inner_train     = X_gexpr_train[inner_train_idx]
                Y_inner_train           = Y_train[inner_train_idx]

                X_drug_feat_inner_val = X_drug_feat_train[inner_val_idx]
                X_drug_adj_inner_val  = X_drug_adj_train[inner_val_idx]
                X_gexpr_inner_val     = X_gexpr_train[inner_val_idx]
                Y_inner_val           = Y_train[inner_val_idx]

                # device로 넘길 tensor 변환(예시)
                X_drug_feat_inner_val_t = torch.FloatTensor(X_drug_feat_inner_val).to(device)
                X_drug_adj_inner_val_t  = torch.FloatTensor(X_drug_adj_inner_val).to(device)
                X_gexpr_inner_val_t     = torch.FloatTensor(X_gexpr_inner_val).to(device)
                Y_inner_val_t           = torch.FloatTensor(Y_inner_val).to(device)

                # 모델 생성
                inner_train_data = [
                    X_drug_feat_inner_train,
                    X_drug_adj_inner_train,
                    X_gexpr_inner_train,
                    Y_inner_train,
                    t_gexpr_feature
                ]
                inner_val_data = [
                    X_drug_feat_inner_val_t,
                    X_drug_adj_inner_val_t,
                    X_gexpr_inner_val_t,
                    Y_inner_val_t
                ]

                model = train_W_PANCDR(inner_train_data, inner_val_data)

                # model.train()이 validation set에 대해 auc를 반환한다고 가정
                auc_val = model.train(trial_params)  
                auc_scores.append(auc_val)

            # inner 5-fold AUC 평균
            mean_auc_score = np.mean(auc_scores)

            # 만약 현재 trial_params가 최고 성능이면 갱신
            if mean_auc_score > best_inner_auc:
                best_inner_auc = mean_auc_score
                best_params = deepcopy(trial_params)

        #------------------------------
        # 5) Outer fold train set 전체로 best_params로 재학습 -> test AUC 산출
        #------------------------------
        # test set을 device로 변환
        X_drug_feat_test_t = torch.FloatTensor(X_drug_feat_test).to(device)
        X_drug_adj_test_t  = torch.FloatTensor(X_drug_adj_test).to(device)
        X_gexpr_test_t     = torch.FloatTensor(X_gexpr_test).to(device)
        Y_test_t           = torch.FloatTensor(Y_test).to(device)

        outer_train_data = [
            X_drug_feat_train,
            X_drug_adj_train,
            X_gexpr_train,
            Y_train,
            t_gexpr_feature
        ]
        outer_test_data = [
            X_drug_feat_test_t,
            X_drug_adj_test_t,
            X_gexpr_test_t,
            Y_test_t
        ]

        final_model = train_W_PANCDR(outer_train_data, outer_test_data)
        
        # weight 저장 경로 설정 (예시)
        weight_path = f'../checkpoint/kfold/W-model_best_outerfold_{outer_fold}.pt'

        # 실제 학습 및 test set AUC 획득
        while True:
            auc_TEST = final_model.train(best_params, weight_path=weight_path)
            # 예시에서 -1이면 에러라고 가정 -> 다시 시도
            if auc_TEST != -1:
                break

        # 결과 저장
        temp_test_df = pd.DataFrame(
            [[auc_TEST, best_params]],
            index=[f'Fold_{outer_fold}'],
            columns=['Test_AUC', 'Best_params']
        )
        auc_test_df = pd.concat([auc_test_df, temp_test_df])
        print(f"[Outer Fold {outer_fold}] Best Params: {best_params}, Test AUC: {auc_TEST:.4f}")

    #------------------------------
    # 6) 전체 outer fold에 대한 mean AUC
    #------------------------------
    mean_auc = auc_test_df['Test_AUC'].mean()
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Mean test AUC = {mean_auc:.4f}')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

    # mean 행을 추가
    auc_test_df = pd.concat([
        auc_test_df,
        pd.DataFrame([mean_auc], index=['mean'], columns=['Test_AUC'])
    ])

    return auc_test_df

class train_PANCDR_regr():
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
    def train(self,params,weight_path='../checkpoint/kfold/model.pt'):
        nz,d_dim,lr,lr_adv,lam,batch_size = params.values()

        X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = self.train_data
        TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test = self.test_data
        X_t_train, X_t_val = train_test_split(t_gexpr_feature.T.values, test_size=0.05, random_state=0)
        X_drug_feat_data_train,X_drug_feat_data_val,X_drug_adj_data_train,X_drug_adj_data_val,X_gexpr_data_train,X_gexpr_data_val,Y_train,Y_val= train_test_split(X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,test_size=0.05, random_state=0)
        
        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train)
        X_t_gexpr_train = torch.FloatTensor(X_t_train)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)
        
        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size = batch_size[0], shuffle=True, drop_last=True)
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size = batch_size[1], shuffle=True, drop_last=True)

        wait, best_p = 0, 0
        EN_model = Encoder(X_gexpr_train.shape[1], nz, device)
        GCN_model = GCN(X_drug_feat_train.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=False,is_regr=True)
        Critic_model = Critic(nz)
        EN_model.to(device)
        GCN_model.to(device)
        Critic_model.to(device)

        optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(),GCN_model.parameters()), lr=lr)
        optimizer_adv = torch.optim.Adam(Critic_model.parameters(), lr=lr_adv)
        criterion = torch.nn.MSELoss()
        loss = torch.nn.BCELoss()

        for epoch in range(1000):
            for i,data in enumerate(zip(GDSC_Loader, cycle(E_TEST_Loader))):

                DataG = data[0]
                t_gexpr = data[1][0]
                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1,1).to(device)
                t_gexpr = t_gexpr.to(device)
                EN_model.train()
                GCN_model.train()
                Critic_model.train()

                optimizer_adv.zero_grad()
                F_gexpr,_,_ = EN_model(gexpr)
                F_t_gexpr,_,_ = EN_model(t_gexpr)

                F_g_t_gexpr = torch.cat((F_gexpr,F_t_gexpr))
                z_true = torch.cat((torch.zeros(F_gexpr.shape[0], device=device), torch.ones(F_t_gexpr.shape[0], device=device)))
                z_true = z_true.view(-1,1)
                z_pred = Critic_model(F_g_t_gexpr)
                if IsNaN(z_pred): return -1
                adv_loss = loss(z_pred, z_true)
                adv_loss.backward()
                optimizer_adv.step()

                optimizer.zero_grad()

                g_latents, _, _ = EN_model(gexpr)
                t_latents, _, _ = EN_model(t_gexpr)

                F_g_t_latents = torch.cat((g_latents,t_latents))
                z_true_ = torch.cat((torch.ones(g_latents.shape[0], device=device), torch.zeros(t_latents.shape[0], device=device)))
                z_true_ = z_true_.view(-1,1)
                z_pred_ = Critic_model(F_g_t_latents)
                y_pred = GCN_model(drug_feat,drug_adj,g_latents)
                if IsNaN(z_pred_) or IsNaN(y_pred): return -1
                
                adv_loss_ = loss(z_pred_, z_true_)
                cdr_loss = criterion(y_pred, y_true)

                Loss = cdr_loss + lam*adv_loss_
                Loss.backward()
                optimizer.step()

            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                Critic_model.eval()

                F_gexpr_val,_,_ = EN_model(X_gexpr_val)
                F_t_gexpr_val,_,_ = EN_model(X_t_gexpr_val)

                F_g_t_gexpr_val = torch.cat((F_gexpr_val, F_t_gexpr_val))
                z_pred_val = Critic_model(F_g_t_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
                loss_val = criterion(y_pred_val, Y_val.view(-1,1)) + lam*loss(z_pred_val, torch.ones(z_pred_val.shape, device=device))
                p_val = pearsonr(Y_val.cpu().view(-1).detach().numpy(), y_pred_val.cpu().view(-1).detach().numpy())[0]
                
            if p_val >= best_p:#loss_val <= best_loss:
                wait = 0
                best_p = p_val
                best_loss = loss_val
                torch.save({'EN_model': EN_model.state_dict(), 'GCN_model':GCN_model.state_dict(), 
                            'Critic_model':Critic_model.state_dict()}, weight_path)

                
            else:
                wait += 1
                if wait >= 10: break
        
        return best_p

    def predict(self, data, params, weight_path):
        nz,d_dim,_,_,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=
        False,is_regr=True)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()

            F_gexpr = EN_model(gexpr)[0]
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)

        return y_pred.cpu()

def train_PANCDR_nested_regr(n_outer_splits,data,best_params_file):
    outer_splits = KFold(n_splits=n_outer_splits,shuffle=True,random_state=0)
    p_test_df = pd.DataFrame(columns=['Test_Pearson','Best_params'])
    X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = data
    best_params_df = pd.read_csv(best_params_file,index_col=0)
    for outer_fold,(idx,test_idx) in enumerate(outer_splits.split(X_drug_feat_data)):
        X_drug_feat_data_ = X_drug_feat_data[idx]
        X_drug_adj_data_ = X_drug_adj_data[idx]
        X_gexpr_data_ = X_gexpr_data[idx]
        Y_ = Y[idx]
        best_params = eval(best_params_df.loc["Fold_%d"%outer_fold,"Best_params"])
        weight_path='../checkpoint/kfold/regr_model_best_outerfold_%d.pt'%outer_fold

        X_drug_feat_data_test = X_drug_feat_data[test_idx]
        X_drug_adj_data_test = X_drug_adj_data[test_idx]
        X_gexpr_data_test = X_gexpr_data[test_idx]
        Y_test = Y[test_idx]

        X_drug_feat_test = torch.FloatTensor(X_drug_feat_data_test).to(device)
        X_drug_adj_test = torch.FloatTensor(X_drug_adj_data_test).to(device)
        X_gexpr_test = torch.FloatTensor(X_gexpr_data_test).to(device)
        Y_test = torch.FloatTensor(Y_test).to(device)

        train_data = [X_drug_feat_data_,X_drug_adj_data_,X_gexpr_data_,Y_,t_gexpr_feature]
        test_data = [X_drug_feat_test,X_drug_adj_test,X_gexpr_test,Y_test]

        model = train_PANCDR_regr(train_data,test_data)
        while True:
            p_val = model.train(best_params, weight_path)
            if p_val != -1: break
        y_pred_TEST = model.predict(test_data[:-1],best_params, weight_path)
        p_TEST = pearsonr(Y_test.cpu().view(-1).detach().numpy(), y_pred_TEST.view(-1).detach().numpy())[0]
        temp_test_df = pd.DataFrame([[p_TEST,best_params]], index=['Fold_%d'%outer_fold], columns=['Test_Pearson','Best_params'])
        p_test_df = pd.concat([p_test_df,temp_test_df])

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('Mean test Pearson - %.4f\n'%p_test_df['Test_Pearson'].mean())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    p_test_df = pd.concat([p_test_df, pd.DataFrame(p_test_df['Test_Pearson'].mean(), index=['mean'], columns=['Test_Pearson'])])
    return p_test_df

def IsNaN(pred):
    return torch.isnan(pred).sum()>0
