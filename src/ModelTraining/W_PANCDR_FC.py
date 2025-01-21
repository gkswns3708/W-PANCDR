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
from tqdm import tqdm

from ModelTraining.model import Encoder_FC, GCN, ADV, gradient_penalty, Critic
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
        nz,d_dim,lr,lr_critic,lam,batch_size = params.values()
        print("Hyperparameters:")
        print(f"weight_path: {weight_path}")
        print(f"nz: {nz}")
        print(f"d_dim: {d_dim}")
        print(f"lr: {lr}")
        print(f"lr_critic: {lr_critic}")
        print(f"lam: {lam}")
        print(f"batch_size: {batch_size}")

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
        
        TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
        TX_drug_adj_data_test  = torch.FloatTensor(TX_drug_adj_data_test).to(device)
        TX_gexpr_data_test     = torch.FloatTensor(TX_gexpr_data_test).to(device)
        TY_test                = torch.FloatTensor(TY_test).to(device)

        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size = batch_size[0], shuffle=True, drop_last=True)
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size = batch_size[1], shuffle=True, drop_last=True)

        wait, best_auc = 0, 0
        EN_model = Encoder_FC(X_gexpr_train.shape[1], nz)
        print("EN_model :", EN_model)
        GCN_model = GCN(X_drug_feat_train.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=False)
        Critic_model = Critic(nz)
        EN_model.to(device)
        GCN_model.to(device)
        Critic_model.to(device)

        optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(),GCN_model.parameters()), lr=lr)
        optimizer_critic = torch.optim.Adam(Critic_model.parameters(), lr=lr_critic)
        loss = torch.nn.BCELoss()

        # F_gexpr: GDSC의 Latent Vector
        # F_t_gexpr: TCGA의 Latent Vector
        # F_g_t_gexpr: GDSC, TCGA Latent Vector를 Concate한 Vector
        for epoch in tqdm(range(1000), desc="Epoch", leave=True):
            for i, data in enumerate(tqdm(zip(GDSC_Loader, cycle(E_TEST_Loader)), desc=f"Batch (Epoch {epoch})", leave=False, total=len(GDSC_Loader))):

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

                optimizer_critic.zero_grad()
                # 이렇게 해야지만 encoder model(generator)이 학습하지 않고 critic model(discriminator)만 학습을 함.
                with torch.no_grad():
                    # print(gexpr.shape, "- W_PANCDR.train_W_PANCDR-gexpr.shape")
                    # print(t_gexpr.shape, "- W_PANCDR.train_W_PANCDR-t_gexpr.shape")
                    F_gexpr = EN_model(gexpr)      # fake (GDSC 라고 가정)
                    F_t_gexpr = EN_model(t_gexpr)  # real (TCGA 라고 가정)
                # print(F_gexpr.shape, "- W_PANCDR.train_W_PANCDR-F_gexpr.shape")
                # print(F_t_gexpr.shape, "- W_PANCDR.train_W_PANCDR-F_t_gexpr.shape")
                # 2) Critic forward(Score 계산)
                real_validity = Critic_model(F_t_gexpr) # D(real)
                fake_validity = Critic_model(F_gexpr)   # D(fake)

                # real_validity vs fake_validity 사이의 차이가 Wasserstein Distnace의 근사치가 됨.
                gp = gradient_penalty(Critic_model, F_gexpr, F_t_gexpr, device, gp_weight=10.0)
                critic_loss = (fake_validity.mean() - real_validity.mean()) + gp

                critic_loss.backward()
                optimizer_critic.step()
                
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
                g_latents = EN_model(gexpr)    # fake sample
                t_latents = EN_model(t_gexpr)  # real sample

                # 2) WGAN generator loss = - D(fake)
                fake_validity_ = Critic_model(F_gexpr)
                adv_loss_ = -fake_validity_.mean()

                # 3) CDR 분류(또는 회귀) 손실
                # 여기서는 IC50을 binary로 예측함.
                y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
                # print(y_pred.shape, y_true.shape, "-W-PANCDR.train_W_PANCDR-y_pred.shape, y_true.shape")
                # print(type(y_pred), type(y_true), "-W-PANCDR.train_W_PANCDR-type(y_pred), type(y_true)")
                # print(y_pred.min().item(), y_pred.max().item(), "-W-PANCDR.train_W_PANCDR-y_pred.min().item(), y_pred.max().item()")
                cdr_loss = loss(y_pred, y_true)  # (classification인 경우)

                # 4) 최종 Loss = CDR + λ*WGAN
                Loss = cdr_loss + lam * adv_loss_

                Loss.backward()
                optimizer.step()
            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                Critic_model.eval()

                # Validation 데이터의 latent 벡터를 추출
                F_gexpr_val = EN_model(X_gexpr_val)
                F_t_gexpr_val = EN_model(X_t_gexpr_val)

                # Critic 모델로 Wasserstein Distance 계산
                real_validity_val = Critic_model(F_t_gexpr_val)  # D(real)
                fake_validity_val = Critic_model(F_gexpr_val)   # D(fake)

                # Validation Loss: Classification Loss + Wasserstein Loss
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
                cdr_loss_val = loss(y_pred_val, Y_val.view(-1, 1))  # Classification Loss
                wasserstein_loss_val = fake_validity_val.mean() - real_validity_val.mean()  # Wasserstein Loss

                # 최종 Validation Loss 계산
                loss_val = cdr_loss_val + lam * wasserstein_loss_val

                # AUC 계산
                auc_val = roc_auc_score(Y_val.cpu().detach().numpy(), y_pred_val.cpu().detach().numpy())
                print(auc_val, "-auc_val")
                # 테스트 데이터에서의 성능 평가
                F_TEST_gexpr = EN_model(TX_gexpr_data_test)
                y_pred_TEST = GCN_model(TX_drug_feat_data_test, TX_drug_adj_data_test, F_TEST_gexpr)
                auc_TEST = roc_auc_score(TY_test.cpu().detach().numpy(), y_pred_TEST.cpu().detach().numpy())
                print(auc_TEST, "-auc_TEST")
            if auc_val >= best_auc:
                wait = 0
                best_auc = auc_val
                best_auc_TEST = auc_TEST
                torch.save({'EN_model': EN_model.state_dict(), 'GCN_model':GCN_model.state_dict(), 
                            'Critic_model':Critic_model.state_dict()}, weight_path)
                
            else:
                wait += 1
                if wait >= 30: break
        
        return best_auc_TEST

    def predict(self, data, params, weight_path):
        nz,d_dim,_,_,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder_FC(gexpr.shape[1], nz)
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

def IsNaN(pred):
    return torch.isnan(pred).sum()>0
