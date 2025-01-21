import numpy as np
import pandas as pd
import torch
import random
from utils import DataGenerate, DataFeature
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from ModelTraining.W_PANCDR import train_W_PANCDR
import os
# 가령, 이런 식으로 하이퍼파라미터 후보 리스트를 정의합니다.
nz_ls = [100, 128, 256]
h_dims_ls = [100, 128, 256]
lr_ls = [0.001, 0.0001]
lr_adv_ls = [0.001, 0.0001]  # lr_adv도 존재한다고 가정
lam_ls = [1, 0.1, 0.01]
batch_size_ls = [[128, 14], [256, 28]]

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TCGA_label_set = ["ACC","ALL","BLCA","BRCA","CESC","CLL","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG","LUSC","MB",
                  "MESO","MM","NB","OV","PAAD","PRAD","SCLC","SKCM","STAD",
                  "THCA","UCEC",'COAD/READ','']
DPATH = '../data'
Drug_info_file = '%s/GDSC/GDSC_drug_binary.csv'%DPATH
Cell_line_info_file = '%s/GDSC/Cell_Lines_Details.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Cancer_response_exp_file = '%s/GDSC/GDSC_binary_response_151.csv'%DPATH
Gene_expression_file = '%s/GDSC/GDSC_expr_z_702.csv'%DPATH
Max_atoms = 100
P_Gene_expression_file = '%s/TCGA/Pretrain_TCGA_expr_702_01A.csv'%DPATH
T_Drug_info_file = '%s/TCGA/TCGA_drug_new.csv'%DPATH
T_Patient_info_file = '%s/TCGA/TCGA_type_new.txt'%DPATH
T_Drug_feature_file = '%s/TCGA/drug_graph_feat'%DPATH
T_Cancer_response_exp_file = '%s/TCGA/TCGA_response_new.csv'%DPATH
T_Gene_expression_file = '%s/TCGA/TCGA_expr_z_702.csv'%DPATH


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
            random.seed(None)  # 시드 해제
            trial_params = {
                'nz': random.choice(nz_ls),
                'd_dim': random.choice(h_dims_ls),
                'lr': random.choice(lr_ls),
                'lr_adv': random.choice(lr_adv_ls),
                'lam': random.choice(lam_ls),
                'batch_size': random.choice(batch_size_ls)
            }
            random.seed(random_state)  # 시드 고정


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


# ------------------------
# 예시 사용법 (data 준비)
# ------------------------
if __name__ == "__main__":
    
    # drug_feature: 약물의 특징(feature)을 나타내는 데이터 (GDSC 데이터셋 기준)
    # gexpr_feature: 유전자 발현 데이터(feature)를 나타내는 데이터 (GDSC 데이터셋 기준)
    # t_gexpr_feature: 사전 학습된 TCGA 유전자 발현 데이터(feature)를 나타내는 데이터
    # data_idx: GDSC 데이터셋에서 각 데이터 인덱스 정보를 포함한 리스트

    # T_drug_feature: TCGA 데이터셋 기준 약물 특징(feature)을 나타내는 데이터
    # T_gexpr_feature: TCGA 데이터셋 기준 유전자 발현 데이터(feature)를 나타내는 데이터
    # T_data_idx: TCGA 데이터셋에서 각 데이터 인덱스 정보를 포함한 리스트

    # TX_drug_data_test: TCGA 테스트 데이터셋에서 약물 데이터를 포함한 리스트 (약물의 특징 및 인접 행렬 구조 포함)
    # TX_gexpr_data_test: TCGA 테스트 데이터셋에서 유전자 발현 데이터를 포함한 리스트
    # TY_test: TCGA 테스트 데이터셋에서 약물 반응 레이블 (타겟 변수)
    # Tcancer_type_test_list: TCGA 테스트 데이터셋에서 암종(cancer type)에 대한 정보를 포함한 리스트


    # GDSC_drug_feature: 약물의 특징(feature)을 나타내는 데이터 (GDSC 데이터셋 기준)
    # GDSC_gexpr_feature: 유전자 발현 데이터(feature)를 나타내는 데이터 (GDSC 데이터셋 기준)
    # GDSC_t_gexpr_feature: 사전 학습된 TCGA 유전자 발현 데이터(feature)를 나타내는 데이터 (GDSC 데이터셋 기준)
    # GDSC_data_idx: GDSC 데이터셋에서 각 데이터 인덱스 정보를 포함한 리스트

    # TCGA_drug_feature: TCGA 데이터셋 기준 약물 특징(feature)을 나타내는 데이터
    # TCGA_gexpr_feature: TCGA 데이터셋 기준 유전자 발현 데이터(feature)를 나타내는 데이터
    # TCGA_data_idx: TCGA 데이터셋에서 각 데이터 인덱스 정보를 포함한 리스트

    # TCGA_X_drug_data_test: TCGA 테스트 데이터셋에서 약물 데이터를 포함한 리스트 (약물의 특징 및 인접 행렬 구조 포함)
    # TCGA_X_gexpr_data_test: TCGA 테스트 데이터셋에서 유전자 발현 데이터를 포함한 리스트
    # TCGA_Y_test: TCGA 테스트 데이터셋에서 약물 반응 레이블 (타겟 변수)
    # TCGA_cancer_type_test_list: TCGA 테스트 데이터셋에서 암종(cancer type)에 대한 정보를 포함한 리스트


    drug_feature,gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file)
    T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,None,T_Cancer_response_exp_file,dataset="TCGA")
    TX_drug_data_test,TX_gexpr_data_test,TY_test,Tcancer_type_test_list = DataFeature(T_data_idx,T_drug_feature,T_gexpr_feature,dataset="TCGA")

    TX_drug_feat_data_test = [item[0] for item in TX_drug_data_test]
    TX_drug_adj_data_test = [item[1] for item in TX_drug_data_test]
    TX_drug_feat_data_test = np.array(TX_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    TX_drug_adj_data_test = np.array(TX_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  

    TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
    TX_drug_adj_data_test = torch.FloatTensor(TX_drug_adj_data_test).to(device)
    TX_gexpr_data_test = torch.FloatTensor(TX_gexpr_data_test).to(device)
    TY_test = torch.FloatTensor(TY_test).to(device)

    X_drug_data,X_gexpr_data,Y,cancer_type_train_list = DataFeature(data_idx,drug_feature,gexpr_feature)
    X_drug_feat_data = [item[0] for item in X_drug_data]
    X_drug_adj_data = [item[1] for item in X_drug_data]
    X_drug_feat_data = np.array(X_drug_feat_data)
    X_drug_adj_data = np.array(X_drug_adj_data)

    data = [X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature]
    

    # Nested CV 수행
    result_df = train_W_PANCDR_nested(
        data,
        n_outer_splits=10,
        n_inner_splits=5,
        n_random_search=20,
        random_state=42,
        device='cpu'
    )

    # 결과 확인
    print(result_df)
