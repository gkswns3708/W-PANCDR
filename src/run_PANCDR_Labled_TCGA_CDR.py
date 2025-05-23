import random,os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
from sklearn import metrics
import pandas as pd
from utils import DataGenerate, DataFeature
from ModelTraining.model_training import train_PANCDR
device = torch.device('cuda')
print(device)
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

if __name__ == '__main__':
    drug_feature,gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file)
    T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,None,T_Cancer_response_exp_file,dataset="TCGA")
    TX_drug_data_test,TX_gexpr_data_test,TY_test,Tcancer_type_test_list = DataFeature(T_data_idx,T_drug_feature,T_gexpr_feature,dataset="TCGA")

    TX_drug_feat_data_test = [item[0] for item in TX_drug_data_test]
    TX_drug_adj_data_test = [item[1] for item in TX_drug_data_test]
    TX_drug_feat_data_test = np.array(TX_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    TX_drug_adj_data_test = np.array(TX_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  

    X_drug_data,X_gexpr_data,Y,cancer_type_train_list = DataFeature(data_idx,drug_feature,gexpr_feature)
    
    X_drug_feat_data = [item[0] for item in X_drug_data]
    X_drug_adj_data = [item[1] for item in X_drug_data]
    X_drug_feat_data = np.array(X_drug_feat_data)
    X_drug_adj_data = np.array(X_drug_adj_data)

    train_data = [X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature]
    test_data = [TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test]
    
    df = pd.read_csv("tuned_hyperparameters/TCGA_CV_params.csv")
    best_params = eval(df.loc[(df["Model"]=="PANCDR") & (df["Classification"]=="T"),"Best_params"].values[0])
    csv_path = 'PANCDR_TCGA_100train_results.csv'
    results = []

    # ➤ 1. 기존 CSV 불러오기 (있으면)
    if os.path.exists(csv_path):
        result_df = pd.read_csv(csv_path)
        done_iters = set(result_df['Iteration'].dropna().astype(int).tolist())
    else:
        result_df = pd.DataFrame(columns=["Iteration", "Accuracy", "AUC", "F1", "Recall", "Precision"])
        done_iters = set()

    # ➤ 2. 반복 시작
    for iter in range(100):
        if iter in done_iters:
            print(f"Skipping iteration {iter}, already recorded.")
            continue

        weight_path = '../checkpoint/TCGA_PANCDR/%d_model.pt'%iter
        model = train_PANCDR(train_data,test_data, outer_fold=iter, project="PANCDR-TCGA")
        TCGA_metric, _ = model.train(best_params, weight_path)

        # ➤ 결과 dict + iteration 번호
        TCGA_metric["Iteration"] = iter
        print(f"iter {iter} - AUC: {TCGA_metric['AUC']:.4f}")

        # ➤ 결과 추가 및 저장
        result_df = pd.concat([result_df, pd.DataFrame([TCGA_metric])], ignore_index=True)
        result_df.to_csv(csv_path, index=False)
    
