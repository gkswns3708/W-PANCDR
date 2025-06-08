import random,os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
from sklearn import metrics
import pandas as pd
from utils import DataGenerate, DataFeature, mkdirs, load_preprocessed_gdsc, load_preprocessed_tcga
import argparse
from ModelTraining.model_training import train_WANCDR
from config import Config
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="W-PANCDR Full CV")
    parser.add_argument(
        "--train_dataset",
        type=str,
        choices=["GDSC", "TCGA"],
        help="Mode of training: PANCDR or WANCDR",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="WANCDR",
        choices=["PANCDR", "WANCDR", "WANCDR_5Critic"],
        help="Mode of training: PANCDR or WANCDR",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="5FoldCV",
        choices=["5FoldCV", "CDRTCGA", "Nested", "None"],
        help="Cross-validation strategy: 5FoldCV or CDRTCGA or Nested",
    )
    parser.add_argument(
        "--test_metric",
        type=str,
        default="AUC",
        choices=["AUC", "W_distance", "Loss"],
        help="Optimization metric: Test AUC or W_distance or Loss",
    )
    parser.add_argument(
        "--use_preprocessed",
        action="store_true",
        help="미리 저장된 .npy 파일을 사용할지 여부",
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="사용할 GPU 디바이스 번호 (기본값: 0)",
    )
    args = parser.parse_args()
    
    config = Config(train_dataset=args.train_dataset, mode=args.mode, strategy=args.strategy, test_metric=args.test_metric)
    config = config.get_config()
    assert config is not None, "Configuration loading failed. Please check the config file."
    config['device'] = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
    ]
    
    mkdirs(config)
    
    
    if args.use_preprocessed:
        # 1) 미리 저장된 npy를 불러오기
        gdsc_data = load_preprocessed_gdsc(config['preprocessed']['gdsc_path'])
        tcga_data = load_preprocessed_tcga(config['preprocessed']['tcga_path'])

        X_drug_feat_data = torch.FloatTensor(gdsc_data["X_drug_feat_data"])
        X_drug_adj_data = torch.FloatTensor(gdsc_data["X_drug_adj_data"])
        X_gexpr_data = torch.FloatTensor(gdsc_data["X_gexpr_data"])
        Y = torch.FloatTensor(gdsc_data["Y"])
        t_gexpr_feature = torch.FloatTensor(gdsc_data["t_gexpr_feature"])

        TX_drug_feat_data_test = torch.FloatTensor(tcga_data["TX_drug_feat_data_test"])
        TX_drug_adj_data_test = torch.FloatTensor(tcga_data["TX_drug_adj_data_test"])
        TX_gexpr_data_test = torch.FloatTensor(tcga_data["TX_gexpr_data_test"])
        TY_test = torch.FloatTensor(tcga_data["TY_test"])

    else:
        # 2) 기존 방식대로 DataGenerate / DataFeature 사용
        DPATH = "../data"
        Drug_info_file = f"{DPATH}/GDSC/GDSC_drug_binary.csv"
        Cell_line_info_file = f"{DPATH}/GDSC/Cell_Lines_Details.txt"
        Drug_feature_file = f"{DPATH}/GDSC/drug_graph_feat"
        Cancer_response_exp_file = f"{DPATH}/GDSC/GDSC_binary_response_151.csv"
        Gene_expression_file = f"{DPATH}/GDSC/GDSC_expr_z_702.csv"
        P_Gene_expression_file = f"{DPATH}/TCGA/Pretrain_TCGA_expr_702_01A.csv"
        T_Drug_info_file = f"{DPATH}/TCGA/TCGA_drug_new.csv"
        T_Patient_info_file = f"{DPATH}/TCGA/TCGA_type_new.txt"
        T_Drug_feature_file = f"{DPATH}/TCGA/drug_graph_feat"
        T_Cancer_response_exp_file = f"{DPATH}/TCGA/TCGA_response_new.csv"
        T_Gene_expression_file = f"{DPATH}/TCGA/TCGA_expr_z_702.csv"

        # GDSC: DataGenerate → DataFeature
        drug_feature, gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(
            Drug_info_file,
            Cell_line_info_file,
            Drug_feature_file,
            Gene_expression_file,
            P_Gene_expression_file,
            Cancer_response_exp_file,
        )
        X_drug_data, X_gexpr_data, Y, cancer_type_train_list = DataFeature(
            data_idx, drug_feature, gexpr_feature
        )
        # 리스트 → np.array
        X_drug_feat_data = np.array([item[0] for item in X_drug_data])
        X_drug_adj_data = np.array([item[1] for item in X_drug_data])
        X_gexpr_data = np.array(X_gexpr_data)
        Y = np.array(Y)

        # TCGA: DataGenerate → DataFeature
        T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(
            T_Drug_info_file,
            T_Patient_info_file,
            T_Drug_feature_file,
            T_Gene_expression_file,
            None,
            T_Cancer_response_exp_file,
            dataset="TCGA",
        )
        TX_drug_data_test, TX_gexpr_data_test, TY_test, _ = DataFeature(
            T_data_idx, T_drug_feature, T_gexpr_feature, dataset="TCGA"
        )
        TX_drug_feat_data_test = torch.FloatTensor(
            np.array([item[0] for item in TX_drug_data_test])
        )
        TX_drug_adj_data_test = torch.FloatTensor(
            np.array([item[1] for item in TX_drug_data_test])
        )
        TX_gexpr_data_test = torch.FloatTensor(np.array(TX_gexpr_data_test))
        TY_test = torch.FloatTensor(np.array(TY_test))

    # --- train_WANCDR_full_cv 호출 ---
    train_data = [X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature]
    test_data  = [TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test]
    df = pd.read_csv("tuned_hyperparameters/TCGA_CV_params.csv")
    best_params = eval(df.loc[(df["Model"]=="WANCDR") & (df["Classification"]=="T"),"Best_params"].values[0])
    csv_path = config['csv']['TCGA_result_file_path']

    # ➤ 1. 기존 CSV 불러오기 (있으면)
    if os.path.exists(csv_path):
        result_df = pd.read_csv(csv_path)
        done_iters = set(result_df['Iteration'].dropna().astype(int).tolist())
    else:
        result_df = pd.DataFrame(columns=["Iteration", "Accuracy", "AUC", "F1", "Recall", "Precision"], dtype=object)
        done_iters = set()

    # ➤ 2. 반복 시작
    for iter in range(100):
        if iter in done_iters:
            print(f"Skipping iteration {iter}, already recorded.")
            continue

        weight_path = f'../checkpoint/TCGA_WANCDR/{iter}_model.pt'
        model = train_WANCDR(train_data, None, test_data, outer_fold=iter, config=config)
        best_metric, end_epoch = model.train(
            best_params,
            weight_path=os.path.join(config['train']['weight_path'], f'model_{iter}.pt')
        )

        # --- 여기에 추가 ---
        # 1) 현재 iteration 결과를 DataFrame으로 만듭니다.
        metrics = best_metric.copy()
        metrics['Iteration'] = iter
        merged_metrics = {**best_params, **metrics}
        df_current = pd.DataFrame([merged_metrics])

        # 2) 기존 CSV가 있으면 불러와서 concat, 없으면 그대로 사용
        if os.path.exists(csv_path):
            df_history = pd.read_csv(csv_path)
            df_all = pd.concat([df_history, df_current], ignore_index=True)
        else:
            df_all = df_current

        # 3) 다시 덮어쓰기
        df_all.to_csv(csv_path, index=False)
        print(f"Saved metrics for iteration {iter} to {csv_path}")