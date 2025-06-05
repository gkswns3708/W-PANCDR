#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from utils import DataGenerate, DataFeature

#— 시드 고정 (필요 시) —#
random.seed(0)
np.random.seed(0)

#— 데이터 원본 파일 경로 설정 —#
DPATH = "../data"

# GDSC 데이터 파일
Drug_info_file           = os.path.join(DPATH, "GDSC", "GDSC_drug_binary.csv")
Cell_line_info_file      = os.path.join(DPATH, "GDSC", "Cell_Lines_Details.txt")
Drug_feature_file        = os.path.join(DPATH, "GDSC", "drug_graph_feat")
Cancer_response_exp_file = os.path.join(DPATH, "GDSC", "GDSC_binary_response_151.csv")
Gene_expression_file     = os.path.join(DPATH, "GDSC", "GDSC_expr_z_702.csv")
P_Gene_expression_file   = os.path.join(DPATH, "TCGA", "Pretrain_TCGA_expr_702_01A.csv")

# TCGA 데이터 파일
T_Drug_info_file           = os.path.join(DPATH, "TCGA", "TCGA_drug_new.csv")
T_Patient_info_file        = os.path.join(DPATH, "TCGA", "TCGA_type_new.txt")
T_Drug_feature_file        = os.path.join(DPATH, "TCGA", "drug_graph_feat")
T_Cancer_response_exp_file = os.path.join(DPATH, "TCGA", "TCGA_response_new.csv")
T_Gene_expression_file     = os.path.join(DPATH, "TCGA", "TCGA_expr_z_702.csv")

#— 미리처리 데이터를 저장할 디렉터리 및 파일 경로 —#
OUT_DIR = os.path.join(DPATH, "Preprocessed")
GDSC_DIR = os.path.join(OUT_DIR, "GDSC")
TCGA_DIR = os.path.join(OUT_DIR, "TCGA")

os.makedirs(GDSC_DIR, exist_ok=True)
os.makedirs(TCGA_DIR, exist_ok=True)


if __name__ == "__main__":
    #— 1) GDSC 데이터 생성 —#
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

    # 리스트 → NumPy 배열
    X_drug_feat_data = np.array([item[0] for item in X_drug_data])
    X_drug_adj_data  = np.array([item[1] for item in X_drug_data])
    X_gexpr_data     = np.array(X_gexpr_data)
    Y                = np.array(Y)
    t_gexpr_feature  = np.array(t_gexpr_feature)

    #— 2) TCGA 데이터 생성 —#
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

    # 리스트 → NumPy 배열
    TX_drug_feat_data_test = np.array([item[0] for item in TX_drug_data_test])
    TX_drug_adj_data_test  = np.array([item[1] for item in TX_drug_data_test])
    TX_gexpr_data_test     = np.array(TX_gexpr_data_test)
    TY_test                = np.array(TY_test)

    #— 3) GDSC용 개별 .npy 파일로 저장 —#
    np.save(os.path.join(GDSC_DIR, "X_drug_feat_data.npy"), X_drug_feat_data)
    np.save(os.path.join(GDSC_DIR, "X_drug_adj_data.npy"),  X_drug_adj_data)
    np.save(os.path.join(GDSC_DIR, "X_gexpr_data.npy"),      X_gexpr_data)
    np.save(os.path.join(GDSC_DIR, "Y.npy"),                 Y)
    np.save(os.path.join(GDSC_DIR, "t_gexpr_feature.npy"),   t_gexpr_feature)
    print(f"GDSC data saved to directory: {GDSC_DIR}")

    #— 4) TCGA용 개별 .npy 파일로 저장 —#
    np.save(os.path.join(TCGA_DIR, "TX_drug_feat_data_test.npy"), TX_drug_feat_data_test)
    np.save(os.path.join(TCGA_DIR, "TX_drug_adj_data_test.npy"),  TX_drug_adj_data_test)
    np.save(os.path.join(TCGA_DIR, "TX_gexpr_data_test.npy"),     TX_gexpr_data_test)
    np.save(os.path.join(TCGA_DIR, "TY_test.npy"),                TY_test)
    print(f"TCGA data saved to directory: {TCGA_DIR}")
