import pandas as pd
import numpy as np
import os
import time
import psutil
import gc
import json
import subprocess
import argparse
from collections import defaultdict
import rapidfuzz
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

import sys

def calculate_macro_f05(val_s1_ids, gt_dict, pred_dict):
    f05_scores = []
    precisions = []
    recalls = []
    for s1_id in val_s1_ids:
        t = set(gt_dict.get(s1_id, []))
        p = set(pred_dict.get(s1_id, []))
        if len(t) == 0 and len(p) == 0:
            f05, prec, rec = 1.0, 1.0, 1.0
        elif len(t) == 0 and len(p) > 0:
            f05, prec, rec = 0.0, 0.0, 0.0
        elif len(t) > 0 and len(p) == 0:
            f05, prec, rec = 0.0, 0.0, 0.0
        else:
            intersection = len(t.intersection(p))
            prec = intersection / len(p)
            rec = intersection / len(t)
            if prec + rec == 0:
                f05 = 0.0
            else:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        f05_scores.append(f05)
        precisions.append(prec)
        recalls.append(rec)
    return np.mean(f05_scores), np.mean(precisions), np.mean(recalls)

def extract_numbers(text):
    if not isinstance(text, str): return set()
    return set(re.findall(r'\d+', text))

def token_metrics(t1, t2):
    s1, s2 = set(str(t1).split()), set(str(t2).split())
    if not s1 or not s2: return 0.0, 0.0
    inter = len(s1.intersection(s2))
    jaccard = inter / len(s1.union(s2))
    contain = inter / min(len(s1), len(s2))
    return jaccard, contain

def fast_normalize(s):
    return str(s).lower().replace(r'[^a-z0-9]', ' ').strip()

def compute_features(df):
    flush_print(f"[{time.time()-t0:.1f}s] Computing RapidFuzz Name Features...")
    df['n1'] = df['name_s1'].fillna("").astype(str)
    df['n2'] = df['name_s23'].fillna("").astype(str)
    
    df['name_ratio'] = [rapidfuzz.fuzz.ratio(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_wratio'] = [rapidfuzz.fuzz.WRatio(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_partial'] = [rapidfuzz.fuzz.partial_ratio(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_jaro'] = [rapidfuzz.distance.JaroWinkler.normalized_similarity(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_lev'] = [rapidfuzz.distance.Levenshtein.normalized_similarity(a, b) for a, b in zip(df['n1'], df['n2'])]
    
    flush_print(f"[{time.time()-t0:.1f}s] Computing Name Token Features...")
    metrics = [token_metrics(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_jaccard'] = [x[0] for x in metrics]
    df['name_contain'] = [x[1] for x in metrics]
    
    df['name_len_diff'] = np.abs(df['n1'].str.len() - df['n2'].str.len())
    df['name_len_ratio'] = df['n1'].str.len() / (df['n2'].str.len() + 1e-5)
    df['exact_name'] = (df['n1'] == df['n2']) & (df['n1'] != "")
    
    flush_print(f"[{time.time()-t0:.1f}s] Computing RapidFuzz Address Features...")
    df['a1'] = df['address_s1'].fillna("").astype(str)
    df['a2'] = df['address_s23'].fillna("").astype(str)
    
    df['addr_ratio'] = [rapidfuzz.fuzz.ratio(a, b) for a, b in zip(df['a1'], df['a2'])]
    df['addr_wratio'] = [rapidfuzz.fuzz.WRatio(a, b) for a, b in zip(df['a1'], df['a2'])]
    
    flush_print(f"[{time.time()-t0:.1f}s] Computing Address Token Features...")
    metrics_a = [token_metrics(a, b) for a, b in zip(df['a1'], df['a2'])]
    df['addr_jaccard'] = [x[0] for x in metrics_a]
    df['addr_contain'] = [x[1] for x in metrics_a]
    
    df['exact_addr'] = (df['a1'] == df['a2']) & (df['a1'] != "")
    df['addr_missing'] = (df['a1'] == "") | (df['a2'] == "")
    
    flush_print(f"[{time.time()-t0:.1f}s] Computing Number Agreement...")
    nums1 = [extract_numbers(a) for a in df['a1']]
    nums2 = [extract_numbers(a) for a in df['a2']]
    num_agree = []
    for n1, n2 in zip(nums1, nums2):
        if not n1 or not n2: num_agree.append(-1.0)
        else: num_agree.append(float(len(n1.intersection(n2)) > 0))
    df['addr_num_agree'] = num_agree
    
    df['country_match'] = (df['country_s1'] == df['country_s23']).astype(float)
    
    features = [
        'name_ratio', 'name_wratio', 'name_partial', 'name_jaro', 'name_lev',
        'name_jaccard', 'name_contain', 'name_len_diff', 'name_len_ratio', 'exact_name',
        'addr_ratio', 'addr_wratio', 'addr_jaccard', 'addr_contain', 'exact_addr',
        'addr_missing', 'addr_num_agree', 'country_match'
    ]
    return df, features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on 1,000 subset instead of 10,000")
    args = parser.parse_args()
    
    global t0
    t0 = time.time()
    
    n_samples = 1000 if args.smoke else 10000
    suffix = "smoke" if args.smoke else "10000"
    cand_file = f"code/business_entity_resolution/experiments/candidates_{suffix}.parquet"
    
    flush_print("============================================================")
    flush_print(f"EXPERIMENT 4: FEATURE ENGINEERING & LIGHTGBM ({suffix})")
    flush_print("============================================================")
    
    if not os.path.exists(cand_file):
        flush_print(f"\n[0/6] Candidates not found. Running Exp4 Candidate Generator...")
        cmd = [sys.executable, "code/business_entity_resolution/src/exp4_generate_candidates.py"]
        if args.smoke: cmd.append("--smoke")
        subprocess.run(cmd, check=True)
        
    flush_print(f"\n[1/6] Loading candidate pairs")
    df_pairs = pd.read_parquet(cand_file)
    flush_print(f"Loaded {len(df_pairs)} pairs. RAM: {memory_usage():.1f} MB")
    
    unique_s1 = set(df_pairs['source1_entity_id'])
    unique_s23 = set(df_pairs['matched_entity_id'])
    
    base_dir = "dataset/student_resource/dataset/train"
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', dtype=str)
    df_s1 = df_s1[df_s1['entity_id'].isin(unique_s1)]
    df_s1.rename(columns={'business_name':'name_s1', 'business_address':'address_s1', 'country':'country_s1'}, inplace=True)
    
    s23_rows = []
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        if not os.path.exists(filepath): continue
        chunk_iter = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str, chunksize=2000000)
        for chunk in chunk_iter:
            c = chunk[chunk['entity_id'].isin(unique_s23)]
            s23_rows.append(c)
    df_s23 = pd.concat(s23_rows, ignore_index=True)
    df_s23.rename(columns={'business_name':'name_s23', 'business_address':'address_s23', 'country':'country_s23'}, inplace=True)
    
    # Pre-normalize for rapidfuzz
    df_s1['name_s1'] = df_s1['name_s1'].apply(fast_normalize)
    df_s1['address_s1'] = df_s1['address_s1'].apply(fast_normalize)
    df_s23['name_s23'] = df_s23['name_s23'].apply(fast_normalize)
    df_s23['address_s23'] = df_s23['address_s23'].apply(fast_normalize)
    
    df_pairs = df_pairs.merge(df_s1[['entity_id', 'name_s1', 'address_s1', 'country_s1']], left_on='source1_entity_id', right_on='entity_id', how='left')
    df_pairs = df_pairs.merge(df_s23[['entity_id', 'name_s23', 'address_s23', 'country_s23']], left_on='matched_entity_id', right_on='entity_id', how='left')
    
    del df_s1, df_s23, s23_rows
    gc.collect()
    
    flush_print(f"\n[2/6] Building features & labels")
    df_gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)
    df_gt = df_gt[df_gt['source1_entity_id'].isin(unique_s1)]
    
    gt_dict = {}
    for _, row in df_gt.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or matches == 'nan' or matches.strip() == '':
            gt_dict[s1] = set()
        else:
            gt_dict[s1] = set([x.strip() for x in matches.split(',')])
            
    labels = []
    for s1, s2 in zip(df_pairs['source1_entity_id'], df_pairs['matched_entity_id']):
        labels.append(1 if s2 in gt_dict.get(s1, set()) else 0)
    df_pairs['label'] = labels
    
    df_pairs, feature_cols = compute_features(df_pairs)
    flush_print(f"Features created: {len(feature_cols)}. RAM: {memory_usage():.1f} MB")
    
    flush_print(f"\n[3/6] Creating train/validation split (S1-level)")
    all_s1_list = list(unique_s1)
    train_s1, val_s1 = train_test_split(all_s1_list, test_size=0.2, random_state=42)
    
    train_mask = df_pairs['source1_entity_id'].isin(train_s1)
    val_mask = df_pairs['source1_entity_id'].isin(val_s1)
    
    X_train = df_pairs.loc[train_mask, feature_cols].astype(np.float32)
    y_train = df_pairs.loc[train_mask, 'label'].astype(np.float32)
    X_val = df_pairs.loc[val_mask, feature_cols].astype(np.float32)
    y_val = df_pairs.loc[val_mask, 'label'].astype(np.float32)
    
    flush_print(f"\n[4/6] Training LightGBM")
    t_train = time.time()
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Fix for callbacks in LightGBM 3/4
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_eval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    
    train_time = time.time() - t_train
    flush_print(f"Training time: {train_time:.1f} sec")
    
    flush_print(f"\n[5/6] Running validation inference")
    t_inf = time.time()
    val_preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    inf_time = time.time() - t_inf
    
    df_val = df_pairs[val_mask].copy()
    df_val['pred_score'] = val_preds
    
    flush_print(f"\n[6/6] Threshold optimization")
    best_thresh = 0.5
    best_f05 = -1.0
    
    thresholds = np.arange(0.1, 0.96, 0.05)
    for thresh in thresholds:
        pred_dict = defaultdict(list)
        df_val_pos = df_val[df_val['pred_score'] >= thresh]
        for s1, s2 in zip(df_val_pos['source1_entity_id'], df_val_pos['matched_entity_id']):
            pred_dict[s1].append(s2)
            
        f05, _, _ = calculate_macro_f05(val_s1, gt_dict, pred_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    # Calculate final metrics at best threshold
    pred_dict = defaultdict(list)
    df_val_pos = df_val[df_val['pred_score'] >= best_thresh]
    for s1, s2 in zip(df_val_pos['source1_entity_id'], df_val_pos['matched_entity_id']):
        pred_dict[s1].append(s2)
        
    final_f05, final_prec, final_rec = calculate_macro_f05(val_s1, gt_dict, pred_dict)
    
    # Candidate recall on val
    cand_dict = defaultdict(list)
    for s1, s2 in zip(df_val['source1_entity_id'], df_val['matched_entity_id']):
        cand_dict[s1].append(s2)
    cand_recall = 0
    total_true = 0
    total_found = 0
    for s1 in val_s1:
        t = set(gt_dict.get(s1, []))
        c = set(cand_dict.get(s1, []))
        if len(t) > 0:
            total_true += len(t)
            total_found += len(t.intersection(c))
    cand_recall = total_found / total_true if total_true > 0 else 1.0
    
    # Singleton stats
    singletons_val = [s1 for s1 in val_s1 if len(gt_dict.get(s1, [])) == 0]
    singletons_correct = sum(1 for s1 in singletons_val if len(pred_dict.get(s1, [])) == 0)
    
    flush_print(f"\n==================================================")
    flush_print(f"RESULTS ({suffix})")
    flush_print(f"==================================================")
    flush_print(f"Best Threshold: {best_thresh:.2f}")
    flush_print(f"Macro F0.5:     {final_f05:.4f}")
    flush_print(f"Precision:      {final_prec:.4f}")
    flush_print(f"Recall:         {final_rec:.4f}")
    flush_print(f"Cand Recall:    {cand_recall:.4f}")
    
    # Outputs
    out_dir = "code/business_entity_resolution/experiments"
    os.makedirs(out_dir, exist_ok=True)
    
    df_val_out = df_val[['source1_entity_id', 'matched_entity_id', 'label', 'pred_score']]
    df_val_out.to_parquet(f"{out_dir}/exp4_validation_predictions.parquet", index=False)
    
    imp = pd.DataFrame({'feature': feature_cols, 'importance': gbm.feature_importance(importance_type='gain')})
    imp = imp.sort_values('importance', ascending=False)
    imp.to_csv(f"{out_dir}/exp4_feature_importance.csv", index=False)
    
    model_path = f"{out_dir}/lightgbm_model.txt"
    gbm.save_model(model_path)
    
    thresh_path = f"{out_dir}/best_threshold.txt"
    with open(thresh_path, "w") as f:
        f.write(str(best_thresh))
    
    res = {
        "n_s1_entities": len(unique_s1),
        "n_candidate_pairs": len(df_pairs),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int(len(y_train) - y_train.sum()),
        "n_features": len(feature_cols),
        "train_time_sec": train_time,
        "inference_time_sec": inf_time,
        "best_threshold": best_thresh,
        "macro_f05": final_f05,
        "precision": final_prec,
        "recall": final_rec,
        "cand_recall": cand_recall,
        "n_predicted_matches": len(df_val_pos),
        "singletons_total": len(singletons_val),
        "singletons_correct": singletons_correct,
        "top_features": imp.head(10).to_dict('records'),
        "peak_ram_mb": memory_usage()
    }
    with open(f"{out_dir}/exp4_lightgbm_results.json", "w") as f:
        json.dump(res, f, indent=2)
        
    flush_print(f"\nModel saved to {model_path}")
    flush_print(f"Threshold saved to {thresh_path}")
    flush_print("\nDONE.")
