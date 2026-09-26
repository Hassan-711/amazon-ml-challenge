import pandas as pd
import numpy as np
import os
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import gc
import json
import scipy.sparse as sp

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2 # MB

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join(c if c.isalnum() else ' ' for c in text)
    text = ' '.join(text.split())
    return text

def calculate_candidate_recall(val_s1_ids, gt_dict, candidates_dict):
    total_true_links = 0
    total_found_links = 0
    
    for s1_id in val_s1_ids:
        t = set(gt_dict.get(s1_id, []))
        c = set(candidates_dict.get(s1_id, []))
        
        if len(t) > 0:
            total_true_links += len(t)
            total_found_links += len(t.intersection(c))
            
    if total_true_links == 0:
        return 1.0
    return total_found_links / total_true_links

import sparse_dot_topn

def get_top_k_sparse(matrix, K):
    top_k_indices = []
    for i in range(matrix.shape[0]):
        start = matrix.indptr[i]
        end = matrix.indptr[i+1]
        data = matrix.data[start:end]
        indices = matrix.indices[start:end]
        
        if len(data) == 0:
            top_k_indices.append([])
            continue
            
        if len(data) <= K:
            sort_idx = np.argsort(-data)
            top_k_indices.append(list(indices[sort_idx]))
        else:
            part_idx = np.argpartition(-data, K - 1)[:K]
            top_k = part_idx[np.argsort(-data[part_idx])]
            top_k_indices.append(list(indices[top_k]))
    return top_k_indices

def run_experiment(val_s1, gt_dict, name_to_s23_ids_exact):
    base_dir = "dataset/student_resource/dataset/train"
    results = {}
    
    subset_val_s1 = val_s1.sample(n=min(30000, len(val_s1)), random_state=42)
    subset_s1_ids = subset_val_s1['entity_id'].tolist()
    
    configs = [
        {"ngram_range": (2,5)},
        {"ngram_range": (3,5)},
        {"ngram_range": (3,6)}
    ]
    k_values = [10, 20, 50]
    
    best_config = None
    best_score = -1 
    
    needed_countries = subset_val_s1['country'].unique()
    
    for config in configs:
        config_name = f"ngram_{config['ngram_range'][0]}_{config['ngram_range'][1]}"
        print(f"Testing {config_name}...")
        
        # We store union with exact matches for evaluation
        candidates_by_k = {k: defaultdict(list) for k in k_values}
        
        # Add exact matches first
        for _, row in subset_val_s1.iterrows():
            if row['norm_name'] in name_to_s23_ids_exact:
                for k in k_values:
                    candidates_by_k[k][row['entity_id']].extend(name_to_s23_ids_exact[row['norm_name']])
        
        for country in needed_countries:
            print(f"  Processing country: {country}")
            s2s3_names = []
            s2s3_ids = []
            
            for source in ["train_source2.tsv", "train_source3.tsv"]:
                filepath = os.path.join(base_dir, source)
                for chunk in pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'country'], dtype=str, chunksize=1000000):
                    c = chunk[chunk['country'] == country]
                    c_names = c['business_name'].apply(normalize_text)
                    valid = c_names != ""
                    s2s3_names.extend(c_names[valid].tolist())
                    s2s3_ids.extend(c['entity_id'][valid].tolist())
                    
            s1_subset_c = subset_val_s1[subset_val_s1['country'] == country]
            if len(s1_subset_c) == 0:
                continue
            
            s1_names = s1_subset_c['norm_name'].tolist()
            s1_ids = s1_subset_c['entity_id'].tolist()
            
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=config['ngram_range'], min_df=2, max_df=0.8)
            if len(s2s3_names) > 0:
                X_s2s3 = vectorizer.fit_transform(s2s3_names)
                X_s1 = vectorizer.transform(s1_names)
                
                batch_size = 5000
                for i in range(0, X_s1.shape[0], batch_size):
                    X_s1_batch = X_s1[i:i+batch_size]
                    max_k = max(k_values)
                    
                    # Use sparse_dot_topn to prevent memory blowout
                    sim = sparse_dot_topn.sp_matmul_topn(X_s1_batch, X_s2s3.T, top_n=max_k)
                    
                    top_k_res = get_top_k_sparse(sim, max_k)

                    
                    for j, top_indices in enumerate(top_k_res):
                        s1_idx = i + j
                        s1_id = s1_ids[s1_idx]
                        
                        for k in k_values:
                            k_indices = top_indices[:k]
                            candidates_by_k[k][s1_id].extend([s2s3_ids[idx] for idx in k_indices])
            
            del s2s3_names
            del s2s3_ids
            gc.collect()
            
        for k in k_values:
            # Deduplicate per s1_id
            for sid in subset_s1_ids:
                candidates_by_k[k][sid] = list(set(candidates_by_k[k][sid]))
                
            cand_recall = calculate_candidate_recall(subset_s1_ids, gt_dict, candidates_by_k[k])
            cand_counts = [len(candidates_by_k[k].get(sid, [])) for sid in subset_s1_ids]
            avg_cand = np.mean(cand_counts)
            
            print(f"    K={k}: Recall={cand_recall:.4f}, Avg Cands={avg_cand:.2f}")
            
            if "results" not in config:
                config["results"] = {}
            config["results"][k] = {
                "recall": float(cand_recall),
                "avg_cand": float(avg_cand),
                "p95_cand": float(np.percentile(cand_counts, 95)),
                "p99_cand": float(np.percentile(cand_counts, 99))
            }
            
            score = cand_recall - 0.0005 * avg_cand
            if score > best_score:
                best_score = score
                best_config = {"ngram_range": config["ngram_range"], "k": k}
                
    print(f"Best configuration found: {best_config}")
    
    with open("code/business_entity_resolution/experiments/exp2_results.json", "w") as f:
        json.dump(configs, f, indent=2)
        
    return best_config

def main():
    start_time = time.time()
    base_dir = "dataset/student_resource/dataset/train"
    
    print(f"[{time.time()-start_time:.1f}s] Loading data...")
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', usecols=['entity_id', 'business_name', 'country'], dtype=str)
    df_gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)
    
    train_s1, val_s1 = train_test_split(df_s1, test_size=0.1, random_state=42, stratify=df_s1['country'])
    val_s1['norm_name'] = val_s1['business_name'].apply(normalize_text)
    val_s1_ids = val_s1['entity_id'].tolist()
    
    df_gt_val = df_gt[df_gt['source1_entity_id'].isin(val_s1_ids)]
    gt_dict = {}
    for _, row in df_gt_val.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or matches == 'nan' or matches.strip() == '':
            gt_dict[s1] = []
        else:
            gt_dict[s1] = [x.strip() for x in matches.split(',')]
            
    del df_gt_val, df_gt, df_s1, train_s1
    gc.collect()
    
    print(f"[{time.time()-start_time:.1f}s] Building exact match index...")
    name_to_s23_ids_exact = defaultdict(list)
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        for chunk in pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name'], dtype=str, chunksize=2000000):
            chunk['norm_name'] = chunk['business_name'].apply(normalize_text)
            chunk = chunk[chunk['norm_name'] != ""]
            for _, row in chunk.iterrows():
                name_to_s23_ids_exact[row['norm_name']].append(row['entity_id'])
    
    print(f"[{time.time()-start_time:.1f}s] Running Benchmark on subset...")
    best_config = run_experiment(val_s1, gt_dict, name_to_s23_ids_exact)
    
    print(f"\n[{time.time()-start_time:.1f}s] Running best config {best_config} on full validation set ({len(val_s1)} entities)...")
    
    final_candidates = defaultdict(list)
    needed_countries = val_s1['country'].unique()
    
    for _, row in val_s1.iterrows():
        if row['norm_name'] in name_to_s23_ids_exact:
            final_candidates[row['entity_id']].extend(name_to_s23_ids_exact[row['norm_name']])
            
    del name_to_s23_ids_exact
    gc.collect()
    
    for country in needed_countries:
        print(f"Processing full country: {country}")
        s2s3_names = []
        s2s3_ids = []
        
        for source in ["train_source2.tsv", "train_source3.tsv"]:
            filepath = os.path.join(base_dir, source)
            for chunk in pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'country'], dtype=str, chunksize=1500000):
                c = chunk[chunk['country'] == country]
                c_names = c['business_name'].apply(normalize_text)
                valid = c_names != ""
                s2s3_names.extend(c_names[valid].tolist())
                s2s3_ids.extend(c['entity_id'][valid].tolist())
                
        s1_subset_c = val_s1[val_s1['country'] == country]
        if len(s1_subset_c) == 0:
            continue
            
        s1_names = s1_subset_c['norm_name'].tolist()
        s1_ids = s1_subset_c['entity_id'].tolist()
        
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=best_config['ngram_range'], min_df=2, max_df=0.8)
        if len(s2s3_names) > 0:
            X_s2s3 = vectorizer.fit_transform(s2s3_names)
            X_s1 = vectorizer.transform(s1_names)
            
            batch_size = 5000
            for i in range(0, X_s1.shape[0], batch_size):
                X_s1_batch = X_s1[i:i+batch_size]
                
                sim = sparse_dot_topn.sp_matmul_topn(X_s1_batch, X_s2s3.T, top_n=best_config['k'])
                top_k_res = get_top_k_sparse(sim, best_config['k'])
                
                for j, top_indices in enumerate(top_k_res):
                    s1_idx = i + j
                    s1_id = s1_ids[s1_idx]
                    final_candidates[s1_id].extend([s2s3_ids[idx] for idx in top_indices])
                    
        del s2s3_names
        del s2s3_ids
        gc.collect()
        
    print(f"[{time.time()-start_time:.1f}s] Computing final metrics...")
    
    # Deduplicate
    for sid in val_s1_ids:
        final_candidates[sid] = list(set(final_candidates.get(sid, [])))
        
    cand_recall = calculate_candidate_recall(val_s1_ids, gt_dict, final_candidates)
    
    cand_counts = [len(final_candidates.get(sid, [])) for sid in val_s1_ids]
    avg_cand = np.mean(cand_counts)
    
    print("\n" + "="*50)
    print(f"EXPERIMENT 2: TF-IDF (Union with Exact) RESULTS")
    print("="*50)
    print(f"Best Config:              {best_config}")
    print(f"Candidate Recall:         {cand_recall:.4f}")
    print(f"Avg Candidates per S1:    {avg_cand:.2f}")
    print(f"95th Pct Candidates:      {np.percentile(cand_counts, 95):.2f}")
    print(f"99th Pct Candidates:      {np.percentile(cand_counts, 99):.2f}")
    print(f"Total Candidate Pairs:    {sum(cand_counts)}")
    print("-" * 50)
    print(f"Runtime:                  {time.time()-start_time:.1f} seconds")
    print(f"Peak Memory:              {memory_usage():.1f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
