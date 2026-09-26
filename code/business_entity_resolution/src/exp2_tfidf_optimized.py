import pandas as pd
import numpy as np
import os
import time
import psutil
from collections import defaultdict
import gc
import json
import scipy.sparse as sp
import sys
import re
import threading
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2 # MB

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

class Heartbeat(threading.Thread):
    def __init__(self, message, interval=30):
        super().__init__()
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self.daemon = True
        self.start_time = time.time()
        
    def run(self):
        while not self._stop_event.wait(self.interval):
            elapsed = int(time.time() - self.start_time)
            flush_print(f"\n[HEARTBEAT] {self.message}")
            flush_print(f"Still running...")
            flush_print(f"Elapsed: {elapsed} sec")
            
    def stop(self):
        self._stop_event.set()

def fast_normalize_series(series):
    # Vectorized pandas/numpy string operations
    return series.fillna("").astype(str).str.lower().str.replace(r'[^a-z0-9]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip().tolist()

def get_top_k_sparse(matrix, K):
    top_k_indices = []
    
    if not isinstance(matrix, sp.csr_matrix):
        matrix = matrix.tocsr()
        
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
            top_k_indices.append(indices[sort_idx].tolist())
        else:
            part_idx = np.argpartition(-data, K - 1)[:K]
            top_k = part_idx[np.argsort(-data[part_idx])]
            top_k_indices.append(indices[top_k].tolist())
            
    return top_k_indices

def calculate_recall(val_s1_ids, gt_dict, candidates_dict):
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

def main():
    flush_print("============================================================")
    flush_print("EXPERIMENT 2 OPTIMIZED")
    flush_print("============================================================")
    
    start_time = time.time()
    base_dir = "dataset/student_resource/dataset/train"
    
    flush_print("\n[1/2] Loading validation S1 (Subset: 10,000)")
    t0 = time.time()
    
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', usecols=['entity_id', 'business_name', 'country'], dtype=str)
    _, val_s1_full = train_test_split(df_s1, test_size=0.1, random_state=42, stratify=df_s1['country'])
    
    val_s1 = val_s1_full.sample(n=10000, random_state=42).copy()
    val_s1['norm_name'] = fast_normalize_series(val_s1['business_name'])
    val_s1_ids = val_s1['entity_id'].tolist()
    
    df_gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)
    df_gt_val = df_gt[df_gt['source1_entity_id'].isin(set(val_s1_ids))]
    
    gt_dict = {}
    for _, row in df_gt_val.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or matches == 'nan' or matches.strip() == '':
            gt_dict[s1] = []
        else:
            gt_dict[s1] = [x.strip() for x in matches.split(',')]
            
    del df_s1, val_s1_full, df_gt, df_gt_val
    gc.collect()
    
    flush_print(f"Loaded: 10,000")
    flush_print(f"Time: {time.time()-t0:.1f} sec")
    flush_print(f"RAM: {memory_usage():.1f} MB")
    
    needed_countries = val_s1['country'].unique()
    
    s1_exact_name_map = defaultdict(list)
    for s1_id, norm_name in zip(val_s1['entity_id'], val_s1['norm_name']):
        if norm_name:
            s1_exact_name_map[norm_name].append(s1_id)
            
    flush_print("\n[2/2] One-pass loading S2/S3 & Exact Match Blocking")
    t0 = time.time()
    
    s2s3_names_by_country = defaultdict(list)
    s2s3_ids_by_country = defaultdict(list)
    exact_matches_by_s1 = defaultdict(list)
    
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        if not os.path.exists(filepath):
            flush_print(f"Warning: {filepath} not found!")
            continue
            
        hb = Heartbeat(f"Stage: Loading {source}", interval=30)
        hb.start()
        
        chunk_iter = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'country'], dtype=str, chunksize=2000000)
        for i, chunk in enumerate(chunk_iter):
            chunk = chunk[chunk['country'].isin(needed_countries)]
            if len(chunk) == 0:
                continue
                
            norm_names = fast_normalize_series(chunk['business_name'])
            chunk['norm_name'] = norm_names
            valid_mask = chunk['norm_name'] != ""
            chunk = chunk[valid_mask]
            
            for country, grp in chunk.groupby('country'):
                s2s3_names_by_country[country].extend(grp['norm_name'].tolist())
                s2s3_ids_by_country[country].extend(grp['entity_id'].tolist())
                
            for eid, name in zip(chunk['entity_id'], chunk['norm_name']):
                if name in s1_exact_name_map:
                    for s1_id in s1_exact_name_map[name]:
                        exact_matches_by_s1[s1_id].append(eid)
                        
            del chunk, norm_names, valid_mask
            gc.collect()
            
        hb.stop()
            
    flush_print(f"Total time loading S2/S3: {time.time()-t0:.1f} sec")
    flush_print(f"RAM: {memory_usage():.1f} MB")
    
    analyzer = "char_wb"
    ngram_range = (3,5)
    K = 20
    
    candidates_by_s1 = defaultdict(list)
    
    # Process country by country
    for country in needed_countries:
        s1_subset = val_s1[val_s1['country'] == country]
        if len(s1_subset) == 0:
            continue
            
        s1_names = s1_subset['norm_name'].tolist()
        s1_ids = s1_subset['entity_id'].tolist()
        
        s2s3_names = s2s3_names_by_country.get(country, [])
        s2s3_ids = s2s3_ids_by_country.get(country, [])
        
        flush_print(f"\n============================================================")
        flush_print(f"Processing Country: {country}")
        flush_print(f"============================================================")
        flush_print(f"S1 records: {len(s1_names)}")
        flush_print(f"S2/S3 records: {len(s2s3_names)}")
        
        if len(s2s3_names) == 0:
            continue
            
        flush_print(f"\n[+] Building TF-IDF")
        t_tfidf = time.time()
        flush_print(f"Analyzer: {analyzer}")
        flush_print(f"N-gram: {ngram_range}")
        
        hb = Heartbeat(f"Stage: TF-IDF Building | Country: {country}", interval=30)
        hb.start()
        
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2, max_df=0.8)
        X_s2s3 = vectorizer.fit_transform(s2s3_names)
        X_s1 = vectorizer.transform(s1_names)
        
        hb.stop()
        
        flush_print(f"Matrix shape S2/S3: {X_s2s3.shape}")
        flush_print(f"Matrix nnz S2/S3: {X_s2s3.nnz}")
        flush_print(f"Time: {time.time() - t_tfidf:.1f} sec")
        flush_print(f"RAM: {memory_usage():.1f} MB")
        
        flush_print(f"\n[+] Sparse retrieval")
        t_retrieval = time.time()
        
        batch_size = 1000
        n_batches = int(np.ceil(X_s1.shape[0] / batch_size))
        
        s1_processed = 0
        total_s1 = X_s1.shape[0]
        
        flush_print(f"Converting matrix to CSR (this may take several minutes)...")
        hb = Heartbeat(f"Stage: CSR Transpose | Country: {country}", interval=30)
        hb.start()
        X_s2s3_T = X_s2s3.T.tocsr()
        hb.stop()
        flush_print(f"Matrix conversion done in {time.time()-t_retrieval:.1f} sec")

        
        for i in range(n_batches):
            b_start = time.time()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_s1)
            X_s1_batch = X_s1[start_idx:end_idx]
            
            hb = Heartbeat(f"Stage: sparse retrieval | Country: {country} | Batch: {i+1}/{n_batches}", interval=30)
            hb.start()
            
            sim = sparse_dot_topn.sp_matmul_topn(X_s1_batch, X_s2s3_T, top_n=K, n_threads=-1)
            top_k_indices = get_top_k_sparse(sim, K)
            
            hb.stop()
            
            batch_candidates = 0
            for j, top_idx in enumerate(top_k_indices):
                s1_id = s1_ids[start_idx + j]
                mapped_ids = [s2s3_ids[idx] for idx in top_idx]
                candidates_by_s1[s1_id].extend(mapped_ids)
                batch_candidates += len(mapped_ids)
                
            s1_processed += (end_idx - start_idx)
            b_time = time.time() - b_start
            speed = (end_idx - start_idx) / max(b_time, 0.001)
            elapsed = time.time() - t_retrieval
            
            flush_print(f"\nBatch {i+1}/{n_batches}")
            flush_print(f"S1 processed: {s1_processed}/{total_s1}")
            flush_print(f"Candidates: {batch_candidates}")
            flush_print(f"Speed: {speed:.1f} S1/sec")
            flush_print(f"Elapsed: {elapsed:.1f} sec")
            flush_print(f"RAM: {memory_usage():.1f} MB")
            
        del X_s1, X_s2s3, X_s2s3_T, vectorizer, s1_names, s1_ids
        s2s3_names_by_country.pop(country, None)
        s2s3_ids_by_country.pop(country, None)
        gc.collect()
        
    flush_print("\n==================================================")
    flush_print("EVALUATION")
    flush_print("==================================================")
    
    candidates_k10 = defaultdict(list)
    candidates_k20 = defaultdict(list)
    union_k10 = defaultdict(list)
    union_k20 = defaultdict(list)
    
    for s1_id in val_s1_ids:
        cands = candidates_by_s1.get(s1_id, [])
        exact = exact_matches_by_s1.get(s1_id, [])
        
        c10 = cands[:10]
        c20 = cands[:20]
        
        candidates_k10[s1_id] = list(set(c10))
        candidates_k20[s1_id] = list(set(c20))
        
        union_k10[s1_id] = list(set(c10 + exact))
        union_k20[s1_id] = list(set(c20 + exact))
        
    def eval_dict(cand_dict):
        counts = [len(cand_dict[s]) for s in val_s1_ids]
        recall = calculate_recall(val_s1_ids, gt_dict, cand_dict)
        avg = np.mean(counts)
        p95 = np.percentile(counts, 95)
        p99 = np.percentile(counts, 99)
        return {
            "recall": float(recall),
            "avg_cand": float(avg),
            "p95_cand": float(p95),
            "p99_cand": float(p99)
        }
        
    res_k10 = eval_dict(candidates_k10)
    res_k20 = eval_dict(candidates_k20)
    res_u10 = eval_dict(union_k10)
    res_u20 = eval_dict(union_k20)
    
    flush_print(f"\nK=10:")
    flush_print(f"- Candidate Recall: {res_k10['recall']*100:.2f}%")
    flush_print(f"- Average candidates/S1: {res_k10['avg_cand']:.2f}")
    flush_print(f"- P95 candidates/S1: {res_k10['p95_cand']:.1f}")
    flush_print(f"- P99 candidates/S1: {res_k10['p99_cand']:.1f}")
    
    flush_print(f"\nK=20:")
    flush_print(f"- Candidate Recall: {res_k20['recall']*100:.2f}%")
    flush_print(f"- Average candidates/S1: {res_k20['avg_cand']:.2f}")
    flush_print(f"- P95 candidates/S1: {res_k20['p95_cand']:.1f}")
    flush_print(f"- P99 candidates/S1: {res_k20['p99_cand']:.1f}")
    
    flush_print(f"\nExperiment 1 recall:")
    flush_print(f"21.84%")
    flush_print(f"\nExperiment 2 recall:")
    flush_print(f"{res_k20['recall']*100:.2f}%")
    flush_print(f"\nUnion recall:")
    flush_print(f"{res_u20['recall']*100:.2f}%")
    
    out_file = "code/business_entity_resolution/experiments/exp2_optimized_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    final_results = {
        "configuration": {
            "analyzer": analyzer,
            "ngram_range": ngram_range,
            "K": K
        },
        "validation_size": len(val_s1_ids),
        "metrics": {
            "K10": res_k10,
            "K20": res_k20,
            "Union_K10": res_u10,
            "Union_K20": res_u20
        },
        "runtime_seconds": time.time() - start_time,
        "peak_ram_mb": memory_usage()
    }
    
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
        
    flush_print(f"\nResults saved to {out_file}")
    flush_print("\nDONE.")

if __name__ == "__main__":
    main()
