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
    flush_print("EXPERIMENT 3: SMOKE TEST (ADDRESS ONLY, 1,000 S1)")
    flush_print("============================================================")
    
    start_time = time.time()
    base_dir = "dataset/student_resource/dataset/train"
    
    flush_print("\n[1/2] Loading validation S1 (Subset: 1,000)")
    t0 = time.time()
    
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', usecols=['entity_id', 'business_address', 'country'], dtype=str)
    _, val_s1_full = train_test_split(df_s1, test_size=0.1, random_state=42, stratify=df_s1['country'])
    
    val_s1 = val_s1_full.sample(n=1000, random_state=42).copy()
    val_s1['norm_address'] = fast_normalize_series(val_s1['business_address'])
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
    
    needed_countries = val_s1['country'].unique()
    
    flush_print("\n[2/2] One-pass loading S2/S3 (Addresses Only)")
    t0 = time.time()
    
    s2s3_addresses_by_country = defaultdict(list)
    s2s3_ids_by_country = defaultdict(list)
    
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        if not os.path.exists(filepath):
            continue
            
        hb = Heartbeat(f"Stage: Loading {source}", interval=30)
        hb.start()
        
        chunk_iter = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_address', 'country'], dtype=str, chunksize=2000000)
        for chunk in chunk_iter:
            chunk = chunk[chunk['country'].isin(needed_countries)]
            if len(chunk) == 0: continue
            
            chunk['norm_address'] = fast_normalize_series(chunk['business_address'])
            chunk = chunk[chunk['norm_address'] != ""]
            
            for country, grp in chunk.groupby('country'):
                s2s3_addresses_by_country[country].extend(grp['norm_address'].tolist())
                s2s3_ids_by_country[country].extend(grp['entity_id'].tolist())
                
            del chunk
            gc.collect()
            
        hb.stop()
        
    K = 20
    address_candidates_by_s1 = defaultdict(list)
    
    retrieval_times = []
    
    for country in needed_countries:
        s1_subset = val_s1[val_s1['country'] == country]
        if len(s1_subset) == 0: continue
            
        s1_addresses = s1_subset['norm_address'].tolist()
        s1_ids = s1_subset['entity_id'].tolist()
        
        s2s3_addresses = s2s3_addresses_by_country.get(country, [])
        s2s3_ids = s2s3_ids_by_country.get(country, [])
        
        if len(s2s3_ids) == 0: continue
            
        flush_print(f"\n============================================================")
        flush_print(f"Processing Country: {country}")
        flush_print(f"============================================================")
        flush_print(f"S1 records: {len(s1_addresses)}")
        flush_print(f"S2/S3 records: {len(s2s3_addresses)}")
        
        flush_print(f"\n[+] Building ADDRESS TF-IDF (word, 1-2)")
        t_tfidf = time.time()
        
        hb = Heartbeat(f"Stage: ADDRESS TF-IDF Building | Country: {country}", interval=30)
        hb.start()
        
        vectorizer_addr = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=0.8)
        X_s2s3_addr = vectorizer_addr.fit_transform(s2s3_addresses)
        X_s1_addr = vectorizer_addr.transform(s1_addresses)
        
        hb.stop()
        
        flush_print(f"Address Matrix shape S2/S3: {X_s2s3_addr.shape}")
        flush_print(f"Address Matrix nnz S2/S3: {X_s2s3_addr.nnz}")
        flush_print(f"Time: {time.time() - t_tfidf:.1f} sec")
        flush_print(f"RAM: {memory_usage():.1f} MB")
        
        flush_print(f"Converting Address matrix to CSR...")
        hb = Heartbeat(f"Stage: ADDRESS CSR Transpose | Country: {country}", interval=30)
        hb.start()
        X_s2s3_addr_T = X_s2s3_addr.T.tocsr()
        hb.stop()
        
        flush_print(f"Address Sparse Retrieval...")
        t_ret = time.time()
        batch_size = 1000
        n_batches = int(np.ceil(X_s1_addr.shape[0] / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_s1_addr.shape[0])
            X_s1_batch = X_s1_addr[start_idx:end_idx]
            
            hb = Heartbeat(f"Stage: ADDRESS retrieval | Country: {country} | Batch: {i+1}/{n_batches}", interval=30)
            hb.start()
            
            sim = sparse_dot_topn.sp_matmul_topn(X_s1_batch, X_s2s3_addr_T, top_n=K, n_threads=-1)
            top_k_indices = get_top_k_sparse(sim, K)
            
            hb.stop()
            
            for j, top_idx in enumerate(top_k_indices):
                s1_id = s1_ids[start_idx + j]
                mapped_ids = [s2s3_ids[idx] for idx in top_idx]
                address_candidates_by_s1[s1_id].extend(mapped_ids)
                
        retrieval_times.append(time.time() - t_ret)
                
        del X_s1_addr, X_s2s3_addr, X_s2s3_addr_T, vectorizer_addr
        s2s3_addresses_by_country.pop(country, None)
        s2s3_ids_by_country.pop(country, None)
        gc.collect()
        
    flush_print("\n==================================================")
    flush_print("SMOKE TEST EVALUATION")
    flush_print("==================================================")
    
    addr_only = defaultdict(list)
    for s1_id in val_s1_ids:
        addr_k20 = address_candidates_by_s1.get(s1_id, [])
        addr_only[s1_id] = list(set(addr_k20))
        
    counts = [len(addr_only[s]) for s in val_s1_ids]
    recall = calculate_recall(val_s1_ids, gt_dict, addr_only)
    
    flush_print(f"Address Candidate Recall: {recall*100:.2f}%")
    flush_print(f"Average candidates/S1: {np.mean(counts):.2f}")
    flush_print(f"Total Retrieval Time (all countries): {sum(retrieval_times):.1f} sec")
    flush_print(f"Peak RAM: {memory_usage():.1f} MB")
    
    out_file = "code/business_entity_resolution/experiments/exp3_smoke_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    final_results = {
        "recall": float(recall),
        "avg_cand": float(np.mean(counts)),
        "retrieval_time_sec": sum(retrieval_times),
        "peak_ram_mb": memory_usage()
    }
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
        
    flush_print("\nDONE.")

if __name__ == "__main__":
    main()
