import pandas as pd
import numpy as np
import os
import time
import psutil
from collections import defaultdict
import gc
import scipy.sparse as sp
import sys
import threading
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn
import argparse

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
            flush_print(f"Still running... Elapsed: {elapsed} sec")
            
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on 1,000 subset instead of 10,000")
    args = parser.parse_args()
    
    n_samples = 1000 if args.smoke else 10000
    suffix = "smoke" if args.smoke else "10000"
    out_file = f"code/business_entity_resolution/experiments/candidates_{suffix}.parquet"
    
    flush_print("============================================================")
    flush_print(f"EXP4: CANDIDATE GENERATION ({n_samples} S1)")
    flush_print("============================================================")
    
    start_time = time.time()
    base_dir = "dataset/student_resource/dataset/train"
    
    flush_print(f"\n[1/2] Loading validation S1 (Subset: {n_samples})")
    
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    _, val_s1_full = train_test_split(df_s1, test_size=0.1, random_state=42, stratify=df_s1['country'])
    
    val_s1 = val_s1_full.sample(n=n_samples, random_state=42).copy()
    val_s1['norm_name'] = fast_normalize_series(val_s1['business_name'])
    val_s1['norm_address'] = fast_normalize_series(val_s1['business_address'])
    
    s1_exact_name_map = defaultdict(list)
    for s1_id, norm_name in zip(val_s1['entity_id'], val_s1['norm_name']):
        if norm_name:
            s1_exact_name_map[norm_name].append(s1_id)
            
    del df_s1, val_s1_full
    gc.collect()
    
    needed_countries = val_s1['country'].unique()
    
    flush_print("\n[2/2] One-pass loading S2/S3 (Names & Addresses)")
    
    s2s3_names_by_country = defaultdict(list)
    s2s3_addresses_by_country = defaultdict(list)
    s2s3_ids_by_country = defaultdict(list)
    candidate_pairs = set() # (s1_id, s2s3_id)
    
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        if not os.path.exists(filepath): continue
            
        hb = Heartbeat(f"Stage: Loading {source}", interval=30)
        hb.start()
        
        chunk_iter = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str, chunksize=2000000)
        for chunk in chunk_iter:
            chunk = chunk[chunk['country'].isin(needed_countries)]
            if len(chunk) == 0: continue
            
            chunk['norm_name'] = fast_normalize_series(chunk['business_name'])
            chunk['norm_address'] = fast_normalize_series(chunk['business_address'])
            
            valid_mask = (chunk['norm_name'] != "") | (chunk['norm_address'] != "")
            chunk = chunk[valid_mask]
            
            for country, grp in chunk.groupby('country'):
                s2s3_names_by_country[country].extend(grp['norm_name'].tolist())
                s2s3_addresses_by_country[country].extend(grp['norm_address'].tolist())
                s2s3_ids_by_country[country].extend(grp['entity_id'].tolist())
                
            for eid, name in zip(chunk['entity_id'], chunk['norm_name']):
                if name and name in s1_exact_name_map:
                    for s1_id in s1_exact_name_map[name]:
                        candidate_pairs.add((s1_id, eid))
                        
            del chunk, valid_mask
            gc.collect()
            
        hb.stop()
        
    for country in needed_countries:
        s1_subset = val_s1[val_s1['country'] == country]
        if len(s1_subset) == 0: continue
            
        s1_names = s1_subset['norm_name'].tolist()
        s1_addresses = s1_subset['norm_address'].tolist()
        s1_ids = s1_subset['entity_id'].tolist()
        
        s2s3_names = s2s3_names_by_country.get(country, [])
        s2s3_addresses = s2s3_addresses_by_country.get(country, [])
        s2s3_ids = s2s3_ids_by_country.get(country, [])
        
        if len(s2s3_ids) == 0: continue
        
        flush_print(f"\n============================================================")
        flush_print(f"Processing Country: {country}")
        flush_print(f"============================================================")
        
        # 1. NAME TF-IDF
        has_valid_names = any(bool(n) for n in s2s3_names)
        if has_valid_names:
            hb = Heartbeat(f"Stage: NAME TF-IDF Building | {country}", interval=30)
            hb.start()
            vectorizer_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_df=0.8)
            X_s2s3_name = vectorizer_name.fit_transform(s2s3_names)
            X_s1_name = vectorizer_name.transform(s1_names)
            hb.stop()
            
            hb = Heartbeat(f"Stage: NAME CSR Transpose | {country}", interval=30)
            hb.start()
            X_s2s3_name_T = X_s2s3_name.T.tocsr()
            hb.stop()
            
            batch_size = 1000
            n_batches = int(np.ceil(X_s1_name.shape[0] / batch_size))
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, X_s1_name.shape[0])
                sim = sparse_dot_topn.sp_matmul_topn(X_s1_name[start:end], X_s2s3_name_T, top_n=20, n_threads=-1)
                top_k_indices = get_top_k_sparse(sim, 20)
                for j, top_idx in enumerate(top_k_indices):
                    s1_id = s1_ids[start + j]
                    for idx in top_idx:
                        candidate_pairs.add((s1_id, s2s3_ids[idx]))
                        
            del X_s1_name, X_s2s3_name, X_s2s3_name_T, vectorizer_name
            gc.collect()

        # 2. ADDRESS TF-IDF
        has_valid_addresses = any(bool(a) for a in s2s3_addresses)
        if has_valid_addresses:
            hb = Heartbeat(f"Stage: ADDRESS TF-IDF Building | {country}", interval=30)
            hb.start()
            vectorizer_addr = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=0.8)
            X_s2s3_addr = vectorizer_addr.fit_transform(s2s3_addresses)
            X_s1_addr = vectorizer_addr.transform(s1_addresses)
            hb.stop()
            
            hb = Heartbeat(f"Stage: ADDRESS CSR Transpose | {country}", interval=30)
            hb.start()
            X_s2s3_addr_T = X_s2s3_addr.T.tocsr()
            hb.stop()
            
            batch_size = 1000
            n_batches = int(np.ceil(X_s1_addr.shape[0] / batch_size))
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, X_s1_addr.shape[0])
                sim = sparse_dot_topn.sp_matmul_topn(X_s1_addr[start:end], X_s2s3_addr_T, top_n=20, n_threads=-1)
                top_k_indices = get_top_k_sparse(sim, 20)
                for j, top_idx in enumerate(top_k_indices):
                    s1_id = s1_ids[start + j]
                    for idx in top_idx:
                        candidate_pairs.add((s1_id, s2s3_ids[idx]))
                        
            del X_s1_addr, X_s2s3_addr, X_s2s3_addr_T, vectorizer_addr
            gc.collect()

        s2s3_names_by_country.pop(country, None)
        s2s3_addresses_by_country.pop(country, None)
        s2s3_ids_by_country.pop(country, None)
        gc.collect()
        
    df_pairs = pd.DataFrame(list(candidate_pairs), columns=['source1_entity_id', 'matched_entity_id'])
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_pairs.to_parquet(out_file, index=False)
    
    flush_print(f"\nSaved {len(df_pairs)} candidate pairs to {out_file}")
    flush_print(f"Total time: {time.time()-start_time:.1f} sec")
    flush_print("DONE.")

if __name__ == "__main__":
    main()
