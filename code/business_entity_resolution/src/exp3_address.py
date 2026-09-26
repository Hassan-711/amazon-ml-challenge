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
    flush_print("EXPERIMENT 3: COMPLEMENTARY ADDRESS BLOCKER")
    flush_print("============================================================")
    
    start_time = time.time()
    base_dir = "dataset/student_resource/dataset/train"
    
    flush_print("\n[1/2] Loading validation S1 (Subset: 10,000)")
    t0 = time.time()
    
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    _, val_s1_full = train_test_split(df_s1, test_size=0.1, random_state=42, stratify=df_s1['country'])
    
    val_s1 = val_s1_full.sample(n=10000, random_state=42).copy()
    val_s1['norm_name'] = fast_normalize_series(val_s1['business_name'])
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
    
    flush_print(f"Loaded: 10,000 S1")
    flush_print(f"Time: {time.time()-t0:.1f} sec")
    flush_print(f"RAM: {memory_usage():.1f} MB")
    
    needed_countries = val_s1['country'].unique()
    
    # We only use EXACT NAME blocker for Exp 1
    s1_exact_name_map = defaultdict(list)
    for s1_id, norm_name in zip(val_s1['entity_id'], val_s1['norm_name']):
        if norm_name:
            s1_exact_name_map[norm_name].append(s1_id)
            
    flush_print("\n[2/2] One-pass loading S2/S3 (Names & Addresses)")
    t0 = time.time()
    
    s2s3_names_by_country = defaultdict(list)
    s2s3_addresses_by_country = defaultdict(list)
    s2s3_ids_by_country = defaultdict(list)
    exact_matches_by_s1 = defaultdict(list)
    
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        if not os.path.exists(filepath):
            flush_print(f"Warning: {filepath} not found!")
            continue
            
        hb = Heartbeat(f"Stage: Loading {source}", interval=30)
        hb.start()
        
        chunk_iter = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str, chunksize=2000000)
        for i, chunk in enumerate(chunk_iter):
            chunk = chunk[chunk['country'].isin(needed_countries)]
            if len(chunk) == 0:
                continue
                
            norm_names = fast_normalize_series(chunk['business_name'])
            norm_addresses = fast_normalize_series(chunk['business_address'])
            chunk['norm_name'] = norm_names
            chunk['norm_address'] = norm_addresses
            
            # Keep if EITHER name or address is valid
            valid_mask = (chunk['norm_name'] != "") | (chunk['norm_address'] != "")
            chunk = chunk[valid_mask]
            
            for country, grp in chunk.groupby('country'):
                s2s3_names_by_country[country].extend(grp['norm_name'].tolist())
                s2s3_addresses_by_country[country].extend(grp['norm_address'].tolist())
                s2s3_ids_by_country[country].extend(grp['entity_id'].tolist())
                
            # Exact Match Blocking on NAME
            for eid, name in zip(chunk['entity_id'], chunk['norm_name']):
                if name and name in s1_exact_name_map:
                    for s1_id in s1_exact_name_map[name]:
                        exact_matches_by_s1[s1_id].append(eid)
                        
            del chunk, norm_names, norm_addresses, valid_mask
            gc.collect()
            
        hb.stop()
            
    flush_print(f"Total time loading S2/S3: {time.time()-t0:.1f} sec")
    flush_print(f"RAM: {memory_usage():.1f} MB")
    
    analyzer = "char_wb"
    ngram_range = (3,5)
    K = 20
    
    name_candidates_by_s1 = defaultdict(list)
    address_candidates_by_s1 = defaultdict(list)
    
    # Process country by country
    for country in needed_countries:
        s1_subset = val_s1[val_s1['country'] == country]
        if len(s1_subset) == 0:
            continue
            
        s1_names = s1_subset['norm_name'].tolist()
        s1_addresses = s1_subset['norm_address'].tolist()
        s1_ids = s1_subset['entity_id'].tolist()
        
        s2s3_names = s2s3_names_by_country.get(country, [])
        s2s3_addresses = s2s3_addresses_by_country.get(country, [])
        s2s3_ids = s2s3_ids_by_country.get(country, [])
        
        flush_print(f"\n============================================================")
        flush_print(f"Processing Country: {country}")
        flush_print(f"============================================================")
        flush_print(f"S1 records: {len(s1_names)}")
        flush_print(f"S2/S3 records: {len(s2s3_names)}")
        
        if len(s2s3_ids) == 0:
            continue
            
        # ------------------------------------------------------------------
        # NAME TF-IDF
        # ------------------------------------------------------------------
        has_valid_names = any(bool(n) for n in s2s3_names)
        if has_valid_names:
            flush_print(f"\n[+] Building NAME TF-IDF")
            t_tfidf = time.time()
            
            hb = Heartbeat(f"Stage: NAME TF-IDF Building | Country: {country}", interval=30)
            hb.start()
            
            vectorizer_name = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2, max_df=0.8)
            X_s2s3_name = vectorizer_name.fit_transform(s2s3_names)
            X_s1_name = vectorizer_name.transform(s1_names)
            
            hb.stop()
            
            flush_print(f"Name Matrix shape S2/S3: {X_s2s3_name.shape}")
            flush_print(f"Name Matrix nnz S2/S3: {X_s2s3_name.nnz}")
            flush_print(f"Time: {time.time() - t_tfidf:.1f} sec")
            
            flush_print(f"Converting Name matrix to CSR...")
            hb = Heartbeat(f"Stage: NAME CSR Transpose | Country: {country}", interval=30)
            hb.start()
            X_s2s3_name_T = X_s2s3_name.T.tocsr()
            hb.stop()
            
            flush_print(f"Name Sparse Retrieval...")
            batch_size = 1000
            n_batches = int(np.ceil(X_s1_name.shape[0] / batch_size))
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_s1_name.shape[0])
                X_s1_batch = X_s1_name[start_idx:end_idx]
                
                hb = Heartbeat(f"Stage: NAME retrieval | Country: {country} | Batch: {i+1}/{n_batches}", interval=30)
                hb.start()
                
                sim = sparse_dot_topn.sp_matmul_topn(X_s1_batch, X_s2s3_name_T, top_n=K, n_threads=-1)
                top_k_indices = get_top_k_sparse(sim, K)
                
                hb.stop()
                
                for j, top_idx in enumerate(top_k_indices):
                    s1_id = s1_ids[start_idx + j]
                    mapped_ids = [s2s3_ids[idx] for idx in top_idx]
                    name_candidates_by_s1[s1_id].extend(mapped_ids)
                    
            del X_s1_name, X_s2s3_name, X_s2s3_name_T, vectorizer_name
            gc.collect()

        # ------------------------------------------------------------------
        # ADDRESS TF-IDF
        # ------------------------------------------------------------------
        has_valid_addresses = any(bool(a) for a in s2s3_addresses)
        if has_valid_addresses:
            flush_print(f"\n[+] Building ADDRESS TF-IDF")
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
            
            flush_print(f"Converting Address matrix to CSR...")
            hb = Heartbeat(f"Stage: ADDRESS CSR Transpose | Country: {country}", interval=30)
            hb.start()
            X_s2s3_addr_T = X_s2s3_addr.T.tocsr()
            hb.stop()
            
            flush_print(f"Address Sparse Retrieval...")
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
                    
            del X_s1_addr, X_s2s3_addr, X_s2s3_addr_T, vectorizer_addr
            gc.collect()

        # Free country memory entirely
        del s1_names, s1_addresses, s1_ids
        s2s3_names_by_country.pop(country, None)
        s2s3_addresses_by_country.pop(country, None)
        s2s3_ids_by_country.pop(country, None)
        gc.collect()
        
    flush_print("\n==================================================")
    flush_print("EVALUATION")
    flush_print("==================================================")
    
    addr_only = defaultdict(list)
    exp1_exp2 = defaultdict(list)
    union_all = defaultdict(list)
    
    for s1_id in val_s1_ids:
        exact = exact_matches_by_s1.get(s1_id, [])
        name_k20 = name_candidates_by_s1.get(s1_id, [])
        addr_k20 = address_candidates_by_s1.get(s1_id, [])
        
        addr_only[s1_id] = list(set(addr_k20))
        exp1_exp2[s1_id] = list(set(exact + name_k20))
        union_all[s1_id] = list(set(exact + name_k20 + addr_k20))
        
    def eval_dict(cand_dict):
        counts = [len(cand_dict[s]) for s in val_s1_ids]
        recall = calculate_recall(val_s1_ids, gt_dict, cand_dict)
        return {
            "recall": float(recall),
            "avg_cand": float(np.mean(counts)),
            "p95_cand": float(np.percentile(counts, 95)),
            "p99_cand": float(np.percentile(counts, 99))
        }
        
    res_addr = eval_dict(addr_only)
    res_e1e2 = eval_dict(exp1_exp2)
    res_all = eval_dict(union_all)
    
    flush_print(f"\n[Address-Only] Candidate Recall: {res_addr['recall']*100:.2f}% (Avg: {res_addr['avg_cand']:.1f})")
    flush_print(f"[Exp1 + Exp2] Candidate Recall: {res_e1e2['recall']*100:.2f}% (Avg: {res_e1e2['avg_cand']:.1f})")
    flush_print(f"[Exp1 + Exp2 + Address] Candidate Recall: {res_all['recall']*100:.2f}% (Avg: {res_all['avg_cand']:.1f})")
    
    out_file = "code/business_entity_resolution/experiments/exp3_address_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    final_results = {
        "configuration": {
            "analyzer": analyzer,
            "ngram_range": ngram_range,
            "K": K
        },
        "validation_size": len(val_s1_ids),
        "metrics": {
            "Address_Only": res_addr,
            "Exp1_Exp2": res_e1e2,
            "Union_All": res_all
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
