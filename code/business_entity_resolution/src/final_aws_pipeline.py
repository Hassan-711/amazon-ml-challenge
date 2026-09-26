import pandas as pd
import numpy as np
import os
import time
import psutil
import gc
import json
import argparse
from collections import defaultdict
import rapidfuzz
import re
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn
import scipy.sparse as sp
import threading
import sys
import signal

sys.shutdown_requested = False
def sigterm_handler(signum, frame):
    sys.shutdown_requested = True

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

class Heartbeat(threading.Thread):
    def __init__(self, message, interval=60):
        super().__init__()
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self.daemon = True
        self.start_time = time.time()
        
    def run(self):
        while not self._stop_event.wait(self.interval):
            elapsed = int(time.time() - self.start_time)
            flush_print(f"\n[HEARTBEAT] {self.message} | Elapsed: {elapsed} sec | RAM: {memory_usage():.1f} MB")
            
    def stop(self):
        self._stop_event.set()

def fast_normalize(s):
    return str(s).lower().replace(r'[^a-z0-9]', ' ').strip()

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

def compute_features(df):
    df['n1'] = df['name_s1'].fillna("").astype(str)
    df['n2'] = df['name_s23'].fillna("").astype(str)
    
    df['name_ratio'] = [rapidfuzz.fuzz.ratio(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_wratio'] = [rapidfuzz.fuzz.WRatio(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_partial'] = [rapidfuzz.fuzz.partial_ratio(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_jaro'] = [rapidfuzz.distance.JaroWinkler.normalized_similarity(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_lev'] = [rapidfuzz.distance.Levenshtein.normalized_similarity(a, b) for a, b in zip(df['n1'], df['n2'])]
    
    metrics = [token_metrics(a, b) for a, b in zip(df['n1'], df['n2'])]
    df['name_jaccard'] = [x[0] for x in metrics]
    df['name_contain'] = [x[1] for x in metrics]
    
    df['name_len_diff'] = np.abs(df['n1'].str.len() - df['n2'].str.len())
    df['name_len_ratio'] = df['n1'].str.len() / (df['n2'].str.len() + 1e-5)
    df['exact_name'] = (df['n1'] == df['n2']) & (df['n1'] != "")
    
    df['a1'] = df['address_s1'].fillna("").astype(str)
    df['a2'] = df['address_s23'].fillna("").astype(str)
    
    df['addr_ratio'] = [rapidfuzz.fuzz.ratio(a, b) for a, b in zip(df['a1'], df['a2'])]
    df['addr_wratio'] = [rapidfuzz.fuzz.WRatio(a, b) for a, b in zip(df['a1'], df['a2'])]
    
    metrics_a = [token_metrics(a, b) for a, b in zip(df['a1'], df['a2'])]
    df['addr_jaccard'] = [x[0] for x in metrics_a]
    df['addr_contain'] = [x[1] for x in metrics_a]
    
    df['exact_addr'] = (df['a1'] == df['a2']) & (df['a1'] != "")
    df['addr_missing'] = (df['a1'] == "") | (df['a2'] == "")
    
    nums1 = [extract_numbers(a) for a in df['a1']]
    nums2 = [extract_numbers(a) for a in df['a2']]
    num_agree = []
    for n1, n2 in zip(nums1, nums2):
        if not n1 or not n2: num_agree.append(-1.0)
        else: num_agree.append(float(len(n1.intersection(n2)) > 0))
    df['addr_num_agree'] = num_agree
    
    df['country_match'] = (df['country_s1'] == df['country_s23']).astype(float)
    
    feature_cols = [
        'name_ratio', 'name_wratio', 'name_partial', 'name_jaro', 'name_lev',
        'name_jaccard', 'name_contain', 'name_len_diff', 'name_len_ratio', 'exact_name',
        'addr_ratio', 'addr_wratio', 'addr_jaccard', 'addr_contain', 'exact_addr',
        'addr_missing', 'addr_num_agree', 'country_match'
    ]
    return df, feature_cols

def get_or_train_model():
    model_path = "output/lightgbm_model.txt"
    os.makedirs("output", exist_ok=True)
    if os.path.exists(model_path):
        flush_print(f"Loading existing model from {model_path}")
        return lgb.Booster(model_file=model_path)
    
    # Needs training! 
    flush_print("Model not found. You must run Exp4 locally or supply model.txt.")
    flush_print("Assuming local candidates_10000.parquet exists. Training quick model...")
    cand_file = "code/business_entity_resolution/experiments/candidates_10000.parquet"
    if not os.path.exists(cand_file):
        raise FileNotFoundError("Missing candidates_10000.parquet to train the model!")
        
    df_pairs = pd.read_parquet(cand_file)
    unique_s1 = set(df_pairs['source1_entity_id'])
    unique_s23 = set(df_pairs['matched_entity_id'])
    
    base_dir = "dataset/student_resource/dataset/train"
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', dtype=str)
    df_s1 = df_s1[df_s1['entity_id'].isin(unique_s1)]
    df_s1.rename(columns={'business_name':'name_s1', 'business_address':'address_s1', 'country':'country_s1'}, inplace=True)
    
    s23_rows = []
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        chunk_iter = pd.read_csv(os.path.join(base_dir, source), sep='\t', dtype=str, chunksize=1000000)
        for chunk in chunk_iter:
            c = chunk[chunk['entity_id'].isin(unique_s23)]
            s23_rows.append(c)
    df_s23 = pd.concat(s23_rows, ignore_index=True)
    df_s23.rename(columns={'business_name':'name_s23', 'business_address':'address_s23', 'country':'country_s23'}, inplace=True)
    
    df_s1['name_s1'] = df_s1['name_s1'].apply(fast_normalize)
    df_s1['address_s1'] = df_s1['address_s1'].apply(fast_normalize)
    df_s23['name_s23'] = df_s23['name_s23'].apply(fast_normalize)
    df_s23['address_s23'] = df_s23['address_s23'].apply(fast_normalize)
    
    df_pairs = df_pairs.merge(df_s1[['entity_id', 'name_s1', 'address_s1', 'country_s1']], left_on='source1_entity_id', right_on='entity_id', how='left')
    df_pairs = df_pairs.merge(df_s23[['entity_id', 'name_s23', 'address_s23', 'country_s23']], left_on='matched_entity_id', right_on='entity_id', how='left')
    
    df_gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)
    df_gt = df_gt[df_gt['source1_entity_id'].isin(unique_s1)]
    gt_dict = {}
    for _, row in df_gt.iterrows():
        matches = str(row['matched_entity_ids'])
        gt_dict[row['source1_entity_id']] = set() if (pd.isna(row['matched_entity_ids']) or matches == 'nan' or matches.strip() == '') else set([x.strip() for x in matches.split(',')])
            
    df_pairs['label'] = [1 if s2 in gt_dict.get(s1, set()) else 0 for s1, s2 in zip(df_pairs['source1_entity_id'], df_pairs['matched_entity_id'])]
    df_pairs, feature_cols = compute_features(df_pairs)
    
    X_train = df_pairs[feature_cols].astype(np.float32)
    y_train = df_pairs['label'].astype(np.float32)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'learning_rate': 0.1, 'num_leaves': 31, 'n_jobs': -1}
    gbm = lgb.train(params, lgb_train, num_boost_round=223) # From Exp4
    gbm.save_model(model_path)
    flush_print(f"Model saved to {model_path}")
    return gbm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/student_resource/dataset/test")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--threads", type=int, default=-1, help="Threads for Cython SparseDotTopN")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size")
    parser.add_argument("--threshold", type=float, default=0.65, help="Optimal threshold from Exp4")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    flush_print("============================================================")
    flush_print("AWS FULL COMPETITION PIPELINE")
    flush_print("============================================================")
    
    gbm = get_or_train_model()
    
    flush_print("\n[1/4] Loading Test S2/S3 globally...")
    s2s3_names_by_country = defaultdict(list)
    s2s3_addresses_by_country = defaultdict(list)
    s2s3_ids_by_country = defaultdict(list)
    
    for source in ["test_source2.tsv", "test_source3.tsv"]:
        filepath = os.path.join(args.test_dir, source)
        if not os.path.exists(filepath): continue
        chunk_iter = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str, chunksize=1000000)
        for chunk in chunk_iter:
            chunk['norm_name'] = fast_normalize_series(chunk['business_name'])
            chunk['norm_address'] = fast_normalize_series(chunk['business_address'])
            chunk = chunk[(chunk['norm_name'] != "") | (chunk['norm_address'] != "")]
            for country, grp in chunk.groupby('country'):
                s2s3_names_by_country[country].extend(grp['norm_name'].tolist())
                s2s3_addresses_by_country[country].extend(grp['norm_address'].tolist())
                s2s3_ids_by_country[country].extend(grp['entity_id'].tolist())
            del chunk; gc.collect()
            
    df_s1_full = pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    df_s1_full['norm_name'] = fast_normalize_series(df_s1_full['business_name'])
    df_s1_full['norm_address'] = fast_normalize_series(df_s1_full['business_address'])
    
    s1_countries = df_s1_full['country'].unique()
    
    for country in s1_countries:
        s2s3_names = s2s3_names_by_country.get(country, [])
        s2s3_addresses = s2s3_addresses_by_country.get(country, [])
        s2s3_ids = s2s3_ids_by_country.get(country, [])
        
        df_s1 = df_s1_full[df_s1_full['country'] == country].copy()
        
        if not s2s3_ids: 
            flush_print(f"\n[{country}] No S2/S3 records found. Saving empty predictions for {len(df_s1)} S1 entities.")
            chk_file_match = f"checkpoints/match_{country}_0.csv"
            chk_file_cand = f"checkpoints/cand_{country}_0.csv"
            df_s1[['entity_id']].assign(matched_entity_ids="").rename(columns={'entity_id':'source1_entity_id'}).to_csv(chk_file_match, index=False)
            df_s1[['entity_id']].assign(candidate_entity_ids="").rename(columns={'entity_id':'source1_entity_id'}).to_csv(chk_file_cand, index=False)
            continue
            
        flush_print(f"\n============================================================")
        flush_print(f"Processing Country: {country} | S1: {len(df_s1)} | S2/S3: {len(s2s3_ids)}")
        
        # Build Name Matrix
        has_names = any(bool(n) for n in s2s3_names)
        if has_names:
            hb = Heartbeat(f"Building NAME TF-IDF | {country}")
            hb.start()
            vec_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_df=0.8)
            X_s2s3_name = vec_name.fit_transform(s2s3_names)
            X_s2s3_name_T = X_s2s3_name.T.tocsr()
            del X_s2s3_name
            gc.collect()
            hb.stop()
            
        # Build Address Matrix
        has_addrs = any(bool(a) for a in s2s3_addresses)
        if has_addrs:
            hb = Heartbeat(f"Building ADDRESS TF-IDF | {country}")
            hb.start()
            vec_addr = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=0.8)
            X_s2s3_addr = vec_addr.fit_transform(s2s3_addresses)
            X_s2s3_addr_T = X_s2s3_addr.T.tocsr()
            del X_s2s3_addr
            gc.collect()
            hb.stop()
            
        # Exact Name Indexing
        s2s3_exact_name_map = defaultdict(list)
        for eid, nm in zip(s2s3_ids, s2s3_names):
            if nm: s2s3_exact_name_map[nm].append(eid)
            
        # Create mapping of S2/S3 entities so we can easily fetch them during feature engineering
        s2s3_df_country = pd.DataFrame({'entity_id': s2s3_ids, 'name_s23': s2s3_names, 'address_s23': s2s3_addresses, 'country_s23': country})
        
        # Process S1 in Chunks
        n_chunks = int(np.ceil(len(df_s1) / args.chunk_size))
        
        for c_idx in range(n_chunks):
            start = c_idx * args.chunk_size
            end = min((c_idx + 1) * args.chunk_size, len(df_s1))
            s1_chunk = df_s1.iloc[start:end].copy()
            
            chk_file_match = f"checkpoints/match_{country}_{c_idx}.tsv"
            chk_file_cand = f"checkpoints/cand_{country}_{c_idx}.tsv"
            
            if os.path.exists(chk_file_match) and os.path.exists(chk_file_cand):
                flush_print(f"Skipping chunk {c_idx+1}/{n_chunks} (Already exists)")
                continue
                
            if getattr(sys, 'shutdown_requested', False):
                flush_print("\n[!] Gracefully shutting down before starting next chunk as requested.")
                sys.exit(0)
                
            flush_print(f"\n[{country}] Processing S1 Chunk {c_idx+1}/{n_chunks} ({len(s1_chunk)} rows)")
            t_chunk = time.time()
            
            candidate_pairs = set()
            s1_names = s1_chunk['norm_name'].tolist()
            s1_addrs = s1_chunk['norm_address'].tolist()
            s1_ids = s1_chunk['entity_id'].tolist()
            
            # Exact
            for s1_id, nm in zip(s1_ids, s1_names):
                if nm in s2s3_exact_name_map:
                    for s2_id in s2s3_exact_name_map[nm]:
                        candidate_pairs.add((s1_id, s2_id))
                        
            # Name TF-IDF
            if has_names:
                X_s1_name = vec_name.transform(s1_names)
                sim = sparse_dot_topn.sp_matmul_topn(X_s1_name, X_s2s3_name_T, top_n=20, n_threads=args.threads)
                top_k = get_top_k_sparse(sim, 20)
                for j, tk in enumerate(top_k):
                    for idx in tk:
                        candidate_pairs.add((s1_ids[j], s2s3_ids[idx]))
                        
            # Address TF-IDF
            if has_addrs:
                X_s1_addr = vec_addr.transform(s1_addrs)
                sim = sparse_dot_topn.sp_matmul_topn(X_s1_addr, X_s2s3_addr_T, top_n=20, n_threads=args.threads)
                top_k = get_top_k_sparse(sim, 20)
                for j, tk in enumerate(top_k):
                    for idx in tk:
                        candidate_pairs.add((s1_ids[j], s2s3_ids[idx]))
                        
            flush_print(f"Generated {len(candidate_pairs)} candidate pairs. RAM: {memory_usage():.1f} MB")
            
            df_pairs = pd.DataFrame(list(candidate_pairs), columns=['source1_entity_id', 'matched_entity_id'])
            if len(df_pairs) == 0:
                s1_chunk[['entity_id']].assign(matched_entity_ids="").rename(columns={'entity_id':'source1_entity_id'}).to_csv(chk_file_match + ".tmp", sep='\t', index=False)
                s1_chunk[['entity_id']].assign(candidate_entity_ids="").rename(columns={'entity_id':'source1_entity_id'}).to_csv(chk_file_cand + ".tmp", sep='\t', index=False)
                os.rename(chk_file_match + ".tmp", chk_file_match)
                os.rename(chk_file_cand + ".tmp", chk_file_cand)
                continue
                
            # Merge Strings for ML
            s1_chunk_renamed = s1_chunk.rename(columns={'entity_id': 'source1_entity_id', 'norm_name': 'name_s1', 'norm_address': 'address_s1', 'country': 'country_s1'})
            df_pairs = df_pairs.merge(s1_chunk_renamed, on='source1_entity_id', how='left')
            df_pairs = df_pairs.merge(s2s3_df_country, left_on='matched_entity_id', right_on='entity_id', how='left')
            
            # Compute Features & Predict
            df_pairs, feature_cols = compute_features(df_pairs)
            X = df_pairs[feature_cols].astype(np.float32)
            df_pairs['pred'] = gbm.predict(X, num_iteration=gbm.best_iteration)
            
            # Filter
            df_matches = df_pairs[df_pairs['pred'] >= args.threshold]
            
            # Format Matches
            match_dict = defaultdict(list)
            for s1, s2 in zip(df_matches['source1_entity_id'], df_matches['matched_entity_id']):
                match_dict[s1].append(s2)
                
            cand_dict = defaultdict(list)
            for s1, s2 in zip(df_pairs['source1_entity_id'], df_pairs['matched_entity_id']):
                cand_dict[s1].append(s2)
                
            # Write out chunk ATOMICALLY
            res_match, res_cand = [], []
            for s1 in s1_ids:
                m_str = ",".join(sorted(match_dict.get(s1, [])))
                c_str = ",".join(sorted(cand_dict.get(s1, [])))
                res_match.append({'source1_entity_id': s1, 'matched_entity_ids': m_str})
                res_cand.append({'source1_entity_id': s1, 'candidate_entity_ids': c_str})
                
            pd.DataFrame(res_match).to_csv(chk_file_match + ".tmp", sep='\t', index=False)
            pd.DataFrame(res_cand).to_csv(chk_file_cand + ".tmp", sep='\t', index=False)
            
            os.rename(chk_file_match + ".tmp", chk_file_match)
            os.rename(chk_file_cand + ".tmp", chk_file_cand)
            
            flush_print(f"Chunk completed in {time.time()-t_chunk:.1f} sec. Found {len(df_matches)} matches.")
            del df_pairs, X, s1_chunk, df_matches
            gc.collect()
            
        # Free country resources
        if has_names: del X_s2s3_name_T, vec_name
        if has_addrs: del X_s2s3_addr_T, vec_addr
        del s2s3_df_country, s2s3_exact_name_map
        gc.collect()
        
    flush_print("\n[4/4] Combining chunks into Final Submission Package...")
    match_dfs, cand_dfs = [], []
    for f in os.listdir("checkpoints"):
        if f.startswith("match_"):
            match_dfs.append(pd.read_csv(os.path.join("checkpoints", f), sep='\t', dtype=str).fillna(""))
        elif f.startswith("cand_"):
            cand_dfs.append(pd.read_csv(os.path.join("checkpoints", f), sep='\t', dtype=str).fillna(""))
            
    final_match = pd.concat(match_dfs, ignore_index=True)
    final_cand = pd.concat(cand_dfs, ignore_index=True)
    
    # Validation per challenge rules
    assert len(final_match) == len(df_s1_full), f"Missing S1 rows! Got {len(final_match)}, expected {len(df_s1_full)}"
    assert len(final_cand) == len(df_s1_full), f"Missing S1 rows in candidate_pairs! Got {len(final_cand)}, expected {len(df_s1_full)}"
    
    final_match.to_csv(os.path.join(args.out_dir, "matching_results.tsv"), sep='\t', index=False)
    final_cand.to_csv(os.path.join(args.out_dir, "candidate_pairs.tsv"), sep='\t', index=False)
    
    flush_print(f"\nRunning official submission validator...")
    import subprocess
    cmd = ["python3", "utils/validate_submission.py", 
           "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
           "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
           "--test-dir", args.test_dir]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        flush_print("Validator found issues!")
        
    flush_print(f"\nDONE! Pipeline successfully produced output/matching_results.tsv")

if __name__ == "__main__":
    main()
