import pandas as pd
import numpy as np
import os
import time
import psutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
import gc

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2 # MB

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # keep alphanumeric and spaces
    text = ''.join(c if c.isalnum() else ' ' for c in text)
    # remove extra spaces
    text = ' '.join(text.split())
    return text

def calculate_macro_f05(val_s1_ids, gt_dict, pred_dict):
    f05_scores = []
    precisions = []
    recalls = []
    
    for s1_id in val_s1_ids:
        t = set(gt_dict.get(s1_id, []))
        p = set(pred_dict.get(s1_id, []))
        
        if len(t) == 0 and len(p) == 0:
            f05 = 1.0
            prec = 1.0
            rec = 1.0
        elif len(t) == 0 and len(p) > 0:
            f05 = 0.0
            prec = 0.0
            rec = 0.0
        elif len(t) > 0 and len(p) == 0:
            f05 = 0.0
            prec = 0.0
            rec = 0.0
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

def main():
    start_time = time.time()
    print(f"[{time.time()-start_time:.1f}s] Starting Experiment 1...")
    
    base_dir = "dataset/student_resource/dataset/train"
    
    print(f"[{time.time()-start_time:.1f}s] Loading S1 and GT...")
    df_s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', usecols=['entity_id', 'business_name', 'country'], dtype=str)
    df_gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)
    
    print(f"[{time.time()-start_time:.1f}s] Splitting validation set (10% to speed up Exp 1)...")
    # Using 10% validation sample for Exp 1 to be completely safe on memory/speed, but still a large sample (~220k S1 entities)
    train_s1, val_s1 = train_test_split(df_s1, test_size=0.1, random_state=42, stratify=df_s1['country'])
    
    val_s1_ids = val_s1['entity_id'].tolist()
    print(f"Train S1: {len(train_s1)}, Val S1: {len(val_s1)}")
    
    val_s1['norm_name'] = val_s1['business_name'].apply(normalize_text)
    
    df_gt_val = df_gt[df_gt['source1_entity_id'].isin(val_s1_ids)].copy()
    gt_dict = {}
    for _, row in df_gt_val.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or matches == 'nan' or matches.strip() == '':
            gt_dict[s1] = []
        else:
            gt_dict[s1] = [x.strip() for x in matches.split(',')]
            
    del df_gt_val
    del df_gt
    del df_s1
    del train_s1
    gc.collect()
    print(f"[{time.time()-start_time:.1f}s] Peak memory so far: {memory_usage():.1f} MB")
    
    print(f"[{time.time()-start_time:.1f}s] Processing S2 and S3 for exact name blocking...")
    name_to_s23_ids = defaultdict(list)
    
    for source in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(base_dir, source)
        chunk_idx = 0
        for chunk in pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name'], dtype=str, chunksize=1000000):
            chunk['norm_name'] = chunk['business_name'].apply(normalize_text)
            chunk = chunk[chunk['norm_name'] != ""]
            
            for _, row in chunk.iterrows():
                name_to_s23_ids[row['norm_name']].append(row['entity_id'])
            
            chunk_idx += 1
            print(f"Processed chunk {chunk_idx} of {source}, Memory: {memory_usage():.1f} MB")
    
    print(f"[{time.time()-start_time:.1f}s] Generating candidates for Val S1...")
    candidates_dict = {}
    candidate_counts = []
    
    for _, row in val_s1.iterrows():
        s1_id = row['entity_id']
        norm_name = row['norm_name']
        
        if norm_name != "" and norm_name in name_to_s23_ids:
            cands = name_to_s23_ids[norm_name]
        else:
            cands = []
            
        cands = list(set(cands))
        candidates_dict[s1_id] = cands
        candidate_counts.append(len(cands))
        
    print(f"[{time.time()-start_time:.1f}s] Computing metrics...")
    cand_recall = calculate_candidate_recall(val_s1_ids, gt_dict, candidates_dict)
    macro_f05, precision, recall = calculate_macro_f05(val_s1_ids, gt_dict, candidates_dict)
    
    total_candidates = sum(candidate_counts)
    avg_candidates = np.mean(candidate_counts)
    p95_candidates = np.percentile(candidate_counts, 95)
    p99_candidates = np.percentile(candidate_counts, 99)
    
    runtime = time.time() - start_time
    peak_mem = memory_usage()
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: EXACT MATCH BASELINE RESULTS")
    print("="*50)
    print(f"Candidate Recall:         {cand_recall:.4f}")
    print(f"Macro F0.5:               {macro_f05:.4f}")
    print(f"Precision:                {precision:.4f}")
    print(f"Recall (from predictions):{recall:.4f}")
    print("-" * 50)
    print(f"Total Val S1 Entities:    {len(val_s1_ids)}")
    print(f"Total Candidate Pairs:    {total_candidates}")
    print(f"Avg Candidates per S1:    {avg_candidates:.2f}")
    print(f"95th Pct Candidates:      {p95_candidates:.2f}")
    print(f"99th Pct Candidates:      {p99_candidates:.2f}")
    print("-" * 50)
    print(f"Runtime:                  {runtime:.1f} seconds")
    print(f"Peak Memory:              {peak_mem:.1f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
