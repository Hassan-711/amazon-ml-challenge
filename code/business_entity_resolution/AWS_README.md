# AWS Production Setup & Run Instructions

## 1. Instance Recommendations
- **Recommended Instance**: `c6i.8xlarge` (32 vCPUs, 64 GB RAM) or `c6i.16xlarge` (64 vCPUs, 128 GB RAM)
- **Minimum Instance**: `c6i.4xlarge` (16 vCPUs, 32 GB RAM)
- **AMI**: Ubuntu 22.04 LTS (Deep Learning AMI recommended for pre-installed drivers)
- **Disk**: 50 GB EBS SSD minimum

## 2. Environment Setup
```bash
# Clone/download your code repository to the AWS instance
cd code/business_entity_resolution

# Install required packages
python3 -m pip install -r requirements.txt
```

## 3. Running the Pipeline
The pipeline is designed to be fully self-contained, memory-safe, and resume-able.

To start the pipeline on the full test set:
```bash
python3 src/final_aws_pipeline.py --test-dir /path/to/dataset/test --out-dir output/ --chunk-size 50000 --threads -1
```

If the SSH connection drops or the process crashes, simply rerun the same command. It reads from the `checkpoints/` directory and will pick up on the exact chunk it left off on!

## 4. Final Submission Generation
Once the script says `DONE!`, simply ZIP the repository:
```bash
zip -r my_submission.zip output/matching_results.tsv output/candidate_pairs.tsv code/ Documentation_template.pdf
```
