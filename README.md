# NYC FHV Fare Prediction — DCNv2 Sprint

This repository captures the **12-hour sprint** during which we cleaned a 745 M-row New York City For-Hire Vehicle (FHV) dataset, engineered features, and trained a Deep & Cross Network v2 (DCNv2) regression model on 50 M rows with a Google Cloud **TPU v5p-16**.  
The final SavedModel, metrics, and helper scripts are provided so that anyone can reproduce our results on GCP or a local GPU.
Data captured from this Kaggle dataset: https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data

---

## Contents

```text
nyc-rideshare-dnn/
├─ metrics.json           # MAE / P95 / MAPE on the held-out 5 M slice
├─ model.zip              # zipped TensorFlow SavedModel (~8 MB)
├─ model/                 # un-zipped SavedModel directory (for direct loading)
├─ requirements.txt       # TF 2.15 · Polars 0.20.31 · gcsfs 2024.6.1
└─ scripts/
   ├─ train_50m.py        # data prep + DCNv2 training on 50 M rows
   └─ eval_5m.py          # evaluation on 5 M rows (pass --rows for 50 M)
```

---

## Quick-start (local machine)

1. **Create and activate a Python 3.12 virtual-env**

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2.	Install exact package versions
   
  ```bash
  pip install -r requirements.txt
  ```

3.	Unzip the SavedModel (optional)

  ```bash
  unzip model.zip      # produces the `model/` folder if you prefer
  ```

4.	Run the 5 M-row evaluation locally

  ```bash
  python scripts/eval_5m.py \
         --model_dir=model \
         --parquet=gs://nyc-taxi-fhv-460946772036/fhvhv_clean.zstd.parquet
  ```

---

End-to-end on Google Cloud TPU v5p-16

# 0. One-time: set your project + zone
```bash
export PROJECT_ID=<your-project>
export ZONE=us-central1-a
```

# 1. Spin up a TPU-VM
```bash
gcloud alpha compute tpus tpu-vm create fhv-v5p-16 \
       --zone=$ZONE --accelerator-type=v5p-16 \
       --version=tpu-vm-tf-2.15.0-pjrt
```

# 2. SSH into the TPU-VM
```bash
gcloud alpha compute tpus tpu-vm ssh fhv-v5p-16 --zone=$ZONE
```

# 3. Pull the code & data inside the VM
```bash
git clone https://github.com/<your-user>/nyc-rideshare-dnn.git
cd nyc-rideshare-dnn
sudo mkdir -p /mnt/disks/nvme && sudo chown $USER:$USER /mnt/disks/nvme
gsutil -m cp gs://nyc-taxi-fhv-460946772036/fhvhv_clean.zstd.parquet \
               /mnt/disks/nvme/
```

# 4. Install deps & train
```bash
pip install -r requirements.txt
python scripts/train_50m.py
```

# 5. Evaluate on 50 M rows
```bash
python scripts/eval_5m.py \
       --model_dir=model \
       --parquet=/mnt/disks/nvme/fhvhv_clean.zstd.parquet \
       --rows 50000000
```

Full TPU cost for the 12-hour sprint: ≈ $99.

---

# Results

| Split | Rows | MAE | P95 | MAPE |
|:------|:-----|:----|:----|:-----|
| Validation (5 M) | 5,000,000 | $0.72 | $1.34 | 3.07% |
| Large Eval | 50,000,000 | $0.69 | $1.27 | 3.12% |

(metrics stored in metrics.json)

---

## Data pipeline & feature engineering
- Source: NYC TLC FHV trip records (2020).
- Cleaning: removed nulls/negatives, clipped extreme numeric values at $400.
- Numeric features: fare components, trip time/miles, derived mph, temporal (hour, weekday, month).
- Categorical features: license # and base IDs, PU/DO locations, binary ride flags (shared, WAV, etc.).
- Sampling: random contiguous 50 M-row slice for training, shuffled and split 80 / 20.
- Polars handled columnar loading and transformations in streaming mode.

---

## Model architecture (DCNv2)
- Dense tower: 3 fully-connected layers (512-256-128) with ReLU + dropout.
- Cross network: 3 cross layers capturing high-order feature interactions.
- Embeddings: 16-dimensional tables for each categorical field.
- Optimizer: AdamW, LR = 1e-3, weight-decay = 1e-4.
- Batching: 2,048 rows per core → 32,768 global on a 16-core v5p.
- Training: 2 epochs (≈ 50 min each), early-stop on validation MAE.

---

## Reproducibility

Everything needed to retrain or fine-tune lives in this repo:
- Data paths hard-coded but can be overridden via CLI flags.
- requirements.txt locks dependencies.
- train_50m.py / eval_5m.py are self-contained—no hidden utilities.
