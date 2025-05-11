#!/usr/bin/env python
"""Evaluate fhv_dcnv2_b50m SavedModel on a 5-million-row random slice."""
import os, random, time, numpy as np, polars as pl, tensorflow as tf

PARQ      = "/mnt/disks/nvme/fhvhv_clean.zstd.parquet"
MODEL_DIR = "/mnt/disks/nvme/fhv_dcnv2_b50m"
N_TEST    = 5_000_000
BATCH     = 2_048         # must divide by global batch size (2 048 OK)

# ── 1. load random slice ───────────────────────────────────────────────
offset = random.randrange(0, 745_287_023 - N_TEST)
t0 = time.time()
df = (pl.scan_parquet(PARQ)
        .slice(offset, N_TEST)
        .collect(streaming=True))
print("Loaded test slice:", df.shape, "in", round(time.time()-t0,1), "s")

# ── 2. feature engineering – identical to training ─────────────────────
NUM = ["trip_miles","trip_time","base_passenger_fare","tolls","bcf",
       "sales_tax","congestion_surcharge","airport_fee","target_amount"]
BIN = ["shared_request_flag","shared_match_flag",
       "wav_request_flag","access_a_ride_flag"]
CAT = ["hvfhs_license_num","dispatching_base_num",
       "PULocationID","DOLocationID"]

df = (df.with_columns([pl.col(c).clip_max(400) for c in NUM[:-1]])
        .with_columns([
            pl.when(pl.col("trip_time")>0)
              .then(pl.col("trip_miles")/(pl.col("trip_time")/3600))
              .otherwise(0.0).alias("mph"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_wday"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month")])
        .with_columns([pl.col(NUM+["mph"]).fill_null(0.0).cast(pl.Float32)])
        .with_columns([pl.col(CAT+BIN).fill_null("UNK")])
        .filter(pl.col("target_amount") > 0))           # remove $0 trips

def tbl_to_dict(tbl: pl.DataFrame) -> dict[str, np.ndarray]:
    out={}
    for c in tbl.columns:
        if c=="target_amount": continue
        col = tbl[c]
        out[c] = (np.nan_to_num(col.to_numpy(), nan=0.0).astype("float32")
                  if col.dtype.is_numeric()
                  else col.to_numpy().astype(str))
    return out

x_all = tbl_to_dict(df)
y_all = df["target_amount"].to_numpy().astype("float32")
print("Rows kept for eval:", len(y_all))

# ── 3. load SavedModel & align feature list ────────────────────────────
saved    = tf.saved_model.load(MODEL_DIR)
infer_fn = saved.signatures['serving_default']
want     = list(infer_fn.structured_input_signature[1].keys())
x_all    = {k:v for k,v in x_all.items() if k in want}
out_key  = list(infer_fn.structured_outputs.keys())[0]
print("✔ serving signature expects", len(want), "features")

# ── 4. batched inference ───────────────────────────────────────────────
preds = np.empty_like(y_all)
for i in range(0, len(y_all), BATCH):
    batch = {k: tf.convert_to_tensor(v[i:i+BATCH]) for k,v in x_all.items()}
    p = infer_fn(**batch)[out_key].numpy().reshape(-1)   # already in dollars
    preds[i:i+len(p)] = p

# ── 5. metrics ─────────────────────────────────────────────────────────
abs_err = np.abs(preds - y_all)
mae  = abs_err.mean()
p95  = np.percentile(abs_err, 95)
mape = (abs_err / np.maximum(y_all, 1e-3)).mean() * 100   # ε to avoid /0

print(f"\n✅  evaluation done in {round(time.time()-t0,1)} s")
print(f"MAE  : ${mae:,.2f}")
print(f"P95  : ${p95:,.2f}")
print(f"MAPE :  {mape:,.2f}%")
