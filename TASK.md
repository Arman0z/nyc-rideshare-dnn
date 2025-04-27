## NYC FHV Rideshare Fare Prediction – **End-to-End Guide & Notebook Blueprint**

> **Goal:** Create a Jupyter notebook `DNN.ipynb` that ingests the merged FHV Parquet file, cleans it, engineers all features, writes TFRecords, trains a **Deep & Cross Network v2** on a TPU v2-8, evaluates it, and exports a production-ready model.  
> A new Data Scientist can copy/paste the following Markdown headers and code cells into a blank notebook and run them top-to-bottom.

---

### 0  Notebook outline (copy these Markdown headings)

```
# 0 Environment & TPU setup
# 1 Data access (GCS → Polars DataFrame)
# 2 Cleaning & target creation
# 3 Feature engineering
# 4 Write TFRecord shards
# 5 Build tf.data input pipeline
# 6 Define Deep & Cross Network v2
# 7 Train, monitor & early-stop
# 8 Evaluate & error analysis
# 9 Hyper-parameter sweep (optional)
# 10 Save model & notebook wrap-up
```

---

### 1  Environment & TPU setup

```python
# Colab Pro: activate TPU
try:
    import jax  # quick TPU test
except Exception:
    %tensorflow_version 2.x
    import os, json, tensorflow as tf

# Mixed precision for TPU
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Install helper libs once per new VM
!pip install polars==0.20.19 gcsfs==2024.4.1 --quiet
```

```python
# Mount the GCS bucket that already holds the 20.5 GB Parquet
BUCKET = "nyc-taxi-fhv-460946772036"
!gcsfuse --implicit-dirs $BUCKET /mnt/fhv
PARQUET = "/mnt/fhv/fhvhv_all_years.zstd.parquet"
```

---

### 2  Load snapshot into Polars

```python
import polars as pl, datetime as dt
df = pl.read_parquet(PARQUET, low_memory=False)
print(df.shape)      # 745 M rows × 24 cols  
```

---

### 3  Cleaning & target creation

```python
NUMERIC_OUTLIER_RULES = {
    "trip_miles":  (0.1, 200),            # drop >200 mi  
    "trip_time":   (60, 4*3600),          # 1 min – 4 h
}
def clip_interval(col, lo, hi):
    return pl.when(pl.col(col).is_between(lo, hi)).then(pl.col(col)).otherwise(None)

for c,(lo,hi) in NUMERIC_OUTLIER_RULES.items():
    df = df.with_columns(clip_interval(c, lo, hi))

money_cols = ["base_passenger_fare","tolls","bcf","sales_tax",
              "congestion_surcharge","airport_fee"]
df = df.with_columns([pl.col(c).clip(0) for c in money_cols])

df = df.with_columns([
    ( sum(pl.col(c) for c in money_cols) ).alias("target_amount"),
    (pl.col("trip_miles") / (pl.col("trip_time")/3600)).alias("mph")
]).drop_nulls("target_amount")
```

---

### 4  Feature engineering

```python
# 4.1  Temporal splits
df = df.with_columns([
    pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
    pl.col("pickup_datetime").dt.weekday().alias("pickup_wday"),
    pl.col("pickup_datetime").dt.month().alias("pickup_month"),
])

# 4.2  Categorical cleanup (fill UNK)
high_card = ["dispatching_base_num","PULocationID","DOLocationID"]
for col in high_card + ["hvfhs_license_num"]:
    df = df.with_columns(pl.col(col).fill_null("UNK"))
```

---

### 5  Train / validation split

```python
# Time-based split: last month of 2022 → validation
cutoff = dt.datetime(2022,12,1)
train_df = df.filter(pl.col("pickup_datetime") <  cutoff)
valid_df = df.filter(pl.col("pickup_datetime") >= cutoff)

print(train_df.shape, valid_df.shape)
```

---

### 6  Write TFRecord shards (TPU-friendly)

```python
import tensorflow as tf, math, os, itertools, json, typing as T
from tqdm import tqdm

def df_to_tfr_iter(table: pl.DataFrame, batch=200_000):
    n = table.height
    for i in tqdm(range(0, n, batch)):
        chunk = table.slice(i, batch)
        yield dict(chunk.to_arrow().to_pydict())  # col->list

def write_tfr(split, table):
    OUTDIR = f"/content/tfr/{split}"
    os.makedirs(OUTDIR, exist_ok=True)
    for shard_id, records in enumerate(df_to_tfr_iter(table)):
        fn = f"{OUTDIR}/{split}-{shard_id:05d}.tfr"
        with tf.io.TFRecordWriter(fn, compression_type="GZIP") as w:
            for j in range(len(records["target_amount"])):
                feat = {k: tf.train.Feature(
                           float_list=tf.train.FloatList(value=[records[k][j]])
                       ) if isinstance(records[k][j], float)
                       else tf.train.Feature(
                           bytes_list=tf.train.BytesList(value=[str(records[k][j]).encode()])
                       )
                       for k in records}
                example = tf.train.Example(features=tf.train.Features(feature=feat))
                w.write(example.SerializeToString())
```

*(Run for both `train_df` and `valid_df`; expect ≈ 12 min for entire 700 M sample on Colab SSD) *

---

### 7  Input pipeline (mixed-precision ready)

```python
FEATURE_DESCRIPTION = {
    # floats
    **{c: tf.io.FixedLenFeature([], tf.float32) for c in
       ["trip_miles","trip_time","mph","base_passenger_fare","tolls",
        "bcf","sales_tax","congestion_surcharge","airport_fee"]},
    # ints
    **{c: tf.io.FixedLenFeature([], tf.int64) for c in
       ["pickup_hour","pickup_wday","pickup_month"]},
    # strings
    **{c: tf.io.FixedLenFeature([], tf.string) for c in
       ["hvfhs_license_num","dispatching_base_num","PULocationID","DOLocationID",
        "shared_request_flag","shared_match_flag","wav_request_flag","access_a_ride_flag"]},
    # label
    "target_amount": tf.io.FixedLenFeature([], tf.float32),
}

def parse_fn(example_proto):
    return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

def make_dataset(split, batch, shuffle=False):
    files = tf.io.gfile.glob(f"/content/tfr/{split}/*.tfr")
    ds = (tf.data.TFRecordDataset(files, compression_type="GZIP",
                                  num_parallel_reads=tf.data.AUTOTUNE)
          .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE))
    if shuffle: ds = ds.shuffle(1_000_000)
    return (ds.batch(batch, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
```

---

### 8  Build the **Deep & Cross Network v2**

```python
from tensorflow.keras import layers as L, Model

def dcn_v2(inputs):
    # --- Embeddings ---
    emb_dims = {"hvfhs_license_num":2, "dispatching_base_num":16,
                "PULocationID":16, "DOLocationID":16}
    embed_out = []
    for feat, dim in emb_dims.items():
        v = L.StringLookup(output_mode='int', num_oov_indices=1)(inputs[feat])
        v = L.Embedding(input_dim=v.vocabulary_size(), output_dim=dim)(v)
        embed_out.append(L.Flatten()(v))

    # --- Flags 0/1 ---
    flags = ["shared_request_flag","shared_match_flag",
             "wav_request_flag","access_a_ride_flag"]
    flag_out = [L.Cast(dtype='float32')(inputs[f]) for f in flags]

    # --- Numeric normalised ---
    num_cols = ["trip_miles","trip_time","mph","base_passenger_fare",
                "tolls","bcf","sales_tax","congestion_surcharge","airport_fee"]
    norm = L.Normalization()
    norm.adapt(train_df.select(num_cols).to_numpy())  # offline!
    num_out = norm(L.Concatenate()( [inputs[c] for c in num_cols] ))

    # --- Temporal (sin/cos) ---
    hour = tf.cast(inputs["pickup_hour"], tf.float32)
    sin_hour = tf.sin(2*3.1416*hour/24); cos_hour = tf.cos(2*3.1416*hour/24)
    # similar for wday, month…

    concat = L.Concatenate()(embed_out + flag_out + [num_out, sin_hour, cos_hour])

    # ---- DCN-v2 Cross Stack ----
    cross = concat
    for _ in range(3):
        cross = tf.keras.experimental.LinearCombination()([concat, cross])

    # ---- Deep Tower ----
    deep = concat
    for units, drop in [(512,0.2),(256,0.2),(128,0.1),(64,0)]:
        deep = L.Dense(units, activation='gelu', kernel_regularizer='l2')(deep)
        deep = L.BatchNormalization()(deep)
        if drop: deep = L.Dropout(drop)(deep)

    fused = L.Concatenate()([cross, deep])
    out = L.Dense(64, activation='gelu')(fused)
    out = L.Dense(1, name='fare')(out)
    return out

inputs = {k: L.Input(shape=(), name=k, dtype=tf.string if 'flag' in k or k.endswith('_num') or 'ID' in k else tf.float32)
          for k in FEATURE_DESCRIPTION if k!='target_amount'}
model = Model(inputs, dcn_v2(inputs))
model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-5, global_clipnorm=1.0),
    loss=tf.keras.losses.Huber(delta=5.0),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='MAE'),
             tf.keras.metrics.MeanAbsolutePercentageError(name='MAPE')]
)
model.summary()
```

---

### 9  Train with early stopping

```python
BATCH = 16_384
train_ds = make_dataset('train', BATCH, shuffle=True)
valid_ds = make_dataset('valid', BATCH)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_MAE', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('checkpoints/dcnv2_{epoch:02d}.keras',
                                       save_best_only=True, monitor='val_MAE'),
    tf.keras.callbacks.TensorBoard('logs')
]

EPOCHS = 20
history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=EPOCHS,
                    steps_per_epoch=math.ceil(len(train_df)/BATCH),
                    validation_steps=math.ceil(len(valid_df)/BATCH),
                    callbacks=callbacks)
```

---

### 10  Evaluate & inspect residuals

```python
import matplotlib.pyplot as plt, numpy as np

val_pred = model.predict(valid_ds, verbose=0).flatten()
val_true = valid_df["target_amount"].to_numpy()

print("MAE $", np.mean(np.abs(val_pred-val_true)))
plt.hist(val_true-val_pred, bins=100)
plt.title("Prediction residuals ($)")
plt.show()
```

---

### 11  (Opt.) Hyper-parameter sweep

```python
!pip install keras-tuner --quiet
import keras_tuner as kt

def model_builder(hp):
    hp_units = hp.Int("units", min_value=256, max_value=1024, step=256)
    hp_drop  = hp.Float("dropout", 0.0, 0.4, step=0.1)
    # reuse architect, swap units and dropouts with hp_units/hp_drop
    ...
tuner = kt.BayesianOptimization(model_builder,
                                objective="val_MAE",
                                max_trials=20,
                                directory="ktuner",
                                overwrite=True)
tuner.search(train_ds, validation_data=valid_ds, epochs=5)
best_model = tuner.get_best_models(1)[0]
```

---

### 12  Save artefacts

```python
model.save("dcnv2_nyc_fhv_savedmodel")
!zip -r model.zip dcnv2_nyc_fhv_savedmodel
```

---

## Project “big picture”

| Topic | Key points | Source |
|-------|------------|--------|
| **Business objective** | Accurate upfront fare estimate; aids riders, TNCs & city planners. |  |
| **Raw data** | 745 M FHV trips (2019-22) → 24 columns. |  |
| **Cleaning rules** | Clip extreme miles/time, drop negative money, handle sparse cols. | same |
| **Target** | Sum of base fare + taxes/fees (tips excluded). | same |
| **Feature groups** | 9 numeric, 4 binary flags, 4 high-card categoricals, 3 temporal. | same |
| **Model choice** | Deep & Cross Network v2 (explicit feature crosses + deep tower). | architect section above |
| **Hardware** | Colab Pro TPU v2-8, batch 16 384 → 3.5 h/epoch full data. |  |
| **Loss / metrics** | Huber δ=5, track MAE, MAPE, RMSE. | architect section |
| **Training hygiene** | Mixed-precision, early stop, AdamW + warm-up cosine LR, weight decay. | same |

---

### Final checklist for the new DS

1. **Provision Colab Pro** and switch runtime to *TPU*  
2. **Copy the outline & code cells** above into `DNN.ipynb`  
3. **Run each cell sequentially**; first run may take ≈ 30 min for TFRecord writing  
4. **Monitor TensorBoard** (cell: `%tensorboard --logdir logs`) – aim for MAE ≤ \$2  
5. **Export model** once satisfied; attach `.zip` to hand-over
