## NYC FHV Fare Prediction – **Build Guide (MAE + SGD)**

### Notebook outline – paste these Markdown headings

```
# 0 Environment & TPU setup
# 1 Data access (GCS → Polars DataFrame)
# 2 Cleaning & target creation
# 3 Feature engineering
# 4 Train/valid split
# 5 Write TFRecord shards  (optional)
# 6 tf.data input pipeline
# 7 Deep & Cross Network v2  (6-layer ReLU deep tower)
# 8 Train (SGD + Nesterov, MAE loss)
# 9 Evaluate & error analysis
#10 Save model & wrap-up
```

---

### 0 Environment & TPU setup

```python
# Colab Pro → TPU v2-8
%tensorflow_version 2.x
import os, json, math, datetime as dt
import tensorflow as tf

# Mixed-precision (bfloat16) for TPU
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# One-time installs
!pip install polars==0.20.19 gcsfs==2024.4.1 --quiet
```

---

### 1 Data access

```python
BUCKET = "nyc-taxi-fhv-460946772036"
!gcsfuse --implicit-dirs $BUCKET /mnt/fhv

PARQUET = "/mnt/fhv/fhvhv_all_years.zstd.parquet"
import polars as pl
df = pl.read_parquet(PARQUET, low_memory=False)
print(df.shape)          # 745 M × 24
```

---

### 2 Cleaning & target creation   *(same rules as v1)*

```python
# Clip extreme miles / time
CLIP = {"trip_miles": (0.1, 200),
        "trip_time":  (60, 4*3600)}
for c,(lo,hi) in CLIP.items():
    df = df.with_columns(
        pl.when(pl.col(c).is_between(lo,hi))
          .then(pl.col(c)).otherwise(None)
    )

money = ["base_passenger_fare","tolls","bcf",
         "sales_tax","congestion_surcharge","airport_fee"]
df = df.with_columns([pl.col(c).clip(0) for c in money])

df = df.with_columns([
    ( sum(pl.col(c) for c in money) ).alias("target_amount"),
    (pl.col("trip_miles") / (pl.col("trip_time")/3600)).alias("mph")
]).drop_nulls("target_amount")
```

---

### 3 Feature engineering

```python
df = df.with_columns([
    pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
    pl.col("pickup_datetime").dt.weekday().alias("pickup_wday"),
    pl.col("pickup_datetime").dt.month().alias("pickup_month"),
])

high_card = ["dispatching_base_num","PULocationID","DOLocationID",
             "hvfhs_license_num"]
for col in high_card:
    df = df.with_columns(pl.col(col).fill_null("UNK"))
```

---

### 4 Time-based split

```python
cutoff = dt.datetime(2022,12,1)
train_df = df.filter(pl.col("pickup_datetime") <  cutoff)
valid_df = df.filter(pl.col("pickup_datetime") >= cutoff)
print(train_df.shape, valid_df.shape)
```

---

### 5 Write TFRecord shards *(optional but fastest on TPU)*

> Skip this section if you have plenty of RAM and prefer a direct in-RAM `tf.data.Dataset`.

```python
import tensorflow as tf, itertools, tqdm

def df_to_tfr_iter(tbl, batch=200_000):
    for i in range(0, tbl.height, batch):
        chunk = tbl.slice(i, batch)
        yield dict(chunk.to_arrow().to_pydict())

def write_tfr(split, tbl):
    outdir = f"/content/tfr/{split}"
    os.makedirs(outdir, exist_ok=True)
    for shard_id, recs in enumerate(df_to_tfr_iter(tbl)):
        fn = f"{outdir}/{split}-{shard_id:05d}.tfr"
        with tf.io.TFRecordWriter(fn, compression_type="GZIP") as w:
            for j in range(len(recs["target_amount"])):
                feat = {
                    k: (tf.train.Feature(
                            float_list=tf.train.FloatList(value=[recs[k][j]]))
                         if isinstance(recs[k][j], float)
                         else tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[str(recs[k][j]).encode()])))
                    for k in recs
                }
                w.write(tf.train.Example(
                        features=tf.train.Features(feature=feat)
                     ).SerializeToString())
```

Write both splits **once**, expect ≈ 12 min.

---

### 6 Input pipeline (mixed-precision ready, hash buckets)

```python
FEATURES = {
    # floats
    **{c: tf.io.FixedLenFeature([], tf.float32) for c in
       ["trip_miles","trip_time","mph","base_passenger_fare","tolls",
        "bcf","sales_tax","congestion_surcharge","airport_fee"]},
    # ints
    **{c: tf.io.FixedLenFeature([], tf.int64) for c in
       ["pickup_hour","pickup_wday","pickup_month"]},
    # strings (to be hashed)
    **{c: tf.io.FixedLenFeature([], tf.string) for c in
       ["hvfhs_license_num","dispatching_base_num",
        "PULocationID","DOLocationID",
        "shared_request_flag","shared_match_flag",
        "wav_request_flag","access_a_ride_flag"]},
    "target_amount": tf.io.FixedLenFeature([], tf.float32),
}

def parse(ex):
    return tf.io.parse_single_example(ex, FEATURES)

def ds_from_tfr(split, batch, shuffle=False):
    files = tf.io.gfile.glob(f"/content/tfr/{split}/*.tfr")
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(1_000_000)
    return (ds.batch(batch, drop_remainder=True)
             .prefetch(tf.data.AUTOTUNE))
```

---

### 7 Model – **Deep & Cross v2** with a 6-layer ReLU tower

```python
from tensorflow.keras import layers as L, regularizers, Model

HASH_BUCKETS = 2000
EMB_DIM      = 8
L2_REG       = 1e-5

inputs = {k: L.Input(shape=(), name=k,
                     dtype=tf.string if FEATURES[k].dtype==tf.string else tf.float32)
          for k in FEATURES if k != "target_amount"}

def hash_embed(feat):
    idx = tf.strings.to_hash_bucket_fast(inputs[feat], HASH_BUCKETS)
    emb = L.Embedding(HASH_BUCKETS, EMB_DIM)(idx)
    return L.Flatten()(emb)

embed_out = [hash_embed(f) for f in
             ["hvfhs_license_num","dispatching_base_num",
              "PULocationID","DOLocationID"]]

# 0/1 flags
flags = [L.Cast(dtype='float32')(inputs[f])
         for f in ["shared_request_flag","shared_match_flag",
                   "wav_request_flag","access_a_ride_flag"]]

# Numeric norm
num_cols = ["trip_miles","trip_time","mph","base_passenger_fare",
            "tolls","bcf","sales_tax","congestion_surcharge","airport_fee"]
norm = L.Normalization()
norm.adapt(train_df.select(num_cols).to_numpy())  # offline
num_out = norm(L.Concatenate()([inputs[c] for c in num_cols]))

# Cyclic time
hour = tf.cast(inputs["pickup_hour"], tf.float32)
sin_hour = tf.sin(2*tf.constant(math.pi)*hour/24)
cos_hour = tf.cos(2*tf.constant(math.pi)*hour/24)

concat = L.Concatenate()(embed_out + flags + [num_out, sin_hour, cos_hour])

# --- Cross stack (3 layers) ---
cross = concat
for _ in range(3):
    cross = tf.keras.experimental.LinearCombination()([concat, cross])

# --- Deep tower (6 layers, ReLU) ---
x = concat
for units, drop in [(1024,0.30),(512,0.30),(256,0.25),
                    (128,0.20),(64,0.10),(32,0.10)]:
    x = L.Dense(units, activation='relu',
                kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = L.BatchNormalization()(x)
    if drop: x = L.Dropout(drop)(x)

fused = L.Concatenate()([cross, x])
out   = L.Dense(1, name='fare')(fused)

model = Model(inputs, out)
```

---

### 8 Compile & train (full dataset, SGD + momentum)

```python
# LR schedule: linear warm-up 5 epochs → cosine decay to 2 %
steps_per_epoch = math.ceil(len(train_df) / 16384)

lr_sched = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.04,        # scaled for 16 384 batch
    decay_steps=15*steps_per_epoch,
    alpha=0.02)

optim = tf.keras.optimizers.SGD(
    learning_rate=lr_sched,
    momentum=0.9, nesterov=True)

model.compile(
    optimizer=optim,
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='MAE'),
             tf.keras.metrics.MeanAbsolutePercentageError(name='MAPE')],
    run_eagerly=False)

# Gradient clip
@tf.function
def train_step(data):
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = model.compiled_loss(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    model.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in model.metrics}
model.train_step = train_step
```

```python
BATCH = 16_384
train_ds = ds_from_tfr('train', BATCH, shuffle=True)
valid_ds = ds_from_tfr('valid', BATCH)

cb = [
    tf.keras.callbacks.EarlyStopping(monitor='val_MAE',
                                     patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/dcnv2_best.keras',
        monitor='val_MAE', save_best_only=True),
    tf.keras.callbacks.TensorBoard('logs', update_freq=200)   # every 200 steps
]

history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=15,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=math.ceil(len(valid_df)/BATCH),
                    callbacks=cb)
```

**Success thresholds**  
* `val_MAE ≤ $2.00` (primary)  
* `val_MAPE ≤ 12 %`, P95 abs. error ≤ \$8  
* Train : valid MAE ratio < 1.15

Training will stop early if `val_MAE` fails to improve for two consecutive epochs.

---

### 9 Evaluate & residuals

```python
import numpy as np, matplotlib.pyplot as plt
pred = model.predict(valid_ds, verbose=0).flatten()
truth = valid_df["target_amount"].to_numpy()

print("final MAE $", np.mean(np.abs(pred-truth)))
p95 = np.percentile(np.abs(pred-truth), 95)
print("P95 abs. error $", p95)

plt.hist(truth-pred, bins=100)
plt.title("Residuals ($ actual – predicted)")
plt.show()
```

---

### 10 Save & export

```python
model.save("dcnv2_sgd_savedmodel")
!zip -r model.zip dcnv2_sgd_savedmodel -q
```
Deliver `model.zip` to production (or ML ops pipeline).

---

## Quick checklist for a new DS

1. **Switch Colab runtime to TPU v2-8**.  
2. **Copy the outline + code cells** above into `DNN.ipynb`.  
3. **Run cells sequentially**; TFRecord writing (step 5) is one-time.  
4. **Monitor TensorBoard** (`%tensorboard --logdir logs`) – watch MAE & P95 curves.  
5. **Stop early** once `val_MAE ≤ $2`, then export the model.
