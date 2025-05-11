# ───────────────── 50-M-row DCNv2 training on v5p-16 ───────────────
import os, random, math, time, polars as pl, numpy as np, tensorflow as tf

PARQ = "/mnt/disks/nvme/fhvhv_clean.zstd.parquet"  # 20 GB file
N_ALL = 745_287_023
N_USE = 50_000_000                               # 50 million rows
BATCH_PER_CORE = 2048                            # => 32 768 global

offset = random.randrange(0, N_ALL - N_USE)

t0 = time.time()
df = (pl.scan_parquet(PARQ)
        .slice(offset, N_USE)
        .collect(streaming=True))
print("Loaded", df.shape, "in", round(time.time() - t0, 1), "s")

# ─── feature engineering ───────────────────────────────────────────
NUM = ["trip_miles","trip_time","base_passenger_fare","tolls","bcf",
       "sales_tax","congestion_surcharge","airport_fee","target_amount"]
BIN = ["shared_request_flag","shared_match_flag",
       "wav_request_flag","access_a_ride_flag"]
CAT = ["hvfhs_license_num","dispatching_base_num",
       "PULocationID","DOLocationID"]

df = (df.with_columns([pl.col(c).clip_max(400) for c in NUM[:-1]])
        .with_columns([
            pl.when(pl.col("trip_time") > 0)
              .then(pl.col("trip_miles") / (pl.col("trip_time")/3600))
              .otherwise(0.0).alias("mph"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_wday"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month")])
        .with_columns([pl.col(NUM+["mph"]).fill_null(0.0).cast(pl.Float32)])
        .with_columns([pl.col(CAT+BIN).fill_null("UNK")])
        .sample(fraction=1.0, seed=7))

cut = int(len(df)*0.8)
train_df, valid_df = df.slice(0, cut), df.slice(cut)
print("Train", len(train_df), " Valid", len(valid_df))

def to_dict(tbl: pl.DataFrame):
    out = {}
    for c in tbl.columns:
        if c == "target_amount": continue
        col = tbl[c]
        out[c] = (np.nan_to_num(col.to_numpy(), nan=0.0).astype("float32")
                  if col.dtype.is_numeric()
                  else col.to_numpy().astype(str))
    return out

x_tr, x_va = to_dict(train_df), to_dict(valid_df)
y_tr = train_df["target_amount"].to_numpy().astype("float32")
y_va = valid_df["target_amount"].to_numpy().astype("float32")

cores = int(os.environ.get("TPU_NUM_DEVICES", "1"))  # = 16
GLOBAL_BS = cores * BATCH_PER_CORE                   # 32 768
print(f"TPU cores {cores} | global batch {GLOBAL_BS}")

tr_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
         .shuffle(5_000_000).batch(GLOBAL_BS).prefetch(tf.data.AUTOTUNE))
va_ds = (tf.data.Dataset.from_tensor_slices((x_va, y_va))
         .batch(GLOBAL_BS).prefetch(tf.data.AUTOTUNE))

# ─── model (Deep & Cross) ──────────────────────────────────────────
from tensorflow.keras import layers as L, regularizers, Model, backend as K
NUM_FEATS = ["trip_miles","trip_time","base_passenger_fare","tolls","bcf",
             "sales_tax","congestion_surcharge","airport_fee","mph"]
CAL = ["pickup_hour","pickup_wday","pickup_month"]
inp = {c: L.Input(shape=(), dtype=('string' if c in CAT+BIN else 'float32'), name=c)
       for c in NUM_FEATS+CAT+BIN+CAL}

def emb(x, buckets=2000, dim=8):
    idx = tf.strings.to_hash_bucket_fast(x, buckets)
    return L.Flatten()(L.Embedding(buckets, dim)(idx))
embs = [emb(inp[c]) for c in CAT]
flags = [tf.expand_dims(tf.cast(inp[c] == 'Y', tf.float32), -1) for c in BIN]
num_vec = L.Concatenate()([tf.expand_dims(inp[c], -1) for c in NUM_FEATS])

hr = tf.cast(inp["pickup_hour"], tf.float32)
sin_hr = tf.expand_dims(tf.sin(2*math.pi*hr/24), -1)
cos_hr = tf.expand_dims(tf.cos(2*math.pi*hr/24), -1)
base = L.Concatenate()(embs + flags + [num_vec, sin_hr, cos_hr])

class Cross(L.Layer):
    def build(self, shp):
        self.w = self.add_weight("w", shape=(shp[-1], 1))
        self.b = self.add_weight("b", shape=(shp[-1],))
    def call(self, x0, xi):
        return x0 + K.dot(xi, self.w) + self.b

cross = base
for _ in range(3):
    cross = Cross()(base, cross)

deep = base
for u, d in [(1024,0.3),(512,0.3),(256,0.25),(128,0.2),(64,0.1),(32,0.1)]:
    deep = L.Dense(u, 'relu', kernel_regularizer=regularizers.l2(1e-5))(deep)
    deep = L.BatchNormalization()(deep)
    if d: deep = L.Dropout(d)(deep)

out = L.Dense(1)(L.Concatenate()([cross, deep]))
model = Model(inp, out)
model.compile(tf.keras.optimizers.Adam(0.001), loss='mae', metrics=['mae','mape'])

steps  = math.ceil(len(train_df) / GLOBAL_BS)
vsteps = math.ceil(len(valid_df) / GLOBAL_BS)

model.fit(tr_ds, validation_data=va_ds,
          epochs=2, steps_per_epoch=steps, validation_steps=vsteps)

model.save("/mnt/disks/nvme/fhv_dcnv2_b50m")
print("✅  Training done & model saved.")
