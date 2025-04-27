# NYC FHV 2019-2022 ― Project Status Brief  


## 1  Raw data ingestion & storage history
| Stage | What we did | Key output |
|-------|-------------|------------|
| **Kaggle → Drive download** | Pulled 46 monthly Parquet shards (≈ 20 GB) from *jeffsinsel/nyc-fhvhv-data* into Drive at `/MyDrive/datasets/nyc_taxi/…/versions/4/` | 46 Parquet files |
| **Schema resolution** | Found columns missing/inconsistent (`wav_match_flag` null ↔ str). Patched on-the-fly with Polars (`Utf8` cast or null fill). | consistent schemas |
| **One-shot merge** | Used **Polars `scan → sink_parquet`** (streaming) to concatenate all 46 shards without ballooning RAM. | `fhvhv_all_years.zstd.parquet` (20.53 GB) |
| **Footer validation** | Verified file metadata; rebuilt once after an interrupted write that left the footer incomplete. | healthy Parquet |
| **Long-term storage** | Created GCS bucket `gs://nyc-taxi-fhv-460946772036` in project **`nyc-taxi-ml`** (us-central1).  Uploaded the merged Parquet with `gsutil -m cp` (composite upload). | bucket object size ≈ 20.5 GB |
| **Colab mount setup** | `gcsfuse --implicit-dirs nyc-taxi-fhv-460946772036 /mnt/fhv` (≈ 10 s). | fast (250–400 MB/s) access |

---

## 2  Dataset snapshot (post-merge, **pre-cleaning**)
*Rows:* **745 287 023**  
*Columns (24):*
| Name | Type | Nulls |
|------|------|-------|
| hvfhs_license_num | str | 0 |
| dispatching_base_num | str | 2 085 |
| originating_base_num | str | 204 294 792 |
| request_datetime | datetime[ns] | 108 958 |
| on_scene_datetime | datetime[ns] | 207 730 776 |
| pickup_datetime | datetime[ns] | 0 |
| dropoff_datetime | datetime[ns] | 0 |
| PULocationID / DOLocationID | int64 | 0 |
| trip_miles | float64 | 0 |
| trip_time (sec) | int64 | 0 |
| base_passenger_fare | float64 | 0 |
| tolls | float64 | 0 |
| bcf (black-car fund) | float64 | 0 |
| sales_tax | float64 | 0 |
| congestion_surcharge | float64 | 51 304 041 |
| airport_fee | float64 | 414 435 |
| tips | float64 | 0 |
| driver_pay | float64 | 0 |
| shared_request_flag / shared_match_flag / access_a_ride_flag / wav_request_flag | str | 0 |
| wav_match_flag | str | 78 037 796 |

---

## 3  Data-quality observations & planned cleaning
* **Outliers / bad rows**  
  * `trip_miles` > 200 mi (max 1 310 mi) ⇒ drop  
  * `trip_time` ≤ 60 s or > 4 h ⇒ drop  
  * Negative money values (fare, surcharges, driver_pay) ⇒ drop or clip to 0  
* **Sparse columns**  
  * `originating_base_num`, `on_scene_datetime`, `wav_match_flag` (handle with **UNK/NULL** category or ignore).  
* **Zone coverage**  
  * After merge: all 265 TLC taxi zones present.  
  * Strategy: keep only zones with **≥ 300** trips (~99 % rows retained).  
* **Target variable** (pre-tip total presented to rider)  
```text
target_amount = base_passenger_fare
              + tolls
              + bcf
              + sales_tax
              + congestion_surcharge
              + airport_fee
```  
  * Excludes `tips` (optional).  
  * Will be cast to `float32` for modeling.

---

## 4  Feature-engineering blueprint (TPU-ready)

| Group | Features | Encoding |
|-------|----------|----------|
| **Numeric (8)** | trip_miles, trip_time, money components | Z-score; `Normalization` layer |
| **Categorical (4 high-card)** | hvfhs_license_num (≈4), dispatching_base_num (≈500), PULocationID (≈260), DOLocationID (≈260) | `StringLookup` → `Embedding`; dims 8-32 |
| **Boolean flags (4)** | shared_request_flag, shared_match_flag, wav_request_flag, access_a_ride_flag | 0/1 |
| **Temporal (3)** | pickup *hour*, *weekday*, *month* | cyclic (`sin`/`cos`) or small embedding |
| **Derived** | mph = trip_miles / (trip_time/3600) | numeric |

---

## 5  Pre-training pipeline status
1. **Cleaning & target creation code** (Polars) – tested, runtime ~30 s.  
2. **TFRecord sharding**  
   * 200 k rows / shard (≈35 MB) → ~3 500 shards.  
   * Write speed on Colab SSD ≈ 180 MB/s ⇒ ~12 min.  
3. **TPU v2-8 dataloader**  
   * `batch_size = 16 384` (2048 × 8 replicas).  
   * Throughput ≈ 50 k samples/s.  
4. **Baseline DNN architecture** (wide-and-deep) is scripted and ready inside a `strategy.scope()`; compiles with Adam 1e-3, MAE/MAPE metrics.

---

## 6  Hardware & runtime notes
* **Runtime of choice:** Colab Pro TPU v2-8 + 336 GB RAM.  
* **Full Parquet ➜ RAM load:** 7–10 s from GCS.  
* **Epoch time (700 M rows, batch = 16 384):** ~3.5 h.  
  Prototype on 1–5 % before full sweep.  
* **Storage costs:** GCS Standard 20 GB ≈ \$0.40 / month; reads free inside Colab.

---

## 7  Next steps for the DS teammate
1. **Mount bucket & eager-load**  
   ```python
   !gcsfuse --implicit-dirs nyc-taxi-fhv-460946772036 /mnt/fhv
   df = pl.read_parquet('/mnt/fhv/fhvhv_all_years.zstd.parquet', low_memory=False)
   ```
2. **Run `clean_and_target()` script** (provided).  
3. **Shard to TFRecords** (or Petastorm/Arrow).  
4. **Launch TPU training**; start with 5 M-row subset → verify loss curves.  
5. **Hyper-parameter sweep**  
   * Embedding dims, hidden widths, LR schedule, dropout 0.1-0.4.  
6. **Monitor** MAE vs median fare (~\$15) & MAPE; watch for over-fit on high-fare tail.  
7. **Optional experiments**  
   * Gradient-boosted trees (XGBoost, LightGBM) as a reference.  
   * Add weather or demand indices via join if desired.
