# Spark Structured Streaming with MLlib - Real-Time Ride-Sharing Analytics

## Overview
Real-time analytics pipeline for ride-sharing platform using Apache Spark Structured Streaming and MLlib for fare prediction and trend analysis.

## Project Structure
```
├── models/
│   ├── fare_model/
│   └── fare_trend_model_v2/
├── data_generator.py
├── task4.py
├── task5.py
├── training-dataset.csv
└── README.md
```

## Tasks

### Task 4: Real-Time Fare Prediction
- Trains LinearRegression model on `distance_km` → `fare_amount`
- Predicts fares for streaming rides in real-time
- Calculates deviation to detect anomalies

### Task 5: Time-Based Fare Trend Prediction
- Aggregates data into 5-minute windows
- Engineers temporal features: `hour_of_day`, `minute_of_hour`
- Predicts average fare trends over time windows

## How to Run

1. **Start Data Generator**
```bash
python data_generator.py
```

2. **Run Task 4**
```bash
spark-submit task4.py
```

3. **Run Task 5**
```bash
spark-submit task5.py
```

## Outputs

### Task 4: Fare Prediction
```
-------------------------------------------
Batch: 1
-------------------------------------------
+------------------------------------+---------+-----------+-----------+-----------------+-----------------+
|trip_id                             |driver_id|distance_km|fare_amount|predicted_fare   |deviation        |
+------------------------------------+---------+-----------+-----------+-----------------+-----------------+
|99c350d2-e7eb-49c1-88a6-145b0040a143|24       |6.6        |5.07       |96.52708032658639|91.45708032658638|
+------------------------------------+---------+-----------+-----------+-----------------+-----------------+

-------------------------------------------
Batch: 2
-------------------------------------------
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
|trip_id                             |driver_id|distance_km|fare_amount|predicted_fare   |deviation         |
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
|95ce9786-f27b-47d1-b097-17809ebdc36c|12       |2.22       |65.54      |97.60374588519498|32.063745885194976|
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
```

### Task 5: Fare Trend Prediction
```
-------------------------------------------
Batch: 9
-------------------------------------------
+-------------------+-------------------+------------------+-----------------------+
|window_start       |window_end         |avg_fare          |predicted_next_avg_fare|
+-------------------+-------------------+------------------+-----------------------+
|2025-10-21 23:52:00|2025-10-21 23:57:00|115.42500000000001|92.39431034482759      |
+-------------------+-------------------+------------------+-----------------------+
```

## Tech Stack
- Apache Spark (Structured Streaming + MLlib)
- Python 3.x
- LinearRegression Model
- Socket Streaming (localhost:9999)

