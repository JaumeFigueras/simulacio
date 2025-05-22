import datetime
import pandas as pd

dfs = list()
for i in range(12):
    df = pd.read_parquet(f"yellow_tripdata_2024-{i+1:02d}.parquet")
    df = df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]]
    dfs.append(df)
    print(f"Read parquet {i}")
df = pd.concat(dfs)
# rename
df.rename(columns={'tpep_pickup_datetime': 'arrival_time'}, inplace=True)
df.rename(columns={'tpep_dropoff_datetime': 'finish_time'}, inplace=True)
df = df[['arrival_time', 'finish_time']]
print("removed columns")
df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df['finish_time'] = pd.to_datetime(df['finish_time'])
# Remove strange data, probably time zones
df = df[~(df['arrival_time'] < datetime.datetime(2024,1,1,0,0,0))]
df = df[~(df['finish_time'] < datetime.datetime(2024,1,1,0,0,0))]
# Remove holidays
week_holidays = [datetime.datetime(2024,1,1,0,0,0),
    datetime.datetime(2024,1,15,0,0,0),
    datetime.datetime(2024,2,19,0,0,0),
    datetime.datetime(2024,5,27,0,0,0),
    datetime.datetime(2024,6,19,0,0,0),
    datetime.datetime(2024,7,4,0,0,0),
    datetime.datetime(2024,9,2,0,0,0),
    datetime.datetime(2024,10,14,0,0,0),
    datetime.datetime(2024,11,5,0,0,0),
    datetime.datetime(2024,11,11,0,0,0),
    datetime.datetime(2024,11,28,0,0,0),
    datetime.datetime(2024,12,25,0,0,0)
]
for i, holiday in enumerate(week_holidays):
    df = df[~((df['arrival_time'] >= holiday) & (df['arrival_time'] < holiday + datetime.timedelta(days=1)))]
    print(f"removed holiday {i}")
df.sort_values("arrival_time", inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv("taxi-nyc-2024.csv")
