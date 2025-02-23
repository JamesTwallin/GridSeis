import requests
import pandas as pd
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def download_raw_frequency_data(year, month):
    file_path = f"raw_data/fnew-{year}-{month}.parquet"
    if os.path.exists(file_path):
        logging.info(f"Raw data for {year}-{month} already exists")
        return
    logging.info(f"Downloading raw data for {year}-{month}")
    try:
        if not os.path.exists("raw_data"):
            os.makedirs("raw_data")
        
        #https://api.neso.energy/dataset/cb1cc925-ecd8-4406-b021-3a3f368196e1/resource/a1ccd82c-e522-4b5d-a4da-57dafab9d6de/download/fnew-2022-9.csv
        url = f"https://api.neso.energy/dataset/cb1cc925-ecd8-4406-b021-3a3f368196e1/resource/a1ccd82c-e522-4b5d-a4da-57dafab9d6de/download/fnew-{year}-{month}.csv"
        response = requests.get(url)
        with open(f"fnew-{year}-{month}.csv", "wb") as file:
            file.write(response.content)
        # write as parquet
        df = pd.read_csv(f"fnew-{year}-{month}.csv")
        df.to_parquet(file_path, index=False)
        os.remove(f"fnew-{year}-{month}.csv")
    except Exception as e:
        logging.error(f"Error downloading raw data for {year}-{month}: {e}")
        return None


if __name__ == "__main__":
    for date in pd.date_range("2014-01-01", "2025-02-01", freq="1MS"):
        download_raw_frequency_data(date.year, date.month)
