# Data

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D.
The dataset is distributed under CC BY 4.0; preserve attribution in derived outputs.

Place the original `online_retail_II.xlsx` in `data/raw/`. It is already present
in this workspace. Keep it immutable and out of Git. Later ingestion will record
its checksum and sheet names, then write normalized Parquet to `data/interim/`
and customer snapshots to `data/processed/`.

Inspect both workbook sheets before combining: check actual column names, date
ranges, overlap, missing customer IDs, cancellations, returns, and duplicates.
Do not assume the website's column names match the workbook exactly. Do not
drop identical-looking invoice lines until their meaning has been investigated.
