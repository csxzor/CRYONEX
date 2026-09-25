"""Cut demo samples from the raw datasets, in their original formats.

**Every sample lies outside the model's training span** (P1 training = the first 60% of each
capture session). Samples may start in the calibration span, which is never trained on,
because the console needs ~26 minutes of history (15 min warm-up + 11 min context) before
it forecasts. The console ingests them exactly like a user upload.

| file | real events inside |
|---|---|
| cic17_friday_scan_ddos.csv.gz | port-scan restart 18:21, DDoS 18:56 (UTC) |
| cic17_wednesday_heartbleed.csv.gz | Heartbleed exploit 18:12 (early warning ~5 min before, release model) |
| dapt_friday_exfiltration.csv.gz | data exfiltration 20:33, 20:40 |
| dapt_wednesday_foothold.csv.gz | repeated foothold attempts from 21:08 |
| ctu13_s04_c2_ddos.binetflow.gz | bot C2 and DDoS from 14:27 |
"""

from __future__ import annotations

import datetime as dt
import gzip

import polars as pl

from kcwm import config
from kcwm.ingest.flowcsv import _parse_timestamp

OUT = config.REPO_ROOT / "samples"


def _utc(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC).timestamp()


def slice_csv(src, start: str, end: str, name: str, *, ts_col: str, header=None) -> None:
    df = pl.read_csv(src, infer_schema=False, has_header=header is None,
                     new_columns=header)
    ts = df.select(_parse_timestamp(ts_col).alias("_t"))["_t"]
    cut = df.filter(ts.is_between(_utc(start), _utc(end)))
    with gzip.open(OUT / name, "wb") as fh:
        cut.write_csv(fh)
    print(f"{name}: {cut.height:,} flows, {(OUT / name).stat().st_size / 1e6:.1f} MB")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    cic = config.dataset_dir("cicids2017")
    slice_csv(cic / "friday.csv", "2017-07-07 17:30", "2017-07-07 19:30",
              "cic17_friday_scan_ddos.csv.gz", ts_col="Timestamp")
    slice_csv(cic / "wednesday.csv", "2017-07-05 17:30", "2017-07-05 18:40",
              "cic17_wednesday_heartbleed.csv.gz", ts_col="Timestamp")
    dapt = config.dataset_dir("dapt2020") / "csv"
    slice_csv(dapt / "enp0s3-tcpdump-friday.pcap_Flow.csv", "2019-07-19 19:40", "2019-07-19 21:10",
              "dapt_friday_exfiltration.csv.gz", ts_col="Timestamp")
    slice_csv(dapt / "enp0s3-public-wednesday.pcap_Flow.csv", "2019-07-17 20:40", "2019-07-17 22:30",
              "dapt_wednesday_foothold.csv.gz", ts_col="Timestamp")
    ctu = sorted((config.dataset_dir("ctu13") / "4").glob("*.binetflow"))[0]
    slice_csv(ctu, "2011-08-15 13:45", "2011-08-15 15:11", "ctu13_s04_c2_ddos.binetflow.gz",
              ts_col="StartTime")


if __name__ == "__main__":
    main()
