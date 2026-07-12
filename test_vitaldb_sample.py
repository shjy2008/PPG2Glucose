import pandas as pd
import numpy as np
import ssl
import urllib.request
from pathlib import Path
import certifi
from vitaldb.dataset import load_trk


def patch_ssl() -> None:
	ssl_ctx = ssl.create_default_context(cafile=certifi.where())
	opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
	urllib.request.install_opener(opener)


patch_ssl()

labels_dir = Path("dataset/vitaldb_sample1/Labels")
label_file = sorted(labels_dir.glob("label_*.csv"))[0]   # sample 1 label
caseid = int(label_file.stem.split("_")[1])

tracks = pd.read_csv("https://api.vitaldb.net/trks")
row = tracks[(tracks["caseid"] == caseid) & (tracks["tname"] == "SNUADC/PLETH")]
tid = str(row.iloc[0]["tid"])

wave = np.asarray(load_trk(tid, interval=1/500.0), dtype=np.float32).reshape(-1)
out = Path("dataset/vitaldb_sample1/RawData") / f"signal_{caseid:04d}_full.csv"
pd.DataFrame({"ppg": wave}).to_csv(out, index=False, header=False)

print("caseid:", caseid)
print("saved:", out)
print("points:", wave.size)