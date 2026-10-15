from __future__ import annotations
import sys
import argparse
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_favorita, chronological_split
from src.features import make_features

def main():
    p=argparse.ArgumentParser(description="Validate Favorita data and create leakage-safe features")
    p.add_argument("--raw-dir",default="data/raw"); p.add_argument("--output",default="data/processed/features.parquet")
    p.add_argument("--train-end",default="2015-12-31"); p.add_argument("--validation-end",default="2016-03-31")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    sales,aux=load_favorita(a.raw_dir)
    logging.info("rows=%d dates=%s..%s series=%d missing_sales=%d zeros=%d",len(sales),sales.date.min(),sales.date.max(),sales.groupby(["store_nbr","family"]).ngroups,sales.sales.isna().sum(),(sales.sales==0).sum())
    parts=chronological_split(sales,a.train_end,a.validation_end)
    logging.info("split rows train=%d validation=%d test=%d",len(parts.train),len(parts.validation),len(parts.test))
    feat=make_features(sales,aux); path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    try: feat.to_parquet(path,index=False)
    except ImportError: path=path.with_suffix(".csv.gz"); feat.to_csv(path,index=False,compression="gzip")
    logging.info("saved %s",path)
if __name__=="__main__": main()
