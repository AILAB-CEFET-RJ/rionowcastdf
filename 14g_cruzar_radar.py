"""Auditoria somente leitura. Execute na raiz de rionowcastdf."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

KEY = "timestamp_utc"
FRAC = "decoded_fraction_before_quarantine"
COLS = ["timestamp", "source", "source_sha256", "status", FRAC]
REFS = ["product_version", "acquisition_evidence", "reference_evidence", "class", "x", "y"]
CLASSES = ["echo", "observed_no_echo", "no_coverage"]


def unique_hours(df, key):
    df[key] = pd.to_datetime(df[key], utc=True, errors="raise")
    t = df[key]
    if t.isna().any() or t.duplicated().any() or not (t.dt.floor("h") == t).all():
        raise ValueError(f"Horas ausentes, não horárias ou duplicadas: {key}")


def audit(a):
    f = pd.read_parquet(a.phase15, columns=[KEY, "split"])
    # Python CSV engine avoids the C parser failure seen with this manifest.
    q = pd.read_csv(a.qc, usecols=COLS, engine="python", dtype=str, keep_default_na=False)
    unique_hours(f, KEY)
    unique_hours(q, "timestamp")
    if f.empty or q.empty or f.split.isna().any() or q.status.eq("").any():
        raise ValueError("Manifesto vazio ou rótulo ausente")
    df = f.merge(q.rename(columns={"timestamp": KEY}), how="left", on=KEY, validate="one_to_one")
    df["status"] = df.status.fillna("unmatched")
    df["month_utc"] = df[KEY].dt.strftime("%Y-%m")
    ok = df[df.status.str.startswith("accepted_")].copy()
    ok[FRAC] = pd.to_numeric(ok[FRAC], errors="raise")
    if not ok[FRAC].between(0, 1).all():
        raise ValueError("Fração decodificada inválida")
    a.out.mkdir(parents=True, exist_ok=False)
    df.to_csv(a.out / "phase15_qc_join.csv", index=False)
    for groups, name in [(["split"], "impact_by_split"), (["split","month_utc"], "impact_by_month")]:
        counts = df.groupby(groups + ["status"]).size().rename("hours").reset_index()
        counts["denominator_hours"] = counts.groupby(groups).hours.transform("sum")
        counts["percent"] = 100 * counts.hours / counts.denominator_hours
        counts.to_csv(a.out / (name + ".csv"), index=False)
    stats = ok.groupby("split")[FRAC].agg(["count", "mean", "min", "median", "max"])
    stats.to_csv(a.out / "accepted_grid_support.csv")
    pool = df[(df.split == "train") & df.status.isin(["accepted_partial","accepted_full",
          "quarantine_map_review","quarantine_no_decodable_observation"])].copy()
    # One reproducible example per year and status; not a prevalence estimate.
    picks = pool.sample(frac=1, random_state=42).groupby([pool[KEY].dt.year,"status"]).head(1).sort_values(KEY)
    picks.to_csv(a.out / "train_examples.csv", index=False)
    template = picks[[KEY]].merge(pd.DataFrame({"class": CLASSES}), how="cross")
    for c in REFS:
        if c not in template: template[c] = ""
    template[[KEY,*REFS]].to_csv(a.out / "reference_template.csv", index=False)
    print(pd.crosstab(df.split,df.status).to_string())
    print("Horas fase 15:",len(f),"| sem correspondência:",sum(df.status=="unmatched"))


def probe(a):
    examples = pd.read_csv(a.audit / "train_examples.csv", dtype=str).set_index(KEY)
    refs = pd.read_csv(a.references, dtype=str, keep_default_na=False)
    if refs.empty or not {KEY,*REFS}.issubset(refs.columns):
        raise ValueError("Forneça referências documentadas usando o modelo gerado")
    records = []
    for ts, group in refs.groupby(KEY):
        source = examples.loc[ts]
        if source["split"] != "train" or set(group["class"]) != set(CLASSES):
            raise ValueError(f"{ts}: requer as três classes no mesmo frame de treino")
        if group[REFS].apply(lambda col: col.str.strip().eq("")).any().any() or group.product_version.nunique()!=1:
            raise ValueError(f"{ts}: preencher versão, evidências independentes e coordenadas")
        path = Path(source.source)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source.source_sha256:
            raise ValueError(f"Fonte diverge do manifesto: {path}")
        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            points = set()
            for row in group.to_dict("records"):
                x,y = int(row["x"]),int(row["y"])
                if not (0<=x<im.width and 0<=y<im.height) or (x,y) in points:
                    raise ValueError(f"{ts}: coordenada inválida ou repetida")
                points.add((x,y))
                records.append({**row,"rgba":str(rgba.getpixel((x,y))),"source_sha256":source.source_sha256})
    sampled = pd.DataFrame(records)
    findings = []
    for version,g in sampled.groupby("product_version"):
        colors = lambda label:set(g.loc[g["class"]==label,"rgba"])
        overlap = sorted(colors("observed_no_echo") & colors("no_coverage"))
        findings.append(dict(product_version=version,frames=int(g[KEY].nunique()),shared_rgba=overlap,
            finding="RGBA insuficiente nos pontos documentados" if overlap else "Distintos na amostra; não valida uma regra histórica"))
    a.out.mkdir(parents=True,exist_ok=False)
    sampled.to_csv(a.out / "point_evidence.csv",index=False)
    report=json.dumps(findings,indent=2,ensure_ascii=False)
    (a.out / "findings.json").write_text(report)
    print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("join"); s.add_argument("--phase15",type=Path,required=True)
    s.add_argument("--qc",type=Path,default=Path("data/radar_cache_qc_v2/qc_manifest.csv"))
    s.add_argument("--out",type=Path,required=True)
    s=sub.add_parser("probe"); s.add_argument("--audit",type=Path,required=True)
    s.add_argument("--references",type=Path,required=True); s.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); (audit if a.command=="join" else probe)(a)
