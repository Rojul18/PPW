"""Pipeline UTS PPW: crawling -> preprocessing -> skip-gram -> Naive Bayes."""
import os
import re
import time
import random
from urllib.parse import urljoin

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

BASE = "https://www.antaranews.com"
KATEGORI_DEFAULT = ["politik", "ekonomi", "olahraga", "tekno", "hiburan"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36"
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "berita.csv")

STOP = set(StopWordRemoverFactory().get_stop_words()) | {
    "antara", "jakarta", "surabaya", "kata", "ujar", "ujarnya", "katanya",
    "yang", "dan", "juga", "akan", "pada", "untuk", "dengan", "dari",
}


# ---------------------------------------------------------------- 1. crawling
def _soup(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def _link_kategori(kat, target, maks_halaman=15):
    links = []
    for page in range(1, maks_halaman + 1):
        url = f"{BASE}/{kat}" if page == 1 else f"{BASE}/{kat}/{page}"
        try:
            soup = _soup(url)
        except Exception:
            break
        for a in soup.select("a[href*='/berita/']"):
            href = urljoin(BASE, a.get("href", "")).split("?")[0]
            if re.search(r"/berita/\d+/", href) and href not in links:
                links.append(href)
        if len(links) >= target:
            break
        time.sleep(random.uniform(0.4, 1.0))
    return links[:target]


def _artikel(url):
    soup = _soup(url)
    judul = soup.find("h1")
    isi = soup.select_one("div.post-content") or soup.select_one("article")
    if not judul or not isi:
        return None
    for t in isi.select("script, style, .baca-juga, .text-muted"):
        t.decompose()
    teks = " ".join(p.get_text(" ", strip=True) for p in isi.find_all("p"))
    if len(teks.split()) < 30:
        return None
    return {"judul": judul.get_text(strip=True), "isi": teks, "url": url}


def crawl(kategori=None, per_kategori=60, progress=None):
    """Crawl berita Antara News. `progress(msg, frac)` opsional untuk UI."""
    kategori = kategori or KATEGORI_DEFAULT
    rows = []
    for i, kat in enumerate(kategori):
        links = _link_kategori(kat, per_kategori + 15)
        n = 0
        for u in links:
            if n >= per_kategori:
                break
            try:
                art = _artikel(u)
            except Exception:
                continue
            if art:
                art["kategori"] = kat
                rows.append(art)
                n += 1
            time.sleep(random.uniform(0.3, 0.8))
            if progress:
                progress(f"{kat}: {n}/{per_kategori}", (i + n / per_kategori) / len(kategori))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="url").reset_index(drop=True)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df


def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None


# ------------------------------------------------------------ 2. preprocessing
def preprocess(teks):
    teks = str(teks).lower()
    teks = re.sub(r"^.*?\(antara\)\s*-?\s*", "", teks)
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return [t for t in teks.split() if t not in STOP and len(t) > 2]


def preprocess_df(df):
    df = df.copy()
    df["tokens"] = (df["judul"].fillna("") + ". " + df["isi"].fillna("")).apply(preprocess)
    return df[df["tokens"].str.len() >= 10].reset_index(drop=True)


# ------------------------------------------------------- 3. modeling (skip-gram)
def doc_vector(tokens, w2v):
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)


def train(df, vector_size=100, window=5, epochs=30, test_size=0.2, seed=42):
    df = preprocess_df(df)
    tr, te = train_test_split(df.index, test_size=test_size, random_state=seed,
                              stratify=df["kategori"])
    w2v = Word2Vec(sentences=df.loc[tr, "tokens"].tolist(), vector_size=vector_size,
                   window=window, min_count=2, sg=1, negative=10, epochs=epochs,
                   workers=2, seed=seed)
    Xtr = np.vstack([doc_vector(t, w2v) for t in df.loc[tr, "tokens"]])
    Xte = np.vstack([doc_vector(t, w2v) for t in df.loc[te, "tokens"]])
    ytr, yte = df.loc[tr, "kategori"].values, df.loc[te, "kategori"].values

    # 4. klasifikasi: Gaussian NB (vektor skip-gram bisa negatif)
    nb = GaussianNB().fit(Xtr, ytr)
    pred = nb.predict(Xte)
    labels = sorted(df["kategori"].unique())
    hasil = {
        "akurasi": accuracy_score(yte, pred),
        "report": classification_report(yte, pred, output_dict=True, zero_division=0),
        "cm": confusion_matrix(yte, pred, labels=labels),
        "labels": labels,
        "n_train": len(tr), "n_test": len(te), "vocab": len(w2v.wv),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    w2v.save(os.path.join(MODEL_DIR, "w2v.model"))
    joblib.dump({"nb": nb, "hasil": hasil}, os.path.join(MODEL_DIR, "nb.joblib"))
    return w2v, nb, hasil


def load_model():
    p1, p2 = os.path.join(MODEL_DIR, "w2v.model"), os.path.join(MODEL_DIR, "nb.joblib")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return None
    d = joblib.load(p2)
    return Word2Vec.load(p1), d["nb"], d["hasil"]


def predict(teks, w2v, nb):
    tokens = preprocess(teks)
    if not any(t in w2v.wv for t in tokens):
        return None, {}
    v = doc_vector(tokens, w2v).reshape(1, -1)
    proba = nb.predict_proba(v)[0]
    return nb.classes_[int(np.argmax(proba))], dict(zip(nb.classes_, proba))
