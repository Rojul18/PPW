import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Klasifikasi Berita — Skip-gram + Naive Bayes", layout="wide")
st.title("Klasifikasi Berita Indonesia")
st.caption("UTS PPW · Word2Vec Skip-gram + Gaussian Naive Bayes")

if "df" not in st.session_state:
    st.session_state.df = pl.load_data()
if "model" not in st.session_state:
    st.session_state.model = pl.load_model()

tab1, tab2, tab3, tab4 = st.tabs(["1 · Crawling", "2 · Preprocessing", "3 · Modeling", "4 · Klasifikasi"])

# ------------------------------------------------------------------ crawling
with tab1:
    st.subheader("Pengambilan data berita (Antara News)")
    c1, c2 = st.columns(2)
    kategori = c1.multiselect("Kategori", pl.KATEGORI_DEFAULT, default=pl.KATEGORI_DEFAULT)
    jumlah = c2.slider("Artikel per kategori", 10, 150, 50, step=10)
    if st.button("Mulai crawling", type="primary"):
        bar, msg = st.progress(0.0), st.empty()
        df = pl.crawl(kategori, jumlah, progress=lambda m, f: (bar.progress(min(f, 1.0)), msg.text(m)))
        if df.empty:
            st.error("Tidak ada artikel yang berhasil diambil (situs memblokir atau struktur berubah). "
                     "Gunakan upload CSV di bawah.")
        else:
            st.session_state.df = df
            st.success(f"Berhasil mengambil {len(df)} artikel.")
    up = st.file_uploader("Atau upload CSV (kolom: judul, isi, kategori)", type="csv")
    if up is not None:
        d = pd.read_csv(up)
        if {"judul", "isi", "kategori"} <= set(d.columns):
            st.session_state.df = d
            st.success(f"CSV dimuat: {len(d)} baris.")
        else:
            st.error("CSV harus punya kolom judul, isi, kategori.")
    df = st.session_state.df
    if df is not None:
        st.dataframe(df[["kategori", "judul", "isi"]].head(50), use_container_width=True)
        st.bar_chart(df["kategori"].value_counts())

# ------------------------------------------------------------- preprocessing
with tab2:
    st.subheader("Preprocessing")
    st.markdown("Case folding → buang dateline/URL/angka/tanda baca → tokenisasi → stopword removal.")
    df = st.session_state.df
    if df is None:
        st.info("Ambil data dulu di tab Crawling.")
    else:
        i = st.number_input("Indeks artikel", 0, len(df) - 1, 0)
        row = df.iloc[int(i)]
        a, b = st.columns(2)
        a.markdown("**Teks asli**"); a.write(str(row["isi"])[:900])
        b.markdown("**Token hasil preprocessing**"); b.write(pl.preprocess(row["judul"] + ". " + str(row["isi"]))[:120])

# ------------------------------------------------------------------ modeling
with tab3:
    st.subheader("Training Skip-gram + Naive Bayes")
    df = st.session_state.df
    if df is None:
        st.info("Ambil data dulu di tab Crawling.")
    else:
        c1, c2, c3 = st.columns(3)
        vs = c1.select_slider("Dimensi vektor", [50, 100, 200, 300], value=100)
        win = c2.slider("Window", 2, 10, 5)
        ep = c3.slider("Epoch", 5, 60, 30)
        if st.button("Latih model", type="primary"):
            with st.spinner("Melatih Word2Vec skip-gram dan Naive Bayes..."):
                st.session_state.model = pl.train(df, vs, win, ep)
            st.success("Model selesai dilatih.")
    m = st.session_state.model
    if m:
        _, _, h = m
        st.metric("Akurasi (data uji)", f"{h['akurasi']*100:.2f}%")
        st.caption(f"Data latih {h['n_train']} · uji {h['n_test']} · vocab {h['vocab']}")
        rep = pd.DataFrame(h["report"]).T.round(3)
        st.dataframe(rep, use_container_width=True)
        st.markdown("**Confusion matrix**")
        st.dataframe(pd.DataFrame(h["cm"], index=[f"aktual {l}" for l in h["labels"]],
                                  columns=[f"prediksi {l}" for l in h["labels"]]),
                     use_container_width=True)

# -------------------------------------------------------------- klasifikasi
with tab4:
    st.subheader("Klasifikasikan berita baru")
    m = st.session_state.model
    if not m:
        st.info("Latih model dulu di tab Modeling.")
    else:
        w2v, nb, _ = m
        teks = st.text_area("Tempel teks berita", height=220,
                            placeholder="Contoh: Timnas Indonesia menang 2-0 atas Thailand ...")
        if st.button("Prediksi", type="primary") and teks.strip():
            label, proba = pl.predict(teks, w2v, nb)
            if label is None:
                st.warning("Kata-kata pada teks tidak dikenal model. Coba teks yang lebih panjang.")
            else:
                st.success(f"Kategori: **{label}**")
                st.bar_chart(pd.Series(proba).sort_values(ascending=False))
