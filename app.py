"""Streamlit explorer for the movie embedding space.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path("output/movies_2d.csv")
VECTORS = Path("output/embeddings.npy")
TOP_N_TAGS = 8          # human eye tops out around 8-10 categorical colours
OTHER = "other"

st.set_page_config(page_title="Movie Embedding Explorer", layout="wide")


@st.cache_data(show_spinner="Loading projection...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)

    # tags is a JSON array string: '["Tense", "Dark"]'.
    # str.split(",")[0] would return the literal '["Tense' -- brackets and
    # quotes included -- which is what produced the unreadable legend.
    parsed = df["tags"].fillna("[]").apply(json.loads)
    df["tag_list"] = parsed
    df["primary_tag"] = parsed.apply(lambda t: t[0] if t else "untagged")
    df["tag_count"] = parsed.apply(len)
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading vectors...")
def load_vectors() -> np.ndarray | None:
    if not VECTORS.exists():
        return None
    v = np.load(VECTORS).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def collapse_rare(series: pd.Series, keep: int) -> pd.Series:
    top = series.value_counts().nlargest(keep).index
    return series.where(series.isin(top), OTHER)


df = load_data()
vectors = load_vectors()

st.title("Movie Embedding Explorer")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Display")
    colour_options = ["primary_tag", "rating_num", "tag_count"]
    if "cluster_name" in df.columns:
        colour_options.insert(0, "cluster_name")
    colour_by = st.selectbox("Colour by", colour_options)
    point_size = st.slider("Point size", 2, 12, 5)
    opacity = st.slider("Opacity", 0.1, 1.0, 0.6)
    tag_filter = st.multiselect(
        "Show only movies tagged", sorted({t for row in df.tag_list for t in row})
    )

# ---------------------------------------------------------------- filters
search = st.text_input("Search title")

view = df
if search:
    view = view[view["title"].str.contains(search, case=False, na=False)]
if tag_filter:
    wanted = set(tag_filter)
    view = view[view["tag_list"].apply(lambda t: wanted.issubset(t))]

# .copy() -- everything above produced a slice; assigning to a slice raises
# SettingWithCopyWarning and may silently fail to write through.
view = view.copy()

st.caption(f"Showing {len(view):,} of {len(df):,} movies")
if view.empty:
    st.warning("No movies match those filters.")
    st.stop()

# --------------------------------------------------------------- plotting
if colour_by in ("primary_tag", "cluster_name"):
    view["_colour"] = collapse_rare(view[colour_by], TOP_N_TAGS)
    kwargs = {
        "color": "_colour",
        "category_orders": {
            "_colour": [c for c in view["_colour"].value_counts().index if c != OTHER]
            + ([OTHER] if (view["_colour"] == OTHER).any() else [])
        },
    }
else:
    kwargs = {"color": colour_by, "color_continuous_scale": "Viridis"}

fig = px.scatter(
    view,
    x="x",
    y="y",
    hover_name="title",
    hover_data={"rating": True, "tags": True, "x": False, "y": False},
    height=800,
    # WebGL: SVG chokes well before 16k markers.
    render_mode="webgl",
    **kwargs,
)
fig.update_traces(marker=dict(size=point_size, opacity=opacity))
fig.update_layout(
    legend_title_text=colour_by,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title=None,
    yaxis_title=None,
)
# UMAP axes carry no units; showing ticks invites over-reading the geometry.
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

# width= is ignored whenever use_container_width=True, so it is not set.
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "UMAP preserves local neighbourhoods only. Distance between clusters "
    "and cluster size are artefacts of the layout, not of the data."
)

# ------------------------------------------------------------- neighbours
if vectors is not None:
    st.subheader("Nearest neighbours")
    st.caption(
        "Random unrelated pairs in this corpus average cosine 0.37, so treat "
        "that as the floor -- not 0."
    )
    choice = st.selectbox("Pick a movie", view["title"].head(500))
    i = int(df.index[df["title"] == choice][0])
    sims = vectors @ vectors[i]
    sims[i] = -np.inf
    top = np.argsort(-sims)[:10]
    st.dataframe(
        pd.DataFrame({
            "title": df["title"].values[top],
            "cosine": np.round(sims[top], 3),
            "rating": df["rating"].values[top],
            "tags": df["tags"].values[top],
        }),
        use_container_width=True,
        hide_index=True,
    )
