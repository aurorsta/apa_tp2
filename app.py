"""
Restaurant Recommender — minimal Streamlit deployment.

Loads the serialized content-based recommender artifact (produced by Section 6 of
TP2_Recommender_System_v8.ipynb) and lets the user pick a restaurant to get the
top-10 most similar ones, using the same content+geo blend tuned in the notebook.

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt + recommender_artifact.pkl to a GitHub repo.
    2. Go to https://share.streamlit.io, connect the repo, point it at app.py.
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifact(path: str = "recommender_artifact.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def recommend(place_id, artifact: dict, top_n: int = 10) -> pd.DataFrame:
    """Same logic as recommend_content_based_with_alpha in the notebook, but reading
    from the serialized artifact instead of notebook globals."""
    content_sim = artifact["content_sim"]
    geo_sim = artifact["geo_sim"]
    place_idx = artifact["place_idx"]
    places_full = artifact["places_full"]
    alpha = artifact["best_alpha"]

    sim = (1 - alpha) * content_sim + alpha * geo_sim
    idx = place_idx[place_id]
    sims = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    rows = []
    for i, score in sims:
        row = places_full.iloc[i].to_dict()
        row["similarity"] = round(float(score), 3)
        rows.append(row)
    return pd.DataFrame(rows)


st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Recommender")
st.caption(
    "Content-based recommender (cosine similarity over cuisine, payment, hours, ambience, "
    "and geographic proximity) — the model that won this dataset's evaluation."
)

artifact = load_artifact()
places_full = artifact["places_full"]

name_to_id = dict(zip(places_full["name"], places_full["placeID"]))
selected_name = st.selectbox("Pick a restaurant you like:", sorted(name_to_id.keys()))
top_n = st.slider("How many recommendations?", min_value=3, max_value=20, value=10)

if st.button("Recommend similar restaurants", type="primary"):
    selected_id = name_to_id[selected_name]
    results = recommend(selected_id, artifact, top_n=top_n)
    st.subheader(f"Restaurants similar to **{selected_name}**")
    st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    f"Blend weight (geo vs. content): alpha = {artifact['best_alpha']:.2f} "
    "(tuned on held-out Recall@10 — see Section 4.5 of the notebook)."
)
