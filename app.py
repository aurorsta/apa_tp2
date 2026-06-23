"""
Restaurant Recommender — minimal Streamlit deployment.

Two modes:
  1. "By restaurant" — pick a restaurant you like, get similar ones (content+geo similarity).
  2. "By filters" — pick a cuisine and price range, get matching restaurants ranked by
     average rating (no anchor restaurant needed — handles the cold-start case).

Run locally with:
    streamlit run app.py
"""

import pickle

import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifact(path: str = "recommender_artifact.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def recommend_by_restaurant(place_id, artifact: dict, top_n: int = 10) -> pd.DataFrame:
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


def recommend_by_filters(cuisines, price, artifact: dict, top_n: int = 10) -> pd.DataFrame:
    places_full = artifact["places_full"]
    candidates = places_full.copy()

    if cuisines:
        pattern = "|".join(cuisines)
        candidates = candidates[candidates["cuisine"].str.contains(pattern, case=False, na=False)]

    if price != "Any":
        candidates = candidates[candidates["price"] == price]

    # Rank by average rating where available; unrated restaurants fall to the bottom
    # rather than being excluded, so new/unrated places still show up.
    candidates = candidates.sort_values("avg_rating", ascending=False, na_position="last")
    return candidates.head(top_n)


st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Recommender")
st.caption(
    "Content-based recommender (cosine similarity over cuisine, payment, hours, ambience, "
    "and geographic proximity) — the model that won this dataset's evaluation."
)

artifact = load_artifact()
places_full = artifact["places_full"]

mode = st.radio("How would you like to search?", ["By cuisine & price", "By a restaurant I like"])

if mode == "By cuisine & price":
    all_cuisines = sorted(
        {c.strip() for entry in places_full["cuisine"].dropna() for c in entry.split(",")}
        - {"Unknown"}
    )
    selected_cuisines = st.multiselect("Cuisine type(s):", all_cuisines)

    price_options = ["Any"] + sorted(places_full["price"].dropna().unique().tolist())
    selected_price = st.selectbox("Price class:", price_options)

    top_n = st.slider("How many results?", min_value=3, max_value=20, value=10, key="filter_n")

    if st.button("Find restaurants", type="primary"):
        results = recommend_by_filters(selected_cuisines, selected_price, artifact, top_n=top_n)
        if results.empty:
            st.warning("No restaurants match that combination — try fewer filters.")
        else:
            st.subheader("Matching restaurants")
            st.dataframe(
                results[["name", "cuisine", "price", "avg_rating"]],
                use_container_width=True,
                hide_index=True,
            )

else:
    name_to_id = dict(zip(places_full["name"], places_full["placeID"]))
    selected_name = st.selectbox("Pick a restaurant you like:", sorted(name_to_id.keys()))
    top_n = st.slider("How many recommendations?", min_value=3, max_value=20, value=10, key="sim_n")

    if st.button("Recommend similar restaurants", type="primary"):
        selected_id = name_to_id[selected_name]
        results = recommend_by_restaurant(selected_id, artifact, top_n=top_n)
        st.subheader(f"Restaurants similar to **{selected_name}**")
        st.dataframe(
            results[["name", "cuisine", "price", "similarity"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    f"Blend weight (geo vs. content): alpha = {artifact['best_alpha']:.2f} "
    "(tuned on held-out Recall@10 — see Section 4.5 of the notebook)."
)