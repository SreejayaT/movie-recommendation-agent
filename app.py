import os
import json
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from groq import Groq

# ------------------------------------------------------------
# Config — reads from Streamlit secrets first, falls back to env vars
# ------------------------------------------------------------
def get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
OMDB_API_KEY = get_secret("OMDB_API_KEY")
REVIEW_EMBEDDINGS_PATH = "./review_embeddings.npy"
REVIEW_DATA_PATH = "./review_data.pkl"

st.set_page_config(page_title="What Should I Watch?", page_icon="🎬", layout="centered")

# ------------------------------------------------------------
# Cached resources — loaded once, not on every rerun
# ------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_review_data():
    embeddings = np.load(REVIEW_EMBEDDINGS_PATH)
    with open(REVIEW_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    return embeddings, data["documents"], data["metadatas"]

@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

embedder = load_embedder()
review_embeddings, review_documents, review_metadatas = load_review_data()
groq_client = load_groq_client()

# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
def search_reviews(query: str, k: int = 6):
    query_vec = embedder.encode([query])[0].astype(np.float32)
    norms = np.linalg.norm(review_embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    similarities = (review_embeddings @ query_vec) / (norms * query_norm + 1e-8)
    top_idx = np.argsort(similarities)[::-1][:k]

    snippets = [
        {"title": review_metadatas[i]["title"], "review_type": review_metadatas[i]["review_type"], "text": review_documents[i]}
        for i in top_idx
    ]
    formatted_text = "\n\n".join(f"[{s['title']} — {s['review_type']}] {s['text']}" for s in snippets)
    return formatted_text if formatted_text else "No matching reviews found.", snippets


def get_movie_details(title: str):
    resp = requests.get("http://www.omdbapi.com/", params={"apikey": OMDB_API_KEY, "t": title})
    data = resp.json()
    if data.get("Response") == "False":
        return f"No OMDb data found for '{title}'.", None

    text = (
        f"Title: {data.get('Title')}\n"
        f"Year: {data.get('Year')}\n"
        f"Runtime: {data.get('Runtime')}\n"
        f"Genre: {data.get('Genre')}\n"
        f"IMDb Rating: {data.get('imdbRating')}\n"
        f"Plot: {data.get('Plot')}"
    )
    card = {
        "title": data.get("Title"),
        "year": data.get("Year"),
        "runtime": data.get("Runtime"),
        "rating": data.get("imdbRating"),
        "poster": data.get("Poster"),
    }
    return text, card

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Search real critic reviews for movies/series matching a mood, theme, or style described in natural language.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Natural-language description of what the user wants"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": "Get current runtime, genre, year, poster, and IMDb rating for a specific movie/series title.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Exact or approximate title"}},
                "required": ["title"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a movie/series recommendation assistant. You recommend
what to watch based on real critic reviews, not just ratings.

Rules:
- If the user's request is vague (no mood, genre, or time constraint), ask ONE
  clarifying question before recommending anything.
- Use search_reviews to ground your picks in actual review language.
- Use get_movie_details to check runtime/year/rating before finalizing a pick
  if runtime or recency matters, or once you've settled on a specific title.
- When you recommend something, explain WHY using specific language from the
  reviews you retrieved — not a generic summary.
- If critics were split on something, say so honestly.
"""

# ------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------
def run_agent(user_message, history, max_turns=5):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]

    collected_snippets = []
    collected_cards = []

    for _ in range(max_turns):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=800,
                temperature=0,
            )
        except Exception as e:
            if "tool_use_failed" in str(e):
                fallback = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages + [{"role": "user", "content": "Answer directly in plain text without calling any tools."}],
                    max_tokens=800,
                    temperature=0,
                )
                return fallback.choices[0].message.content, collected_snippets, collected_cards
            raise

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content, collected_snippets, collected_cards

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "search_reviews":
                result_text, snippets = search_reviews(**fn_args)
                collected_snippets.extend(snippets)
            elif fn_name == "get_movie_details":
                result_text, card = get_movie_details(**fn_args)
                if card:
                    collected_cards.append(card)
            else:
                result_text = "Unknown tool."

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})

    return "Reached max turns without a final answer.", collected_snippets, collected_cards

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🎬 What Should I Watch?")
st.caption("Recommendations grounded in real critic reviews, not just star ratings.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_log" not in st.session_state:
    st.session_state.display_log = []

for entry in st.session_state.display_log:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

        if entry.get("cards"):
            cols = st.columns(len(entry["cards"]))
            for col, card in zip(cols, entry["cards"]):
                with col:
                    if card.get("poster") and card["poster"] != "N/A":
                        st.image(card["poster"], width=140)
                    st.markdown(f"**{card['title']}** ({card['year']})")
                    st.caption(f"{card['runtime']} · ⭐ {card['rating']}")

        if entry.get("snippets"):
            with st.expander(f"See {len(entry['snippets'])} reviews used for this recommendation"):
                for s in entry["snippets"]:
                    badge = "🍅" if s["review_type"].lower() == "fresh" else "🟢" if s["review_type"].lower() == "rotten" else ""
                    st.markdown(f"**{s['title']}** {badge}")
                    st.write(s["text"])
                    st.divider()

with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("What are you in the mood for?", key="user_query")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    st.session_state.display_log.append({"role": "user", "content": user_input, "cards": None, "snippets": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, snippets, cards = run_agent(user_input, st.session_state.messages)
        st.markdown(answer)

        if cards:
            cols = st.columns(len(cards))
            for col, card in zip(cols, cards):
                with col:
                    if card.get("poster") and card["poster"] != "N/A":
                        st.image(card["poster"], width=140)
                    st.markdown(f"**{card['title']}** ({card['year']})")
                    st.caption(f"{card['runtime']} · ⭐ {card['rating']}")

        if snippets:
            with st.expander(f"See {len(snippets)} reviews used for this recommendation"):
                for s in snippets:
                    badge = "🍅" if s["review_type"].lower() == "fresh" else "🟢" if s["review_type"].lower() == "rotten" else ""
                    st.markdown(f"**{s['title']}** {badge}")
                    st.write(s["text"])
                    st.divider()

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.display_log.append({"role": "assistant", "content": answer, "cards": cards, "snippets": snippets})
