"""Streamlit front end (talks to the API):  streamlit run src/verified_doc_qa/ui.py"""
from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
BADGES = {"answered": ("Verified", "green"), "unverified": ("Not verified", "orange"), "refused": ("Not in documents", "gray")}

st.set_page_config(page_title="Verified Doc QA", page_icon=":material/fact_check:", layout="wide")
st.title("Verified Doc QA")

with st.sidebar:
    st.subheader("Documents")
    uploads = st.file_uploader("Add .md, .txt or .pdf", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if uploads and st.button("Index", type="primary"):
        files = [("files", (u.name, u.getvalue())) for u in uploads]
        r = requests.post(f"{API}/ingest", files=files, timeout=600)
        st.success(r.json()) if r.ok else st.error(r.text)
    try:
        docs = requests.get(f"{API}/documents", timeout=10).json()
        st.caption(f"{len(docs)} documents, {sum(docs.values())} chunks")
        for name, n in docs.items():
            st.write(f"- {name} ({n})")
    except requests.RequestException:
        st.error(f"API not reachable at {API}")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if question := st.chat_input("Ask about your documents"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Retrieving, answering, verifying..."):
        r = requests.post(f"{API}/ask", json={"question": question}, timeout=600)
        if not r.ok:
            st.error(r.text)
        else:
            resp = r.json()
            label, color = BADGES[resp["status"]]
            st.markdown(f":{color}-badge[{label}] · relevance {resp['relevance']} · "
                        f"{resp['attempts']} attempt(s) · {resp['seconds']}s")
            st.markdown(resp["answer"])
            for c in resp["citations"]:
                with st.expander(f"[{c['n']}] {c['source']}{' - ' + c['section'] if c['section'] else ''}"):
                    st.text(c["text"])
            v = resp.get("verification")
            if v and not v["passed"]:
                st.warning("Unsupported: " + "; ".join(v["unsupported_claims"] + v["contradictions"] or [v["feedback"]]))
            with st.expander("Trace"):
                st.dataframe(resp["steps"])
            st.session_state.history.append({"role": "assistant", "content": resp["answer"]})
