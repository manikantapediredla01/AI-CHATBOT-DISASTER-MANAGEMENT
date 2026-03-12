"""
app.py  –  DisasterBot (Professional Light Edition)
A high-end, clean Streamlit chatbot powered by Groq Llama 3.3 + FAISS RAG.
Designed with enterprise-grade professional aesthetics for disaster awareness.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ─── Config ───────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
INDEX_PATH     = "faiss_index"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL     = "llama-3.3-70b-versatile"
TOP_K_DOCS     = 5

SYSTEM_PROMPT = """You are DisasterBot, a professional AI assistant for public disaster preparedness.
Structure your answers clearly using headers, bullet points, and numbered lists.
Maintain a calm, authoritative, and helpful tone.
"""

QUICK_PROMPTS = [
    ("🏃", "Evacuation Steps"),
    ("⛺", "Relief Centers"),
    ("🌊", "Flood Safety"),
    ("🏚️", "Earthquake Tips"),
    ("🔥", "Fire Safety"),
    ("🩹", "First Aid Basics"),
]

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DisasterBot | Public Awareness",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Professional UI CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #1e293b;        /* Deep Slate Blue */
    --accent: #dc2626;         /* Emergency Red */
    --bg-main: #f8fafc;        /* Off White */
    --card-bg: #ffffff;
    --text-p: #0f172a;         /* Near Black */
    --text-s: #475569;         /* Slate Gray */
    --border: #e2e8f0;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

/* ── Global Styles ── */
.stApp {
    background-color: var(--bg-main) !important;
}

#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3, p, span, div {
    font-family: 'Inter', sans-serif;
    color: var(--text-p) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hero Banner ── */
.hero-container {
    background-color: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--primary) !important;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.hero-sub {
    font-size: 1.1rem;
    color: var(--text-s) !important;
    max-width: 700px;
    line-height: 1.6;
}

/* ── Status Bar ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    background: #f1f5f9;
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.75rem;
    border: 1px solid var(--border);
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: white !important;
    color: var(--primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.stButton > button:hover {
    border-color: var(--primary) !important;
    background: #f8fafc !important;
    transform: translateY(-1px);
}

/* ── Chat Flow ── */
.msg-row {
    display: flex;
    flex-direction: column;
    margin-bottom: 1.5rem;
}

.msg-bubble {
    padding: 1.25rem 1.5rem;
    border-radius: 18px;
    max-width: 80%;
    line-height: 1.6;
    font-size: 0.95rem;
}

.user-bubble {
    align-self: flex-end;
    background-color: var(--primary);
    color: white !important;
    border-bottom-right-radius: 4px;
}
.user-bubble * { color: white !important; }

.bot-bubble {
    align-self: flex-start;
    background-color: white;
    border: 1px solid var(--border);
    color: var(--text-p) !important;
    border-bottom-left-radius: 4px;
    box-shadow: var(--shadow);
}

/* ── Reference Cards ── */
.ref-box {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px dashed var(--border);
    font-size: 0.8rem;
    color: var(--text-s) !important;
}

/* ── Loader ── */
.loader-box {
    text-align: center;
    padding: 2rem;
    background: white;
    border-radius: 12px;
    border: 1px solid var(--border);
}

/* ── Sidebar Typography ── */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--primary) !important;
}

</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "vectorstore"    not in st.session_state: st.session_state.vectorstore    = None
if "groq_client"    not in st.session_state: st.session_state.groq_client    = None
if "kb_ready"       not in st.session_state: st.session_state.kb_ready       = False
if "api_configured" not in st.session_state: st.session_state.api_configured = False

# ─── Functions ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def init_groq_api(key):
    return Groq(api_key=key)

# ─── Sidebar (Internal Logic & Minimal UI) ────────────────────────────────────
with st.sidebar:
    st.markdown("### DisasterBot AI")
    st.caption("Professional Crisis Support")
    
    # Background API Key processing
    if GROQ_API_KEY and not st.session_state.api_configured:
        try:
            st.session_state.groq_client = init_groq_api(GROQ_API_KEY)
            st.session_state.api_configured = True
        except Exception as e:
            st.sidebar.error(f"GROQ API Error: {e}")

    # Background KB processing
    if os.path.exists(INDEX_PATH) and not st.session_state.kb_ready:
        try:
            st.session_state.vectorstore = load_assets()
            st.session_state.kb_ready = True
        except Exception as e:
            st.sidebar.error(f"KB Error: {e}")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── Main Content ───

# Header Pill Area
pill_col1, pill_col2 = st.columns([1, 1])
with pill_col1:
    stat_color = "#22c55e" if st.session_state.api_configured else "#ef4444"
    st.markdown(f"""
    <div class="status-pill">
        <div class="status-dot" style="background:{stat_color};"></div>
        GROQ LLM SERVICE: {"ACTIVE" if st.session_state.api_configured else "OFFLINE"}
    </div>
    """, unsafe_allow_html=True)
with pill_col2:
    kb_color = "#3b82f6" if st.session_state.kb_ready else "#64748b"
    st.markdown(f"""
    <div class="status-pill">
        <div class="status-dot" style="background:{kb_color};"></div>
        KNOWLEDGE BASE: {"SYNCED" if st.session_state.kb_ready else "DISCONNECTED"}
    </div>
    """, unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-container">
    <div class="hero-title">DisasterBot AI</div>
    <div class="hero-sub">
        Expert emergency response procedures and preparedness guidelines. 
        Powered by high-performance Llama 3 models and specialized knowledge indexing.
    </div>
</div>
""", unsafe_allow_html=True)

# Shortcuts
st.markdown("<p style='font-weight:700; color:var(--primary); font-size:0.8rem; margin-left:5px;'>HELP SHORTCUTS</p>", unsafe_allow_html=True)
q_cols = st.columns(6)
for i, (icon, label) in enumerate(QUICK_PROMPTS):
    with q_cols[i]:
        if st.button(f"{icon} {label}", key=f"btn_{i}"):
            st.session_state.pending_query = label

# Chat Screen
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 5rem 0; color:var(--text-s);">
        <div style="font-size:3rem; margin-bottom:1rem;">🏢</div>
        <div style="font-weight:600;">System Standby</div>
        <div style="font-size:0.9rem;">Submit a query or use a shortcut above to begin.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        content = msg["content"].replace("\n", "<br>")
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        label = "YOU" if msg["role"] == "user" else "DISASTERBOT"
        
        source_html = ""
        if msg.get("sources"):
            tags = " • ".join(msg["sources"])
            source_html = f'<div class="ref-box"><b>REFERENCE:</b> {tags}</div>'
            
        st.markdown(f"""
        <div class="msg-row">
            <div class="msg-bubble {role_class}">
                <div style="font-size:0.65rem; font-weight:700; opacity:0.7; margin-bottom:0.6rem; letter-spacing:0.05em;">{label}</div>
                {content}
                {source_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Chat Input
query = st.chat_input("Ask a preparedness question...")

if "pending_query" in st.session_state and not query:
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Consulting guidelines..."):
        if not st.session_state.api_configured:
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please configure your API key in the sidebar."})
        else:
            try:
                # Search
                results = st.session_state.vectorstore.similarity_search(query, k=5)
                ctx = "\n\n".join([r.page_content for r in results])
                src = list({os.path.basename(r.metadata.get("source", "Unknown")) for r in results})
                
                # Ask
                response = st.session_state.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}
                    ]
                )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.choices[0].message.content,
                    "sources": src
                })
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Connection Error: {str(e)}"})
    
    st.rerun()

st.markdown("<div style='height: 50px'></div>", unsafe_allow_html=True)
