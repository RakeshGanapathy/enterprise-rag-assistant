import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import requests
import json
import subprocess
import sys
from datetime import datetime

# Configuration
try:
    API_BASE = st.secrets.get('api_base', 'http://localhost:8000')
except StreamlitSecretNotFoundError:
    API_BASE = 'http://localhost:8000'

lf_base = 'http://localhost:3000'
project_id = 'cmq3yj7310006mr07xfhwo85r'
try:
    lf_base = st.secrets.get('langfuse_url', lf_base)
    project_id = st.secrets.get('langfuse_project_id', project_id)
except StreamlitSecretNotFoundError:
    pass

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "ingestion_complete" not in st.session_state:
    st.session_state.ingestion_complete = False
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Page config
st.set_page_config(page_title='RAG Knowledge Assistant', layout='wide')

# ── Sidebar: auth ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Authentication")

    if st.button("Auto-generate token", use_container_width=True):
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
import sys, datetime
sys.path.insert(0,'.')
from app.config import get_settings
import jose.jwt as j
s = get_settings()
print(j.encode(
    {'sub':'ui-user','domain':'all',
     'actions':['read:public','read:internal','read:confidential','read:restricted'],
     'exp': datetime.datetime.now(datetime.UTC)+datetime.timedelta(hours=8)},
    s.jwt_secret, algorithm='HS256'))
"""],
                capture_output=True, text=True, cwd="."
            )
            if result.returncode == 0:
                st.session_state.jwt_token = result.stdout.strip()
                st.success("Token generated!")
            else:
                st.error(f"Error: {result.stderr[:200]}")
        except Exception as e:
            st.error(f"Failed: {e}")

    token_input = st.text_area(
        "JWT Token", value=st.session_state.jwt_token,
        height=100, help="Paste a JWT or click Auto-generate"
    )
    if token_input != st.session_state.jwt_token:
        st.session_state.jwt_token = token_input

    if st.session_state.jwt_token:
        st.success("Token set")
    else:
        st.warning("No token — authenticated endpoints will fail")

    st.markdown("---")
    st.caption(f"API: {API_BASE}")


def _auth_headers():
    h = {}
    if st.session_state.jwt_token:
        h["Authorization"] = f"Bearer {st.session_state.jwt_token}"
    return h


# ── Header ────────────────────────────────────────────────────────────────────
col_header1, col_header2 = st.columns([2, 1])
with col_header1:
    st.title('RAG Knowledge Assistant')
with col_header2:
    st.write('')
    st.write('Intelligent document search powered by LLM')

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    try:
        r = requests.get(f'{API_BASE}/health', timeout=2)
        data = r.json()
        db = data.get('db', '?')
        st.metric('API', 'Ready' if data.get('status') == 'ok' else 'Degraded',
                  delta=f'DB: {db}')
    except Exception as e:
        st.metric('API', 'Offline', delta=str(e)[:30])
with col2:
    st.metric('Vector Store', 'Ready', delta='PgVector')
with col3:
    st.metric('Embedding', 'Ready', delta='text-embedding-3-small')
with col4:
    st.metric('LLM', 'Ready', delta='GPT-4o')

st.markdown('---')

# Main tabs
tab_upload, tab_documents, tab_ask, tab_settings = st.tabs(
    ['Upload & Ingest', 'Documents', 'Ask Questions', 'Settings & Debug']
)

# ============================================================================
# TAB 1: UPLOAD & INGEST
# ============================================================================
with tab_upload:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Upload Documents')
        st.write('Add new files to the knowledge base')

        uploaded_files = st.file_uploader(
            'Drag & drop files here',
            type=['pdf', 'txt', 'md', 'docx'],
            accept_multiple_files=True,
            label_visibility='collapsed'
        )

        if uploaded_files:
            st.write(f'**Selected files ({len(uploaded_files)})**')
            for f in uploaded_files:
                st.write(f'+ {f.name}')

            if st.button('Upload & Index Files', key='upload_btn', use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                for i, uploaded_file in enumerate(uploaded_files):
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                    try:
                        status_text.text(f'Processing {uploaded_file.name}...')
                        r = requests.post(
                            f'{API_BASE}/documents/upload',
                            files=files,
                            headers=_auth_headers(),
                            timeout=60,
                        )
                        r.raise_for_status()
                        result = r.json()
                        with results_container:
                            with st.expander(f'{uploaded_file.name}', expanded=True):
                                st.write(f'**Job ID:** `{result.get("job_id","")}`')
                                st.write(f'**Status:** {result.get("status","")}')
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        st.session_state.ingestion_complete = True
                    except Exception as e:
                        st.error(f'Failed to upload {uploaded_file.name}: {e}')

                status_text.text('Upload complete!')

    with col2:
        st.subheader('Sample Documents')
        st.write('Ingest example documents bundled with the app')

        if st.button('Ingest Sample Documents', use_container_width=True):
            with st.spinner('Ingesting sample documents...'):
                try:
                    r = requests.post(
                        f'{API_BASE}/documents/ingest-samples',
                        headers=_auth_headers(),
                        timeout=120,
                    )
                    r.raise_for_status()
                    result = r.json()
                    st.success(f'Ingestion complete! Indexed {result.get("chunks_indexed", 0)} chunks')
                    st.session_state.ingestion_complete = True
                except Exception as e:
                    st.error(f'Ingestion failed: {e}')


# ============================================================================
# TAB 2: DOCUMENTS
# ============================================================================
with tab_documents:
    st.subheader('Document Library')

    if st.button('Refresh'):
        st.rerun()

    try:
        r = requests.get(
            f'{API_BASE}/debug/chunks?limit=100',
            headers=_auth_headers(),
            timeout=30,
        )
        r.raise_for_status()
        chunks = r.json()

        if chunks:
            docs_set = {}
            for chunk in chunks:
                try:
                    metadata = json.loads(chunk.get('metadata_json', '{}')) \
                        if isinstance(chunk.get('metadata_json'), str) \
                        else chunk.get('metadata_json', {})
                    source = metadata.get('source', 'unknown')
                    if source not in docs_set:
                        docs_set[source] = {'count': 0, 'sample_text': chunk['text'][:200]}
                    docs_set[source]['count'] += 1
                except Exception:
                    pass

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric('Total Documents', len(docs_set))
            with col_s2:
                st.metric('Total Chunks', len(chunks))
            with col_s3:
                st.metric('Last Indexed', 'Up to date')

            st.markdown('---')
            st.write('**Ingested Documents**')
            for source, info in sorted(docs_set.items()):
                with st.expander(f'{source} ({info["count"]} chunks)', expanded=False):
                    st.write(f'**Sample:** {info["sample_text"]}...')
        else:
            st.info('No documents indexed yet. Upload files or ingest samples.')

    except Exception as e:
        st.warning(f'Could not fetch document list: {e}')


# ============================================================================
# TAB 3: ASK QUESTIONS (CHAT INTERFACE)
# ============================================================================
with tab_ask:
    st.subheader('Ask Questions')

    col_nc, col_cid = st.columns([1, 3])
    with col_nc:
        if st.button('New conversation', use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.chat_messages = []
            st.rerun()
    with col_cid:
        if st.session_state.conversation_id:
            st.caption(f'Conversation ID: `{st.session_state.conversation_id}`')

    # Sample question buttons
    col_s1, col_s2, col_s3 = st.columns(3)
    sample_questions = [
        "What is the PTO policy?",
        "What are the security requirements?",
        "What is the incident escalation process?",
    ]
    for i, col in enumerate([col_s1, col_s2, col_s3]):
        with col:
            if st.button(f'"{sample_questions[i]}"', use_container_width=True):
                st.session_state.pending_question = sample_questions[i]
                st.rerun()

    st.markdown('---')

    # Chat history
    for msg in st.session_state.chat_messages:
        if msg['role'] == 'user':
            with st.chat_message('user'):
                st.write(msg['content'])
        else:
            with st.chat_message('assistant'):
                st.write(msg['content'])
                if msg.get('cached'):
                    st.caption('(served from cache)')
                if msg.get('sources'):
                    with st.expander('Sources'):
                        for s in msg['sources']:
                            relevance = f" ({s.get('score', 0)*100:.0f}%)" if s.get('score') else ""
                            st.write(f"**{s.get('source','?')}{relevance}**")

    st.markdown('---')
    user_input = st.chat_input('Ask something about the documents...')

    # Sample question buttons fire a rerun with pending_question set
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:
        st.session_state.chat_messages.append({'role': 'user', 'content': user_input})
        try:
            with st.spinner('Searching and generating answer...'):
                payload = {'question': user_input, 'top_k': 4}
                if st.session_state.conversation_id:
                    payload['conversation_id'] = st.session_state.conversation_id

                r = requests.post(
                    f'{API_BASE}/ask',
                    json=payload,
                    headers=_auth_headers(),
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()

                if data.get('conversation_id'):
                    st.session_state.conversation_id = data['conversation_id']

                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': data.get('answer', 'No answer generated'),
                    'sources': data.get('sources', []),
                    'cached': data.get('cached', False),
                })
                st.rerun()
        except Exception as e:
            st.error(f'Error: {e}')


# ============================================================================
# TAB 4: SETTINGS & DEBUG
# ============================================================================
with tab_settings:
    st.subheader('Settings & Debug')

    col1, col2 = st.columns(2)
    with col1:
        st.write('**API Configuration**')
        st.code(f'API Base: {API_BASE}\nLangfuse: {lf_base}\nProject:  {project_id}')

    with col2:
        st.write('**Cache**')
        if st.button('Show cache stats'):
            try:
                r = requests.get(f'{API_BASE}/cache/stats', headers=_auth_headers(), timeout=10)
                r.raise_for_status()
                st.json(r.json())
            except Exception as e:
                st.error(str(e))

        if st.button('Flush expired cache'):
            try:
                r = requests.delete(f'{API_BASE}/cache?expired_only=true', headers=_auth_headers(), timeout=10)
                r.raise_for_status()
                st.success(str(r.json()))
            except Exception as e:
                st.error(str(e))

    st.markdown('---')
    show_debug = st.checkbox('Show debug chunks')
    if show_debug:
        try:
            r = requests.get(f'{API_BASE}/debug/chunks?limit=50', headers=_auth_headers(), timeout=30)
            r.raise_for_status()
            chunks = r.json()
            st.write(f'**Chunks (first 50): {len(chunks)}**')
            for chunk in chunks[:10]:
                with st.expander(f"Chunk {chunk['id'][:20]}..."):
                    st.write(chunk['text'][:300])
        except Exception as e:
            st.error(str(e))

    st.markdown('---')
    st.write('**Chat History**')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Clear chat history'):
            st.session_state.chat_messages = []
            st.session_state.conversation_id = None
            st.success('Cleared')
            st.rerun()
    with col2:
        if st.button('Export chat as JSON'):
            st.json(st.session_state.chat_messages)
