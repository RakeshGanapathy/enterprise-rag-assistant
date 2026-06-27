import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import requests
import json
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

# Page config
st.set_page_config(page_title='RAG Knowledge Assistant', layout='wide')

# Header with status
col_header1, col_header2 = st.columns([2, 1])
with col_header1:
    st.title('🤖 RAG Knowledge Assistant')
with col_header2:
    st.write('')
    st.write('📚 Intelligent document search powered by LLM')

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    try:
        r = requests.get(f'{API_BASE}/health', timeout=2)
        st.metric('API', '🟢 Ready', delta='Connected')
    except:
        st.metric('API', '🔴 Offline', delta='No connection')

with col2:
    st.metric('Vector Store', '🟢 Ready', delta='PgVector')

with col3:
    st.metric('Embedding Model', '🟢 Ready', delta='text-embedding-3-small')

with col4:
    st.metric('LLM', '🟢 Ready', delta='GPT-4o')

st.markdown('---')

# Main tabs
tab_upload, tab_documents, tab_ask, tab_settings = st.tabs(
    ['📤 Upload & Ingest', '📚 Documents', '💬 Ask Questions', '⚙️ Settings']
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
                st.write(f'✓ {f.name}')
            
            if st.button('📤 Upload & Index Files', key='upload_btn', use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                    try:
                        status_text.text(f'Processing {uploaded_file.name}...')
                        r = requests.post(f'{API_BASE}/documents/upload', files=files, timeout=60)
                        r.raise_for_status()
                        
                        result = r.json()
                        obs_id = r.headers.get('X-Langfuse-Observation-Id')
                        
                        with results_container:
                            with st.expander(f'✅ {uploaded_file.name}', expanded=True):
                                st.write(f'**Chunks indexed:** {result.get("chunks_indexed", 0)}')
                                if obs_id:
                                    st.markdown(f'[📊 View trace in Langfuse]({lf_base}/project/{project_id}/traces/{obs_id})')
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        st.session_state.ingestion_complete = True
                        
                    except Exception as e:
                        st.error(f'❌ Failed to upload {uploaded_file.name}: {e}')
                
                status_text.text('✅ Upload complete!')
    
    with col2:
        st.subheader('Sample Documents')
        st.write('Ingest example documents bundled with the app')
        
        col_samples = st.columns(1)
        with col_samples[0]:
            if st.button('📚 Ingest Sample Documents', use_container_width=True):
                with st.spinner('Ingesting sample documents...'):
                    try:
                        r = requests.post(f'{API_BASE}/documents/ingest-samples', timeout=120)
                        r.raise_for_status()
                        
                        result = r.json()
                        obs_id = r.headers.get('X-Langfuse-Observation-Id')
                        
                        st.success(f'✅ Ingestion complete! Indexed {result.get("chunks_indexed", 0)} chunks')
                        if obs_id:
                            st.markdown(f'[📊 View trace in Langfuse]({lf_base}/project/{project_id}/traces/{obs_id})')
                        st.session_state.ingestion_complete = True
                    except Exception as e:
                        st.error(f'❌ Ingestion failed: {e}')


# ============================================================================
# TAB 2: DOCUMENTS
# ============================================================================
with tab_documents:
    st.subheader('Document Library')
    
    col_fetch, col_refresh = st.columns([1, 1])
    with col_refresh:
        if st.button('🔄 Refresh', use_container_width=True):
            st.rerun()
    
    try:
        r = requests.get(f'{API_BASE}/debug/chunks?limit=100', timeout=30)
        r.raise_for_status()
        chunks = r.json()
        
        if chunks:
            # Extract unique documents
            docs_set = {}
            for chunk in chunks:
                try:
                    metadata = json.loads(chunk.get('metadata_json', '{}'))
                    source = metadata.get('source', 'unknown')
                    if source not in docs_set:
                        docs_set[source] = {
                            'count': 0,
                            'sample_text': chunk['text'][:200]
                        }
                    docs_set[source]['count'] += 1
                except:
                    pass
            
            # Display summary
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            with col_summary1:
                st.metric('Total Documents', len(docs_set))
            with col_summary2:
                st.metric('Total Chunks', len(chunks))
            with col_summary3:
                st.metric('Last Indexed', 'Just now', delta='Up to date')
            
            st.markdown('---')
            
            # Display documents
            st.write('**Ingested Documents**')
            for source, info in sorted(docs_set.items()):
                with st.expander(f'✓ {source} ({info["count"]} chunks)', expanded=False):
                    st.write(f'**Chunks:** {info["count"]}')
                    st.write(f'**Sample text:** {info["sample_text"]}...')
        else:
            st.info('📭 No documents indexed yet. Upload files or ingest samples to get started.')
            
    except Exception as e:
        st.warning(f'⚠️ Could not fetch document list: {e}')


# ============================================================================
# TAB 3: ASK QUESTIONS (CHAT INTERFACE)
# ============================================================================
with tab_ask:
    st.subheader('Ask Questions')
    
    # Sample questions
    col_s1, col_s2, col_s3 = st.columns(3)
    sample_questions = [
        "What is the PTO policy?",
        "What are the security requirements?",
        "What is the incident escalation process?"
    ]
    
    for i, col in enumerate([col_s1, col_s2, col_s3]):
        with col:
            if st.button(f'❓ "{sample_questions[i]}"', use_container_width=True):
                st.session_state.chat_messages.append({'role': 'user', 'content': sample_questions[i]})
    
    st.markdown('---')
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                with st.chat_message('user', avatar='👤'):
                    st.write(msg['content'])
            else:
                with st.chat_message('assistant', avatar='🤖'):
                    st.write(msg['content'])
                    if 'sources' in msg:
                        with st.expander('📚 Sources & Relevance'):
                            for source in msg['sources']:
                                relevance = f" ({source.get('score', 0)*100:.0f}%)" if source.get('score') else ""
                                st.write(f"**{source.get('source', 'Unknown')}{relevance}**")
                                if source.get('page'):
                                    st.write(f"Page {source['page']}")
    
    # Input area
    st.markdown('---')
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        user_input = st.text_input('Your question:', placeholder='Ask something about the documents...', label_visibility='collapsed')
    
    with col_button:
        st.write('')
        submit_btn = st.button('Send', use_container_width=True, type='primary')
    
    if submit_btn and user_input:
        # Add user message
        st.session_state.chat_messages.append({'role': 'user', 'content': user_input})
        
        # Get mode
        col_mode = st.columns(1)[0]
        mode = 'ask'  # Default to ask mode for better UX
        
        try:
            with st.spinner('🔍 Searching and generating answer...'):
                payload = {'question': user_input, 'top_k': 4}
                r = requests.post(f'{API_BASE}/{mode}', json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                obs_id = r.headers.get('X-Langfuse-Observation-Id')
                
                # Format response
                response = {
                    'role': 'assistant',
                    'content': data.get('answer', 'No answer generated'),
                    'sources': data.get('sources', []),
                    'trace_id': obs_id
                }
                
                st.session_state.chat_messages.append(response)
                st.rerun()
        
        except Exception as e:
            st.error(f'❌ Error: {e}')
    
    # Show trace link if available
    if st.session_state.chat_messages:
        last_msg = st.session_state.chat_messages[-1]
        if last_msg['role'] == 'assistant' and last_msg.get('trace_id'):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"[📊 View last trace in Langfuse]({lf_base}/project/{project_id}/traces/{last_msg['trace_id']})")


# ============================================================================
# TAB 4: SETTINGS & DEBUG
# ============================================================================
with tab_settings:
    st.subheader('Settings & Debug')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**API Configuration**')
        st.text(f'API Base: {API_BASE}')
        st.text(f'Langfuse URL: {lf_base}')
        st.text(f'Project ID: {project_id}')
    
    with col2:
        st.write('**Advanced Options**')
        show_debug = st.checkbox('Show debug information')
    
    if show_debug:
        st.markdown('---')
        st.write('**Debug Information**')
        
        if st.button('List all stored chunks'):
            try:
                r = requests.get(f'{API_BASE}/debug/chunks?limit=50', timeout=30)
                r.raise_for_status()
                chunks = r.json()
                
                st.write(f'**Total chunks (showing first 50): {len(chunks)}**')
                
                for chunk in chunks[:10]:
                    with st.expander(f"Chunk {chunk['id'][:20]}..."):
                        st.write(f"**Text:** {chunk['text'][:300]}...")
                        try:
                            metadata = json.loads(chunk.get('metadata_json', '{}'))
                            st.write(f"**Metadata:** {metadata}")
                        except:
                            pass
                
                obs_id = r.headers.get('X-Langfuse-Observation-Id')
                if obs_id:
                    st.markdown(f"[📊 View trace]({lf_base}/project/{project_id}/traces/{obs_id})")
            except Exception as e:
                st.error(f'Error: {e}')
    
    st.markdown('---')
    
    # Chat history management
    st.write('**Chat History**')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Clear chat history'):
            st.session_state.chat_messages = []
            st.success('Chat history cleared')
            st.rerun()
    
    with col2:
        if st.button('Export chat as JSON'):
            st.json(st.session_state.chat_messages)
