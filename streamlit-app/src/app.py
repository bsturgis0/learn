import os
import boto3
import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
from tools.page_reader import PageReaderTool
from tools.voice_control import VoiceControlTool
from langgraph.graph.message import add_messages
from tools.document_loader import DocumentLoadTool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Optional, Union, Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END
from uuid import uuid4
import time

# Set wide layout by default
st.set_page_config(
    page_title="Zoti Document Teacher",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better text sizing and layout
st.markdown("""
<style>
    .small-text {
        font-size: 0.8rem !important;
    }
    .medium-text {
        font-size: 1rem !important;
    }
    .large-text {
        font-size: 1.2rem !important;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 100%;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f3ff;
    }
    .stApp {
        max-width: 100%;
    }
    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #4285f4;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize memory saver
memory = MemorySaver()

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if 'current_document' not in st.session_state:
    st.session_state.current_document = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0

if 'document_metadata' not in st.session_state:
    st.session_state.document_metadata = {}

if 'teaching_mode' not in st.session_state:
    st.session_state.teaching_mode = "standard"  # Options: standard, interactive, summarized

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

# Initialize voice configuration with expanded languages and voices
if 'voice_config' not in st.session_state:
    st.session_state.voice_config = {
        'enabled': True,
        'selected_voice': 'Joanna',
        'rate': 'medium',  # New: speech rate (slow, medium, fast)
        'current_language': '🇺🇸 English',
        'available_voices': {
            '🇺🇸 English': {
                'male': ['Matthew', 'Stephen', 'Kevin', 'Gregory'],
                'female': ['Joanna', 'Kendra', 'Salli', 'Ruth']
            },
            '🇪🇸 Spanish': {
                'male': ['Miguel', 'Enrique'],
                'female': ['Lupe', 'Penélope']
            },
            '🇫🇷 French': {
                'male': ['Mathieu', 'Rémi'],
                'female': ['Céline', 'Léa']
            },
            '🇩🇪 German': {
                'male': ['Hans', 'Daniel'],
                'female': ['Marlene', 'Vicki']
            },
            '🇮🇹 Italian': {
                'male': ['Giorgio', 'Marco'],
                'female': ['Bianca', 'Carla']
            },
            '🇯🇵 Japanese': {
                'male': ['Takumi', 'Kazuha'],
                'female': ['Mizuki', 'Haruka']
            },
            '🇰🇷 Korean': {
                'male': ['Seoyeon'],
                'female': ['Seoyeon']
            },
            '🇮🇳 Hindi': {
                'male': ['Arjun'],
                'female': ['Kajal']
            }
        }
    }

# Initialize learning analytics tracking
if 'learning_analytics' not in st.session_state:
    st.session_state.learning_analytics = {
        'start_time': time.time(),
        'interaction_count': 0,
        'documents_studied': [],
        'pages_completed': 0,
        'questions_asked': 0,
        'session_duration': 0,
        'focus_areas': {}
    }

# Initialize Amazon Polly client
try:
    polly_client = boto3.client('polly',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
except Exception as e:
    st.sidebar.error(f"Error initializing Polly client: {str(e)}")
    polly_client = None

# Define the tools for the agent to use
@st.cache_resource
def get_tools():
    try:
        tavily_search = TavilySearchResults(
            max_results=5,  # Increased from 3
            search_depth="advanced",
            include_raw_content=True,
            include_answer=True,
            include_images=True,  # New: include images in search results
            tavily_api_key=os.getenv('TAVILY_API_KEY')
        )
        document_loader = DocumentLoadTool()
        page_reader = PageReaderTool()
        voice_control = VoiceControlTool()
        
        return [
            tavily_search,
            document_loader,
            page_reader,
            voice_control
        ]
    except Exception as e:
        st.error(f"Error initializing tools: {str(e)}")
        return []

# Get tools and create tool node
tools = get_tools()
tool_node = ToolNode([tool for tool in tools])

# Define state schema
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the model's call function with error handling and retries
async def call_model(state):
    max_retries = 3
    retry_count = 0
    backoff_time = 1  # Start with 1 second backoff
    
    while retry_count < max_retries:
        try:
            # Get the model from the cached workflow
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                max_output_tokens=8192,
                temperature=0.3,  # Slightly increased for more natural responses
                google_api_key=os.getenv('GOOGLE_API_KEY'),
                streaming=True  # Enable streaming for faster initial responses
            ).bind_tools(tools)
            
            # Call the model
            response = await model.ainvoke(state["messages"])
            
            # Update learning analytics
            st.session_state.learning_analytics['interaction_count'] += 1
            st.session_state.last_activity = time.time()
            
            return {"messages": [response]}
            
        except Exception as error:
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                continue
            else:
                print(f'Error in call_model after {max_retries} retries:', error)
                return {
                    "messages": [
                        AIMessage(content="I'm having trouble connecting to my AI service right now. "
                                "Please check your API keys or try again later. "
                                "If this problem persists, please contact support.")
                    ]
                }

# Define the workflow with improved error handling
@st.cache_resource
def create_workflow():
    try:
        # Create the workflow with checkpointing
        workflow = (
            StateGraph(state_schema=State)
            .add_node("agent", call_model)
            .add_edge(START, "agent")
            .add_node("tools", tool_node)
            .add_edge("tools", "agent")
            .add_conditional_edges(
                "agent",
                lambda state: "tools" if any(hasattr(msg, 'tool_calls') and msg.tool_calls 
                                        for msg in state["messages"][-1:]) else END
            )
        )
        
        # Compile with memory checkpointing
        return workflow.compile(checkpointer=memory)
    except Exception as e:
        st.error(f"Error creating workflow: {str(e)}")
        return None

# Initialize the workflow
app = create_workflow()

# Function to update learning analytics
def update_analytics(action_type, details=None):
    if action_type == "page_complete":
        st.session_state.learning_analytics['pages_completed'] += 1
    elif action_type == "question_asked":
        st.session_state.learning_analytics['questions_asked'] += 1
    elif action_type == "document_loaded":
        if details and details not in st.session_state.learning_analytics['documents_studied']:
            st.session_state.learning_analytics['documents_studied'].append(details)
    elif action_type == "focus_area":
        if details:
            if details in st.session_state.learning_analytics['focus_areas']:
                st.session_state.learning_analytics['focus_areas'][details] += 1
            else:
                st.session_state.learning_analytics['focus_areas'][details] = 1
    
    # Update session duration
    st.session_state.learning_analytics['session_duration'] = time.time() - st.session_state.learning_analytics['start_time']

# Function to synthesize speech using Amazon Polly with enhanced options
async def synthesize_speech(text: str, voice_id: str) -> None:
    if not polly_client:
        st.warning("Voice synthesis is not available. Please check your AWS credentials.")
        return
        
    try:
        # Configure speech rate
        speech_rate_map = {
            'slow': 'x-slow',
            'medium': 'medium',
            'fast': 'x-fast'
        }
        speech_rate = speech_rate_map.get(st.session_state.voice_config['rate'], 'medium')
        
        # Apply SSML for better speech control
        ssml_text = f"""
        <speak>
            <prosody rate="{speech_rate}">
                {text}
            </prosody>
        </speak>
        """
        
        response = polly_client.synthesize_speech(
            Text=ssml_text,
            TextType='ssml',
            OutputFormat='mp3',
            VoiceId=voice_id,
            Engine='neural'
        )
        
        if "AudioStream" in response:
            # Create an audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                f.write(response["AudioStream"].read())
                audio_file = f.name
            
            # Display audio with autoplay enabled
            st.audio(audio_file, format='audio/mp3')
            
            # Add JavaScript to autoplay the audio
            st.markdown(
                f"""
                <script>
                    const audio = document.querySelector('audio');
                    if (audio) {{
                        audio.play();
                    }}
                </script>
                """,
                unsafe_allow_html=True
            )
            
            # Cleanup temp file
            os.unlink(audio_file)
            
    except Exception as e:
        st.warning(f"Error synthesizing speech: {str(e)}")


# Sidebar for file upload, voice settings, and new features
with st.sidebar:
    st.title("📚 Zoti Document Teacher")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Documents", "Voice", "Settings", "Analytics"])
    
    with tab1:
        st.markdown("### 📂 Document Management")
        
        # File uploader with expanded file types
        uploaded_files = st.file_uploader(
            "Upload documents", 
            accept_multiple_files=True, 
            type=["pdf", "docx", "doc", "txt", "csv", "pptx", "xlsx", "json", "md"]
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.markdown("#### Uploaded Files:")
            for file in uploaded_files:
                st.markdown(f"- {file.name}")
                # Update analytics when a new document is loaded
                update_analytics("document_loaded", file.name)
        
        # Document navigation controls
        if st.session_state.current_document:
            st.markdown(f"### 📄 Current Document")
            st.markdown(f"**{st.session_state.current_document}**")
            st.markdown(f"Page {st.session_state.current_page}/{st.session_state.total_pages}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("◀️ Previous", disabled=st.session_state.current_page <= 1):
                    st.session_state.current_page -= 1
            with col2:
                if st.button("Next ▶️", disabled=st.session_state.current_page >= st.session_state.total_pages):
                    st.session_state.current_page += 1
                    update_analytics("page_complete")
            with col3:
                page_num = st.number_input("Go to page", min_value=1, max_value=st.session_state.total_pages, 
                                          value=st.session_state.current_page)
                if page_num != st.session_state.current_page:
                    st.session_state.current_page = page_num
    
    with tab2:
        st.markdown("### 🎙️ Voice Settings")
        
        # Enable/Disable voice
        voice_enabled = st.toggle("Enable voice", value=st.session_state.voice_config['enabled'])
        st.session_state.voice_config['enabled'] = voice_enabled
        
        if voice_enabled:
            # Language selection
            languages = list(st.session_state.voice_config['available_voices'].keys())
            selected_language = st.selectbox(
                "Select language",
                languages,
                index=languages.index(st.session_state.voice_config['current_language']) 
                if st.session_state.voice_config['current_language'] in languages else 0
            )
            
            # Update language if changed
            if selected_language != st.session_state.voice_config['current_language']:
                st.session_state.voice_config['current_language'] = selected_language
                # Set default voice for the selected language
                voices = st.session_state.voice_config['available_voices'][selected_language]
                st.session_state.voice_config['selected_voice'] = voices['male'][0]
            
            # Voice type selection
            voice_type = st.radio("Voice type", ["Male", "Female"], horizontal=True)
            
            # Get voices for the selected language and type
            available_voices = st.session_state.voice_config['available_voices'][selected_language]
            voice_list = available_voices['male'] if voice_type == "Male" else available_voices['female']
            
            # Voice selection
            selected_voice = st.selectbox(
                "Select voice",
                voice_list,
                index=voice_list.index(st.session_state.voice_config['selected_voice'])
                if st.session_state.voice_config['selected_voice'] in voice_list
                else 0
            )
            
            # Update selected voice
            st.session_state.voice_config['selected_voice'] = selected_voice
            
            # New: Speech rate control
            speech_rate = st.select_slider(
                "Speech rate",
                options=["slow", "medium", "fast"],
                value=st.session_state.voice_config['rate']
            )
            st.session_state.voice_config['rate'] = speech_rate
    
    with tab3:
        st.markdown("### ⚙️ Teaching Settings")
        
        # Teaching mode selection
        teaching_mode = st.radio(
            "Teaching Mode",
            ["Standard", "Interactive", "Summarized"],
            index=["standard", "interactive", "summarized"].index(st.session_state.teaching_mode),
            horizontal=True
        )
        
        # Update teaching mode if changed
        if teaching_mode.lower() != st.session_state.teaching_mode:
            st.session_state.teaching_mode = teaching_mode.lower()
        
        # Document focus areas (for better targeting the teaching)
        st.markdown("#### Focus Areas")
        focus_areas = st.multiselect(
            "Select focus areas",
            ["Key Concepts", "Definitions", "Formulas", "Examples", "Practical Applications", "Historical Context"],
            default=[]
        )
        
        for area in focus_areas:
            update_analytics("focus_area", area)
        
        # New: Text formatting settings
        st.markdown("#### Display Settings")
        text_size = st.select_slider(
            "Text Size",
            options=["Small", "Medium", "Large"],
            value="Medium"
        )
        
        # Option to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "👋 Hello! I'm Zoti, your document teaching assistant. What would you like to learn today? If you've uploaded a document, let me know its name and I'll help you understand it step by step."
            })
            st.experimental_rerun()
            
    with tab4:
        st.markdown("### 📊 Learning Analytics")
        
        # Display current session stats
        st.markdown("#### Session Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Time Spent", f"{int(st.session_state.learning_analytics['session_duration'] // 60)} min")
            st.metric("Pages Completed", st.session_state.learning_analytics['pages_completed'])
        with col2:
            st.metric("Documents Studied", len(st.session_state.learning_analytics['documents_studied']))
            st.metric("Questions Asked", st.session_state.learning_analytics['questions_asked'])
        
        # Show documents studied
        if st.session_state.learning_analytics['documents_studied']:
            st.markdown("#### Documents Studied")
            for doc in st.session_state.learning_analytics['documents_studied']:
                st.markdown(f"- {doc}")
        
        # Show focus areas
        if st.session_state.learning_analytics['focus_areas']:
            st.markdown("#### Focus Areas")
            for area, count in st.session_state.learning_analytics['focus_areas'].items():
                st.markdown(f"- {area}: {count} interactions")

# Main content area - Chat interface with improved layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1 style='font-size: 1.8rem;'>Zoti Document Teacher</h1>", unsafe_allow_html=True)
    st.markdown("<p class='medium-text'>Your AI teaching assistant for understanding documents</p>", unsafe_allow_html=True)
    
    # Initialize messages container
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "👋 Hello! I'm Zoti, your document teaching assistant. What would you like to learn today? If you've uploaded a document, let me know its name and I'll help you understand it step by step."
        })
    
    # Display chat messages with improved styling
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            message_class = "user-message" if message["role"] == "user" else "assistant-message"
            with st.chat_message(message["role"]):
                st.markdown(f"<div class='chat-message {message_class}'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Enhanced system message for Zoti
    system_message = SystemMessage(content="""# 📚 You are Zoti, a dedicated and insightful educator. Your purpose is to facilitate genuine understanding of document content in a professional, engaging, and human-like manner.

## ✨ Your Core Identity & Mission:
*   **Be an Explainer, Not a Repeater:** Your primary role is to *understand* the material on each page and then *explain* it clearly and thoroughly. Do not simply present the raw text.
*   **Foster Deep Comprehension:** Guide the student towards grasping the concepts, their context, and significance. Use examples and analogies where appropriate.
*   **Personalize the Interaction:** Always learn and use the student's name, creating a more personal and engaging learning environment.
*   **Be Structurally Methodical:** Follow a clear, step-by-step process for teaching each page, ensuring the student keeps pace.
*   **Act Naturally:** Perform all background tasks (like accessing page content or searching for extra info if requested) seamlessly. **Never mention internal tool names or processes.**

## 📖 Your Teaching Methodology:
1.  **Get Acquainted:** Start by warmly greeting the student and asking for their name. Remember and use their name throughout your interactions.
2.  **Load the Material:** Ask which document they wish to study and load it. Confirm the document title and total page count.
3.  **Content Assessment (Each Page):**
    *   Access the content of the current page.
    *   **Critically Evaluate:** Determine if the page contains substantive educational content OR if it's primarily metadata (e.g., title page, table of contents, author block, dedication, blank page).
4.  **Teaching Substantive Content:**
    *   **Acknowledge the Page:** Announce the page number (e.g., "Okay, [Student's Name], let's move on to page X of Y.").
    *   **Conceptual Breakdown:** Mentally divide the page's core content into approximately **three logical sections or key ideas.**
    *   **Teach Section by Section:**
        *   **Explain:** Clearly explain the concept(s) in the first section using your own words. Synthesize the information.
        *   **Elaborate & Exemplify:** Provide relevant examples, analogies, or further context to deepen understanding of this section.
        *   **Check Understanding:** Engage the student about *this specific section*. Ask open-ended questions like, "Does that explanation of [concept] make sense, [Student's Name]?", "How would you apply this idea?", or "Can you think of an example related to this?"
        *   **Confirm Readiness:** Once the student confirms understanding of the section, ask if they're ready to move to the next section on the page (e.g., "Shall we look at the next key point on this page?").
    *   **Repeat:** Continue this explain-elaborate-check-confirm process for the remaining sections of the page.
5.  **Handling Non-Substantive Pages:**
    *   If a page is identified as metadata (like the title page example you provided), **do not teach it section by section.**
    *   **Acknowledge Briefly:** Simply state what the page is (e.g., "Alright, [Student's Name], page 1 is the title page for this chapter.")
    *   **Proceed Smoothly:** Immediately ask if they are ready to move to the first page with actual content (e.g., "Shall we proceed to the main content starting on the next page?"). Do *not* ask if they "understand" the metadata page.
6.  **Page Transition:** After successfully covering all sections of a substantive page (or acknowledging a non-substantive one), confirm the student is ready to proceed to the *next page*. (e.g., "Excellent. Are you ready to move on to page [Next Page Number], [Student's Name]?")
7.  **Offer External Context (If Needed):** If, during an explanation, the student asks for information clearly *outside* the document's scope, you can offer to look up additional context. Phrase it naturally, like: "That's a great question that goes a bit beyond this text. Would you like me to quickly look up some more information on [topic]?" Proceed only if they agree.

## 🚫 Critical Boundaries:
*   **Never Display Raw Text Verbatim:** Always explain in your own words after understanding the content.
*   **Never Mention Tool Names:** Actions like reading pages or searching should be invisible to the student.
*   **Intelligently Skip Non-Substantive Pages:** Recognize and handle metadata/introductory pages appropriately.
*   **Maintain Page Order:** Always proceed sequentially through the document.
*   **Pacing is Key:** Don't move to the next section or page until the student confirms understanding and readiness.

## 🏁 Getting Started:
Begin your interaction by introducing yourself briefly, asking for the student's name, and then inquiring about the document they wish to learn from.""")

with col2:
    # Replace the entire content in col2 with just the session timer
    minutes = int(st.session_state.learning_analytics['session_duration'] // 60)
    seconds = int(st.session_state.learning_analytics['session_duration'] % 60)
    st.markdown(f"**Session time:** {minutes:02d}:{seconds:02d}")

# Function to get conversation state
def get_conversation_state():
    """Retrieve the current conversation state from memory"""
    if 'thread_id' in st.session_state:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        return app.get_state(config)
    return None

# Inactivity detection (new feature)
current_time = time.time()
if (current_time - st.session_state.last_activity) > 300:  # 5 minutes inactivity
    st.session_state.last_activity = current_time
    # We could add a reengagement message here if desired

# Chat input handler with enhanced analytics
async def handle_chat_input(prompt: str):
    try:
        # Update analytics
        if any(keyword in prompt.lower() for keyword in ["why", "how", "what", "explain", "clarify", "question"]):
            update_analytics("question_asked")
        
        # Configure thread for memory
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Prepare messages for the model, including teaching mode
        messages = [
            system_message,  # Include system message first
            *[HumanMessage(content=m["content"]) if m["role"] == "user" 
              else AIMessage(content=m["content"]) 
              for m in st.session_state.messages[1:]]  # Skip welcome message
        ]
        
        # Inject teaching mode context
        mode_context = f"\nCurrent teaching mode: {st.session_state.teaching_mode}.\n"
        if st.session_state.teaching_mode == "interactive":
            mode_context += "Use the Socratic method, ask more questions, and guide the student to discover information."
        elif st.session_state.teaching_mode == "summarized":
            mode_context += "Focus on key points and provide more concise explanations."
        
        # Add teaching mode context to the last system message
        messages[0] = SystemMessage(content=messages[0].content + mode_context)
        
        # Add the new user message
        messages.append(HumanMessage(content=prompt))
        
        # Invoke the agent through the workflow with memory
        result = await app.ainvoke(
            {"messages": messages},
            config
        )
        response = result["messages"][-1].content
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display the response
        with st.chat_message("assistant"):
            st.markdown(f"<div class='chat-message assistant-message'>{response}</div>", unsafe_allow_html=True)
            
            # Generate speech if enabled
            if st.session_state.voice_config['enabled']:
                await synthesize_speech(
                    response,
                    st.session_state.voice_config['selected_voice']
                )
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add session idle restart functionality (auto restart if inactive for too long)
idle_threshold = 1800  # 30 minutes
if (time.time() - st.session_state.last_activity) > idle_threshold:
    # Reset session state
    for key in ['messages', 'thread_id', 'current_document', 'current_page']:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# Main chat interface
if prompt := st.chat_input("What would you like to learn today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message with custom styling
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message user-message'>{prompt}</div>", unsafe_allow_html=True)
    
    # Handle the chat input asynchronously
    asyncio.run(handle_chat_input(prompt))