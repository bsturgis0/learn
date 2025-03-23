#streamlit-app/src/app.py
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

load_dotenv()

# Initialize voice configuration in session state if it doesn't exist
if 'voice_config' not in st.session_state:
    st.session_state.voice_config = {
        'enabled': True,
        'selected_voice': 'Joanna',  # Default Amazon Polly voice
        'current_language': '🇺🇸 English',  # Default language
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
                'male': ['Mathieu', 'Léa'],
                'female': ['Céline', 'Léa']
            },
            '🇩🇪 German': {
                'male': ['Hans', 'Daniel'],
                'female': ['Marlene', 'Vicki']
            },
            '🇮🇹 Italian': {
                'male': ['Giorgio', 'Carla'],
                'female': ['Bianca', 'Carla']
            }
        }
    }

# Initialize Amazon Polly client
polly_client = boto3.client('polly',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize chat history in session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize memory saver
memory = MemorySaver()

# Define the tools for the agent to use
@st.cache_resource
def get_tools():
    tavily_search = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_raw_content=True,
        include_answer=True,
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

# Get tools and create tool node
tools = get_tools()
tool_node = ToolNode([tool for tool in tools])  # Create list of instantiated tools

# Define state schema
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the model's call function
async def call_model(state):
    try:
        # Get the model from the cached workflow
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            max_output_tokens=8192,
            temperature=0.2,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        ).bind_tools(tools)  # Bind the tools to the model
        
        # Call the model
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}
        
    except Exception as error:
        print('Error in call_model:', error)
        return {
            "messages": [
                AIMessage(content="I'm having trouble connecting to my AI service right now. "
                          "Please check your API keys and try again.")
            ]
        }

# Define the workflow
@st.cache_resource
def create_workflow():
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

# Initialize the workflow
app = create_workflow()

# Sidebar for file upload and voice settings
with st.sidebar:
    st.title("📚 Zoti Document Teacher")
    st.markdown("Upload your documents and let Zoti teach you!")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload your document", accept_multiple_files=True, 
                                      type=["pdf", "docx", "doc", "txt", "csv"])
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.markdown("### Uploaded Files:")
        for file in uploaded_files:
            st.markdown(f"- {file.name}")
    
    # Voice settings section
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
        voice_type = st.radio("Voice type", ["Male", "Female"])
        
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

# Main content area - Chat interface
st.title("Zoti Document Teacher")
st.markdown("Your AI teaching assistant for understanding documents")

# Initialize messages container
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Hello! I'm Zoti, your document teaching assistant. What would you like to learn today? If you've uploaded a document, let me know its name and I'll help you understand it step by step."
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Enhanced system message for Zoti
system_message = SystemMessage(content="""# 📚 You are Zoti, a precise and methodical educator focused on teaching document content exactly as written.

## 🎯 Core Teaching Requirements:
1. **MANDATORY**: Use the page_reader tool for EVERY page - never skip this step
2. **NO ASSUMPTIONS**: Teach only what is explicitly written in the document
3. **NO SUMMARIES**: Present the complete content as written
4. **NO HALLUCINATIONS**: Never add information not present in the document

## 📖 Teaching Process:
1. **Start Page**: Always begin by using page_reader tool to get exact content
2. **Exact Teaching**: Teach the content exactly as provided by page_reader
3. **Verification**: Ask student if they understood the exact content before proceeding
4. **Next Page**: Only move to next page after student confirms understanding
5. **Page Tracking**: Keep track of current page number and total pages
6. **Further Explanation**: If student requests more context, use web search tool, but ONLY if requested, never assume, and your search must be based on what you are teaching currently.
7. 

## 🔍 Document Interaction Rules:
- MUST use page_reader tool before teaching any page
- MUST teach complete content without summarizing
- MUST verify student understanding of current page before proceeding
- MUST ask student's permission before moving to next page
- MUST maintain exact sequence of pages without skipping

## ⚠️ Critical Restrictions:
- NO summarizing content
- NO skipping pages
- NO adding external information
- NO assumptions about content
- NO teaching without first using page_reader

## 🛠️ Required Tool Usage:
1. **Page Reader**: MANDATORY use for every page (page_reader tool)
2. **Document Loader**: Only for initial document loading
3. **Voice Control**: For voice output adjustments
4. **Web Search**: ONLY if student specifically requests additional context

Remember: Your primary duty is to teach EXACTLY what is in the document, page by page, using the page_reader tool for each page. Never deviate from or summarize the actual content. 
Do not be stressed by the pace of teaching, as it is important to ensure the student understands each page before moving on.
Do not be strict in the teaching process, but be patient, friendly and understanding with the student's learning pace.
If student requests additional context, use web search tool, but only if requested, never assume, and your search must be based on what you are teaching currently.
If student asks questions, answer them based on the content you are teaching, and if you are unsure, use the web search tool to find the answer.
If student has no slide to upload, you can use the web search tool to find a document to teach, but only if student agrees to it. It should be about what the student is interested in learning.


Example interaction flow:
1. Use page_reader tool for current page
2. Teach exact content from page_reader, make sure you explain it very well like a human being will do.
3. Verify student understanding
4. Ask permission for next page
5. Repeat process

Begin by asking which document to teach and use the document_loader tool to start.""")

# Function to synthesize speech using Amazon Polly
async def synthesize_speech(text: str, voice_id: str) -> None:
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id,
            Engine='neural'
        )
        
        if "AudioStream" in response:
            # Create an audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                f.write(response["AudioStream"].read())
                audio_file = f.name
            
            # Display audio in Streamlit
            st.audio(audio_file, format='audio/mp3')
            
            # Cleanup temp file
            os.unlink(audio_file)
            
    except Exception as e:
        st.error(f"Error synthesizing speech: {str(e)}")

# Initialize thread ID in session state if it doesn't exist
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid4())

def get_conversation_state():
    """Retrieve the current conversation state from memory"""
    if 'thread_id' in st.session_state:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        return app.get_state(config)
    return None

# Chat input handler
async def handle_chat_input(prompt: str):
    try:
        # Configure thread for memory
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Prepare messages for the model
        messages = [
            system_message,  # Include system message first
            *[HumanMessage(content=m["content"]) if m["role"] == "user" 
              else AIMessage(content=m["content"]) 
              for m in st.session_state.messages[1:]]  # Skip welcome message
        ]
        
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
            st.markdown(response)
            
            # Generate speech if enabled
            if st.session_state.voice_config['enabled']:
                await synthesize_speech(
                    response,
                    st.session_state.voice_config['selected_voice']
                )
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Main chat interface
if prompt := st.chat_input("What would you like to learn today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Handle the chat input asynchronously
    asyncio.run(handle_chat_input(prompt))