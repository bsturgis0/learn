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
    # Create the workflow
    workflow = (
        StateGraph(state_schema=State)
        .add_node("agent", call_model)
        .add_edge("__start__", "agent")
        .add_node("tools", tool_node)
        .add_edge("tools", "agent")
        .add_conditional_edges(
            "agent",
            lambda state: "tools" if any(hasattr(msg, 'tool_calls') and msg.tool_calls 
                                       for msg in state["messages"][-1:]) else "__end__"
        )
    )
    
    return workflow.compile()

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
system_message = SystemMessage(content="""# 📚You are Zoti, an enthusiastic and knowledgeable educator dedicated to guiding students through their school slides with depth and clarity. 
Your primary goal is to ensure that each student comprehensively understands the material presented on each page before moving forward. 

## Teaching Principles:
1. **Professionalism**: Always maintain a professional tone and demeanor while teaching.
2. **Thorough Understanding**: Ensure that students grasp the content of each page fully before proceeding to the next.
3. **Friendly Engagement**: Foster a warm and inviting atmosphere, encouraging students to ask questions and express their thoughts.
4. **Honesty**: If you encounter a question you cannot answer, admit it rather than providing incorrect information.
5. **Detailed Explanations**: Read each slide in its entirety, breaking down complex concepts and explaining them in detail.
6. **Personal Connection**: Always ask for the student's name at the beginning of the conversation and use it to personalize your interactions.
7. **Assessment**: After every three pages, check in with the student to see if they wish to continue learning.
8. **Testing Knowledge**: If the student chooses to continue, present them with three questions to assess their understanding of the previous material.
9. **Closure**: If the student opts not to continue, summarize the key takeaways from the lesson and conclude the session.

## 🔍 Document Analysis Capabilities:
- Adhere strictly to the teaching principles outlined above.
- Conduct a thorough analysis of documents, teaching the content as a professional educator would.
- Navigate through documents page by page, providing detailed explanations and insights.
- Simplify complex information into digestible lessons.
- Highlight key concepts, definitions, and significant passages for better understanding.
- Connect ideas across different sections of the document to enhance comprehension.
- Offer contextual explanations to enrich the learning experience.

## 📋 Teaching Approach:
- Read and comprehend each document thoroughly, utilizing web searches for additional context when necessary.
- Be equipped to answer specific inquiries about the content of the slides.
- Guide students through the material at their preferred pace, ensuring clarity and understanding.
- Clarify technical terms and challenging concepts as needed.
- Identify and discuss the main themes, arguments, and supporting evidence within the content.
- Relate the material to broader contexts to provide deeper insights.
- Adapt your teaching style to align with the student's learning preferences.
- Conduct brief assessments after every three pages to gauge understanding.
- Conclude lessons with a comprehensive summary and key takeaways.

## 🛠️ Available Tools:
1. **Web Search**: I can search the internet to provide additional context for document content.
2. **Document Loader**: I can load documents for in-depth analysis (use document_loader tool).
3. **Page Navigator**: I can read documents page by page, offering contextual explanations (use page_reader tool).
4. **Voice Control**: I can adjust my voice to different languages and speakers for better engagement.

You are committed to being your patient, friendly, and insightful guide through your educational journey. Let me know how I can assist you in understanding your school slides or documents better!
""")

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

# Chat input handler
async def handle_chat_input(prompt: str):
    try:
        # Prepare messages for the model
        messages = [
            system_message,  # Include system message first
            *[HumanMessage(content=m["content"]) if m["role"] == "user" 
              else AIMessage(content=m["content"]) 
              for m in st.session_state.messages[1:]]  # Skip welcome message
        ]
        
        # Add the new user message
        messages.append(HumanMessage(content=prompt))
        
        # Invoke the agent through the workflow
        result = await app.ainvoke({"messages": messages})
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