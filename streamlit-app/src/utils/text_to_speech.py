import pyttsx3
import streamlit as st

def initialize_tts():
    """Initialize the text-to-speech engine."""
    engine = pyttsx3.init()
    return engine

def set_voice(engine, voice_id):
    """Set the voice for the text-to-speech engine."""
    engine.setProperty('voice', voice_id)

def speak_text(engine, text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def get_available_voices(engine):
    """Retrieve a list of available voices."""
    voices = engine.getProperty('voices')
    return [(voice.id, voice.name) for voice in voices]

def text_to_speech(text):
    """Main function to convert text to speech based on the selected voice."""
    engine = initialize_tts()
    
    # Get the selected voice from session state
    selected_voice = st.session_state.voice_config['selected_voice']
    
    # Set the voice
    set_voice(engine, selected_voice)
    
    # Speak the text
    speak_text(engine, text)