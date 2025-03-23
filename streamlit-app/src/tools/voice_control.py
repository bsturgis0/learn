#streamlit-app/src/voice_control.py
import streamlit as st
import boto3
from langchain_core.tools import BaseTool
from typing import Optional, Any

class VoiceControlTool(BaseTool):
    name: str = "voice_control"
    func: str = "control_voice"
    description: str = "Control voice settings for Zoti. Input should be one of: 'list_voices', 'list_languages', 'set_language:{language_name}', 'set_voice:{voice_name}', 'enable_voice', 'disable_voice', 'voice_status'."
    
    def _run(self, command: str) -> str:
        command_lower = command.lower()
        
        try:
            if command_lower == 'list_voices':
                response = "Available Amazon Polly Neural voices:\n"
                for language, voices in st.session_state.voice_config['available_voices'].items():
                    response += f"\n{language}:\n"
                    response += f"  Male: {', '.join(voices['male'])}\n"
                    response += f"  Female: {', '.join(voices['female'])}\n"
                response += f"\nCurrently selected: {st.session_state.voice_config['selected_voice']}"
                return response
            
            elif command_lower == 'list_languages':
                languages = list(st.session_state.voice_config['available_voices'].keys())
                return f"Available languages: {', '.join(languages)}"
            
            elif command_lower.startswith('set_language:'):
                language_name = command.split(':')[1].strip()
                languages = list(st.session_state.voice_config['available_voices'].keys())
                
                if language_name in languages:
                    st.session_state.voice_config['current_language'] = language_name
                    voices = st.session_state.voice_config['available_voices'][language_name]
                    st.session_state.voice_config['selected_voice'] = voices['male'][0]
                    return f"Language set to {language_name}. Default voice set to {st.session_state.voice_config['selected_voice']}."
                else:
                    return f"Language \"{language_name}\" not found. Available languages: {', '.join(languages)}"
            
            elif command_lower.startswith('set_voice:'):
                voice_name = command.split(':')[1].strip()
                voice_found = False
                voice_language = ''
                
                for language, voices in st.session_state.voice_config['available_voices'].items():
                    if isinstance(voices, dict):
                        if voice_name in voices['male'] or voice_name in voices['female']:
                            voice_found = True
                            voice_language = language
                            break
                
                if voice_found:
                    st.session_state.voice_config['selected_voice'] = voice_name
                    st.session_state.voice_config['current_language'] = voice_language
                    return f"Voice set to {voice_name} ({voice_language})."
                else:
                    return f"Voice \"{voice_name}\" not found. Use 'list_voices' to see available voices."
            
            elif command_lower == 'enable_voice':
                st.session_state.voice_config['enabled'] = True
                return 'Voice output enabled.'
            
            elif command_lower == 'disable_voice':
                st.session_state.voice_config['enabled'] = False
                return 'Voice output disabled.'
            
            elif command_lower == 'voice_status':
                return f"""Voice output is currently {'enabled' if st.session_state.voice_config['enabled'] else 'disabled'}.
Selected voice: {st.session_state.voice_config['selected_voice']}
Current language: {st.session_state.voice_config['current_language'] or 'Not set'}"""
            
            else:
                return f"Unknown command: \"{command}\". Valid commands are: 'list_voices', 'list_languages', 'set_language:{{language_name}}', 'set_voice:{{voice_name}}', 'enable_voice', 'disable_voice', 'voice_status'."
                
        except Exception as error:
            print('Error in voice control:', error)
            return f"Error in voice control: {str(error)}"

    async def _arun(self, command: str) -> str:
        return self._run(command)