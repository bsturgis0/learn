import boto3
from typing import Dict, List

def get_available_polly_voices(polly_client) -> Dict[str, Dict[str, List[str]]]:
    """Get available Amazon Polly neural voices grouped by language and gender."""
    try:
        response = polly_client.describe_voices(Engine='neural')
        voices = response['Voices']
        
        # Group voices by language and gender
        voice_groups = {
            '🇺🇸 English': {'male': [], 'female': []},
            '🇪🇸 Spanish': {'male': [], 'female': []},
            '🇫🇷 French': {'male': [], 'female': []},
            '🇩🇪 German': {'male': [], 'female': []},
            '🇮🇹 Italian': {'male': [], 'female': []},
            '🇯🇵 Japanese': {'male': [], 'female': []}
        }
        
        for voice in voices:
            language_code = voice['LanguageCode']
            gender = voice['Gender'].lower()
            voice_id = voice['Id']
            
            # Map language codes to our language groups
            language_map = {
                'en-US': '🇺🇸 English',
                'es-ES': '🇪🇸 Spanish',
                'fr-FR': '🇫🇷 French',
                'de-DE': '🇩🇪 German',
                'it-IT': '🇮🇹 Italian',
                'ja-JP': '🇯🇵 Japanese'
            }
            
            if language_code in language_map:
                language = language_map[language_code]
                if language in voice_groups:
                    voice_groups[language][gender].append(voice_id)
        
        return voice_groups
    
    except Exception as e:
        print(f"Error getting Polly voices: {str(e)}")
        return {}