# Zoti Document Teacher

## Overview
Zoti is an AI-powered document teaching assistant built with Streamlit. It helps users understand their documents by providing detailed explanations and insights, navigating through pages, and offering voice interaction capabilities.

## Features
- Load and analyze various document formats (PDF, DOCX, CSV, TXT).
- Navigate through document pages with ease.
- Voice control for a more interactive learning experience.
- User-friendly interface for uploading documents and managing settings.

## Project Structure
```
streamlit-app
├── src
│   ├── app.py                  # Main entry point of the Streamlit application
│   ├── tools
│   │   ├── document_loader.py   # Tool for loading documents
│   │   ├── page_reader.py       # Tool for reading document pages
│   │   └── voice_control.py      # Tool for managing voice settings
│   └── utils
│       └── text_to_speech.py    # Utility functions for text-to-speech
├── .env                         # Environment variables
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-app
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables in the `.env` file.

5. Run the application:
   ```
   streamlit run src/app.py
   ```

## Usage
- Upload your documents using the sidebar.
- Interact with Zoti by sending messages to load documents and navigate through them.
- Use voice settings to enhance your learning experience.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.