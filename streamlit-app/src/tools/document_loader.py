#streamlit-app/src/document_loader.py
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from typing import Optional, List, Dict, Any

class DocumentLoadTool(BaseTool):
    name: str = "document_loader"
    func: str = "load_document"
    description: str = "Load a document for analysis and teaching. Input is the document name that has been uploaded."
    
    def _run(self, document_name: str) -> str:
        try:
            if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
                return "Error: No files have been uploaded. Please upload a file first."
            
            # Initialize document store if not already done
            if 'document_store' not in st.session_state:
                st.session_state.document_store = {
                    'document_pages': [],
                    'current_page_index': 0,
                    'total_pages': 0,
                    'document_name': ""
                }
            
            file_found = False
            for uploaded_file in st.session_state.uploaded_files:
                if uploaded_file.name == document_name:
                    file_found = True
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    loader = None
                    docs = None
                    
                    if file_ext == '.pdf':
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                    elif file_ext in ['.docx', '.doc']:
                        loader = Docx2txtLoader(temp_path)
                        docs = loader.load()
                    elif file_ext == '.csv':
                        loader = CSVLoader(temp_path)
                        docs = loader.load()
                    elif file_ext == '.txt':
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        docs = [Document(page_content=text)]
                    else:
                        os.unlink(temp_path)
                        return f'Error: Unsupported file format "{file_ext}". Supported formats are PDF, DOCX, CSV, and TXT.'
                    
                    st.session_state.document_store['document_pages'] = docs
                    st.session_state.document_store['current_page_index'] = 0
                    st.session_state.document_store['total_pages'] = len(docs)
                    st.session_state.document_store['document_name'] = uploaded_file.name
                    
                    os.unlink(temp_path)
                    
                    return f'Successfully loaded {st.session_state.document_store["document_name"]} with {st.session_state.document_store["total_pages"]} pages. You can now use the page_reader tool to explore the document page by page. Start with "read_current_page" to view the first page.'
            
            if not file_found:
                return f'Error: File "{document_name}" not found. Please check the filename and try again.'
                
        except Exception as error:
            print('Error loading document:', error)
            return f'Error loading document: {str(error)}'

    async def _arun(self, document_name: str) -> str:
        return self._run(document_name)