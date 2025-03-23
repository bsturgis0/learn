#streamlit-app/src/page_reader.py
import os
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
import streamlit as st
from typing import Optional, List, Dict, Any

class PageReaderTool(BaseTool):
    name: str = "page_reader"
    func: str = "read_page"
    description: str = "Read pages from the loaded document. Input should be one of: 'read_current_page', 'next_page', 'previous_page', 'go_to_page:{number}', 'document_summary', 'page_count'."
    
    def _run(self, command: str) -> str:
        if 'document_store' not in st.session_state:
            return "No document store initialized. Please use the document_loader tool first."
            
        if not st.session_state.document_store.get('document_pages'):
            return "No document is currently loaded. Please use the document_loader tool first."
        
        command_lower = command.lower()
        
        try:
            if command_lower == 'read_current_page':
                current_page = st.session_state.document_store['document_pages'][st.session_state.document_store['current_page_index']]
                return f"[Page {st.session_state.document_store['current_page_index'] + 1}/{st.session_state.document_store['total_pages']} of \"{st.session_state.document_store['document_name']}\"]\n\n{current_page.page_content}"
            
            elif command_lower == 'next_page':
                if st.session_state.document_store['current_page_index'] < st.session_state.document_store['total_pages'] - 1:
                    st.session_state.document_store['current_page_index'] += 1
                    current_page = st.session_state.document_store['document_pages'][st.session_state.document_store['current_page_index']]
                    return f"[Page {st.session_state.document_store['current_page_index'] + 1}/{st.session_state.document_store['total_pages']} of \"{st.session_state.document_store['document_name']}\"]\n\n{current_page.page_content}"
                else:
                    return f"You are already at the last page ({st.session_state.document_store['total_pages']}) of the document."
            
            elif command_lower == 'previous_page':
                if st.session_state.document_store['current_page_index'] > 0:
                    st.session_state.document_store['current_page_index'] -= 1
                    current_page = st.session_state.document_store['document_pages'][st.session_state.document_store['current_page_index']]
                    return f"[Page {st.session_state.document_store['current_page_index'] + 1}/{st.session_state.document_store['total_pages']} of \"{st.session_state.document_store['document_name']}\"]\n\n{current_page.page_content}"
                else:
                    return "You are already at the first page of the document."
            
            elif command_lower.startswith('go_to_page:'):
                try:
                    page_num = int(command_lower.split(':')[1])
                    if page_num < 1 or page_num > st.session_state.document_store['total_pages']:
                        return f"Invalid page number. Please specify a page between 1 and {st.session_state.document_store['total_pages']}."
                    st.session_state.document_store['current_page_index'] = page_num - 1
                    current_page = st.session_state.document_store['document_pages'][st.session_state.document_store['current_page_index']]
                    return f"[Page {st.session_state.document_store['current_page_index'] + 1}/{st.session_state.document_store['total_pages']} of \"{st.session_state.document_store['document_name']}\"]\n\n{current_page.page_content}"
                except ValueError:
                    return "Invalid page number format. Please use 'go_to_page:X' where X is a number."
            
            elif command_lower == 'document_summary':
                return f"""Document Information:
- Name: {st.session_state.document_store['document_name']}
- Total Pages: {st.session_state.document_store['total_pages']}
- Current Page: {st.session_state.document_store['current_page_index'] + 1}
- Format: {os.path.splitext(st.session_state.document_store['document_name'])[1][1:].upper()}"""
            
            elif command_lower == 'page_count':
                return f"The document \"{st.session_state.document_store['document_name']}\" contains {st.session_state.document_store['total_pages']} pages."
            
            else:
                return f"Unknown command: \"{command}\". Valid commands are: 'read_current_page', 'next_page', 'previous_page', 'go_to_page:{{number}}', 'document_summary', 'page_count'."
                
        except Exception as error:
            print('Error reading page:', error)
            return f"Error reading page: {str(error)}"

    async def _arun(self, command: str) -> str:
        return self._run(command)