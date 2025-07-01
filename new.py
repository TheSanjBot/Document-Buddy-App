import streamlit as st
from streamlit import session_state
import time
import base64
import os
import io
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_bytes
from PIL import Image
from vectors import EmbeddingsManager
from chatbot import ChatbotManager
import base64
import streamlit as st
import streamlit.components.v1 as components

import fitz  # PyMuPDF
from PIL import Image
import io

def displayPDF(file):
    try:
        # Load the file into PyMuPDF
        pdf_data = file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")

        st.markdown("### 📖 PDF Preview (First 3 Pages)")
        for page_num in range(min(3, len(doc))):  # Show up to 3 pages
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            st.image(Image.open(io.BytesIO(img_data)), use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error displaying PDF preview: {e}")


# Initialize session_state variables if not already present
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="DocuBuddy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    st.image("logo-Photoroom.png", use_column_width=True)
    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")
    
    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot"]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 DocuBuddy App")
    st.markdown("""
    Welcome to **Document Buddy App**! 🚀

    **Built using Open Source Stack (DeepSeek R1, BGE Embeddings, and Qdrant Cloud)**

    - **Upload Documents**: Easily upload your PDF documents.
    - **Summarize**: Get concise summaries of your documents.
    - **Chat**: Interact with your documents through our intelligent chatbot.

    Enhance your document management experience with Document Buddy! 😊
    """)

# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface (DeepSeek R1 RAG)")
    st.markdown("---")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: File Uploader and Preview
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            # Display file name and size
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
            
            # Display PDF preview using displayPDF function
            st.markdown("### 📖 PDF Preview (First 3 Pages)")
            displayPDF(uploaded_file)

            uploaded_file.seek(0)
            
            # Save the uploaded file to a temporary location
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store the temp_pdf_path in session_state
            st.session_state['temp_pdf_path'] = temp_pdf_path

    # Column 2: Create Embeddings
    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings")
        if create_embeddings:
            if st.session_state['temp_pdf_path'] is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    # Initialize the EmbeddingsManager
                    embeddings_manager = EmbeddingsManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url=st.secrets["QDRANT_URL"],
                        qdrant_api_key=st.secrets["QDRANT_API_KEY"],
                        collection_name="vector_db"
                    )

                    
                    with st.spinner("🔄 Embeddings are in process..."):
                        # Create embeddings
                        result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        time.sleep(1)  # Optional: To show spinner for a bit longer
                    st.success(result)
                    
                    # Initialize the ChatbotManager after embeddings are created
                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatbotManager(
                            openrouter_api_key=st.secrets["OPENROUTER_API_KEY"],
                            qdrant_url=st.secrets["QDRANT_URL"],
                            qdrant_api_key=st.secrets["QDRANT_API_KEY"],
                            collection_name="vector_db"
                        )

                    
                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Column 3: Chatbot Interface
    with col3:
        st.header("💬 Chat with Document")
        
        if st.session_state['chatbot_manager'] is None:
            st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
        else:
            # Display existing messages
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            # User input
            if user_input := st.chat_input("Type your message here..."):
                # Display user message
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        # Get the chatbot response using the ChatbotManager
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)  # Simulate processing time
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"
                
                # Display chatbot message
                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})


# Footer
st.markdown("---")

