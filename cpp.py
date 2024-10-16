import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter
import logging
import pandas as pd
import os
from tqdm import tqdm
import pickle
from PIL import Image
import numpy as np
import platform
import subprocess
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.schema import Document
import tempfile
from streamlit_extras.switch_page_button import switch_page
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from datetime import datetime
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

############### nEEd to add diffetnt sort of memeroy funcanlaty to set the context

def get_db_connection():
    conn = psycopg2.connect(
        host="172.20.10.64",
        database="postgres",
        user="postgres",
        password="7PrlfcME06m29tZ&gClO5Preu9",
        sslmode='require',  # Require SSL
        sslcert='/home/localstudio/rag-demo/DB_cert/client-cert.pem',  # Path to the client certificate
        sslkey='/home/localstudio/rag-demo/DB_cert/client-key.pem',    # Path to the client key
        sslrootcert='/home/localstudio/rag-demo/DB_cert/ca-cert.pem'
    )
    cursor = conn.cursor()
    cursor.execute("SET search_path TO conversation_history;")
    return conn





def insert_chat_data(username, user_message, bot_response, feedback):
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()

    # SQL query to insert data into the PostgreSQL table
    insert_query = """
        INSERT INTO chat_logs (username, user_message, bot_response, feedback, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    created_at = datetime.now()  # Timestamp for the chat interaction
    cur.execute(insert_query, (username, user_message, bot_response, feedback, created_at))

    conn.commit()  # Commit the transaction
    cur.close()  # Close the cursor
    conn.close()  # Close the connection


# Update chat data function
def update_chat_data(username, user_message, bot_response, feedback):
    print(f"Updating chat data for {username}: user_message={user_message}, bot_response={bot_response}, feedback={feedback}")
    
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()

    try:
        update_query = """
            WITH recent_chat AS (
                SELECT id  -- Assuming there is an 'id' column as a primary key
                FROM chat_logs 
                WHERE username = %s AND user_message = %s AND bot_response = %s
                ORDER BY created_at DESC 
                LIMIT 1
            )
            UPDATE chat_logs 
            SET feedback = %s 
            WHERE id IN (SELECT id FROM recent_chat)
        """
        cur.execute(update_query, (username, user_message, bot_response, feedback))
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error updating feedback: {str(e)}")
    finally:
        cur.close()  # Close the cursor
        conn.close()  # Close the connection

img_embeddings = OllamaEmbeddings(model='bge-m3')
st.sidebar.header("Model Selection 🧠")
model_options = {
    "Llama 3.1(7B)": "llama3.1",
    "Codellama": "codellama",
    "llava-llama3(Muti-model)" : "llava-llama3",
    "llama3.2 (3B)" : "llama3.2",
    "Mistral":"mistral",
    "qwen2.5-coder" : "qwen2.5-coder"
}
selected_model = st.sidebar.selectbox("Select Language Model", options=list(model_options.keys()))
model_name = model_options[selected_model]
# Initialize logging
logging.basicConfig(level=logging.INFO)
start_time = time.time()

# Username handling
if 'username' not in st.session_state:
    st.session_state['username'] = ""

username_input = st.text_input("Enter your username to start:", value=st.session_state['username'])
if username_input:
    st.session_state['username'] = username_input
else:
    st.stop()

# Log the user login
logging.info(f"User logged in as: {st.session_state['username']}")
login_time = time.time()
logging.info(f"Time after login: {login_time - start_time:.2f} seconds")

st.title(f"🤖 Bodhee Bot | Welcome, {st.session_state['username']}!")

# Function to save chat history with feedback
def save_chat_history_with_feedback():
    if st.session_state.get("messages"):
        chat_history = f"Chat History for {st.session_state['username']}:\n\n"
        for i, msg in enumerate(st.session_state['messages']):
            if msg['role'].lower() == "user":
                chat_history += f"{st.session_state['username']}: {msg['content']}\n"
            else:
                chat_history += f"Bot: {msg['content']}\n"
                if 'feedback' in msg:
                    feedback = 'Positive' if msg['feedback'] else 'Negative'
                    chat_history += f"Feedback: {feedback}\n"
        
        file_path = f"{st.session_state['username']}_chat_history_with_feedback.txt"
        with open(file_path, "w") as f:
            f.write(chat_history)
        return file_path
    return None

# Embedding models
img_embeddings = OllamaEmbeddings(model='bge-m3')
file_embeddings = OllamaEmbeddings(model='mxbai-embed-large')

# Upload files
uploaded_file = st.file_uploader("Upload a file (PDF, Excel, CSV, TXT)", type=["pdf", "xlsx", "csv", "txt"])
if uploaded_file:
    file_name = os.path.splitext(uploaded_file.name)[0]
    VECTORSTORE_PATH = os.path.join("/home/localstudio/rag-demo/rag_chatbot/upload_vector", f"{file_name}_vectorstore.pkl")

    if os.path.exists(VECTORSTORE_PATH):
        st.info("Loading existing vectorstore from disk...")
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        st.info("Creating a new vectorstore...")
        if uploaded_file.name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name
            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(pages)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            documents = [Document(page_content='\n'.join(df.iloc[i:i+10].to_string() for i in range(0, len(df), 10)))]
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            documents = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            document = Document(page_content=content)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents([document])

        def process_chunks(batch):
            return file_embeddings.embed_documents([chunk.page_content for chunk in batch])

        start_time_file = time.time()
        with st.spinner("Creating embeddings and vectorstore..."):
            batch_size = 10
            embeddings_list = []
            with tqdm(total=len(chunks), desc="Vectorizing chunks", unit="chunk") as pbar:
                with ThreadPoolExecutor() as executor:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        embeddings = executor.submit(process_chunks, batch)
                        embeddings_list.extend(embeddings.result())
                        pbar.update(len(batch))

            vectorstore = FAISS.from_documents(chunks, file_embeddings)
            with open(VECTORSTORE_PATH, 'wb') as f:
                pickle.dump(vectorstore, f)

        end_time_file = time.time()
        st.success(f"Embeddings created successfully in {end_time_file - start_time_file:.2f} seconds!")

else:
    pdf_options = {
    "BPS": '/home/localstudio/rag-demo/local-rag-example/DES1005-BPS_Scheduling_Engine_Design_Document_v1.0.pdf',
    "CPA": '/home/localstudio/rag-demo/local-rag-example/Change Point Analyzer -Training Manual 1.pdf',
    "Material": '/home/localstudio/rag-demo/local-rag-example/Material Constraint Training Manual (1).pdf',
    "BPS-Equipment Constraint" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-1-Neewee Bodhee Production Scheduler -Planner Constraints Training Module - Equipment.pdf",
    "BPS-Employee Constraints" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-2-Neewee Bodhee Production Scheduler -Planner Constraints Training Module - Employee.pdf",
    "BPS-Material Constraints" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-3-Neewee Bodhee Production Scheduler -Planner Constraints Training Module - Material.pdf",
    "BPS-OptimizationObjectives & Assumption" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-4-Neewee Bodhee Production Scheduler -Planner -Objective and Optimization Training Manual.pdf",
    "BPS-Micro Planner/Scheduler": "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-017- Neewee Bodhee Production Scheduler - Micro Planner Training Manual.pdf",
    "BPS-Change Point Analyzer": "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-018- Neewee Bodhee Production Scheduler - Change Point Analyzer.pdf"
}
    vectorstore_paths = {
        "BPS": '/home/localstudio/rag-demo/rag_chatbot/embeddings/bps.pkl',
        "CPA": '/home/localstudio/rag-demo/rag_chatbot/embeddings/cpa.pkl',
        "Material": '/home/localstudio/rag-demo/rag_chatbot/embeddings/materail.pkl',
        "BPS-Equipment Constraint": '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Equipment_Constraint.pkl',
        "BPS-Employee Constraints" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Employee_Constraints.pkl',
        "BPS-Material Constraints" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Material_Constraints.pkl',
        "BPS-OptimizationObjectives & Assumption" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-OptimizationObjectives_Assumption.pkl',
        "BPS-Micro Planner/Scheduler" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/bps_micro_planner.pkl',
        "BPS-Change Point Analyzer" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Change_Point_Analyzer.pkl',

    }

    selected_pdf = st.selectbox("Select a PDF", options=list(pdf_options.keys()))
    PDF_FILE = pdf_options[selected_pdf]
    VECTORSTORE_PATH = vectorstore_paths[selected_pdf]

    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        loader = PyPDFLoader(PDF_FILE)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(chunks, img_embeddings)
        with open(VECTORSTORE_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)

# Image retrieval
csv_file_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling description.csv'
faiss_index_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling_image_faiss_index.pkl'
faiss_data_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling_faiss_data.pkl'
mapping_file_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling_image_summary.csv'

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error(f"File not found: {csv_file_path}")
    raise

required_columns = ['image_file', 'description']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        raise KeyError(f"Missing required column: {col}")

if os.path.exists(faiss_index_path) and os.path.exists(faiss_data_path) and os.path.exists(mapping_file_path):
    with open(faiss_index_path, 'rb') as f:
        index = pickle.load(f)
    with open(faiss_data_path, 'rb') as f:
        docstore, index_to_docstore_id = pickle.load(f)
    img_vectorstore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=img_embeddings)
    mapping_df = pd.read_csv(mapping_file_path)
else:
    st.write("FAISS index not found. Creating a new FAISS store.")
    documents = [Document(page_content=row['description'], metadata={'faiss_id': i, 'image_file': row['image_file']})
                 for i, row in df.iterrows()]
    img_vectorstore = FAISS.from_documents(documents, img_embeddings)
    image_paths = df['image_file'].tolist()
    image_path_mapping = {i: image_paths[i] for i in range(len(image_paths))}
    with open(faiss_index_path, 'wb') as f:
        pickle.dump(img_vectorstore.index, f)
    with open(faiss_data_path, 'wb') as f:
        pickle.dump((img_vectorstore.docstore, img_vectorstore.index_to_docstore_id), f)
    mapping_df = pd.DataFrame(list(image_path_mapping.items()), columns=['faiss_id', 'image_path'])
    mapping_df.to_csv(mapping_file_path, index=False)






# Define your custom prompt templates as shown
prompt_templates = {
    "detailed": """
    You are an assistant that provides comprehensive, detailed answers to questions based on
    a given context. Provide in-depth explanations and cover all aspects of the topic.

    Context: {context}

    Question: {question}
    """,
    
    "concise": """
    You are an assistant that provides concise, to-the-point answers to questions based on
    a given context. Keep your responses short and clear.

    Context: {context}

    Question: {question}
    """
}

# Keywords that trigger the 'detailed' prompt template
detailed_keywords = ["explain", "detail", "describe", "in-depth", "thorough", "elaborate"]

# Function to choose the appropriate prompt template
def select_prompt_template(question):
    if any(keyword in question.lower() for keyword in detailed_keywords):
        return prompt_templates["detailed"]
    else:
        return prompt_templates["concise"]
# Chat and Memory Initialization
if VECTORSTORE_PATH:
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)

    model = ChatOllama(model=model_name)

    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Handle user input
    user_input = st.text_input("Ask a question:")
    if user_input:
        # Select the appropriate prompt template based on the user's input
        selected_template = select_prompt_template(user_input)

        # Create the chat prompt template using the selected template
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", selected_template.format(context="Relevant context here", question=user_input)),
            ("human", user_input),
        ])

        # Create the chain with the selected prompt
        chain = LLMChain(
            llm=model,
            prompt=chat_prompt_template,
            memory=st.session_state["memory"],  
            verbose=True
        )

        # Run the chain and get the response
        response = chain.run(question=user_input)
        st.write(f"Bot: {response}")

        # Thumbs up/down feedback
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button("👍"):
                st.session_state["memory"].chat_memory[-1]["feedback"] = True
                st.success("Thanks for your positive feedback!")
        with feedback_col2:
            if st.button("👎"):
                st.session_state["memory"].chat_memory[-1]["feedback"] = False
                st.error("Thanks for your feedback. We'll work to improve!")

# Save chat history as a downloadable text file
if st.button("Save Chat History with Feedback"):
    file_path = save_chat_history_with_feedback()
    if file_path:
        with open(file_path, "rb") as f:
            st.download_button("Download Chat History", data=f, file_name=file_path)
