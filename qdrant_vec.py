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
import concurrent.futures
from PIL import Image
import numpy as np
import platform
import subprocess
from langchain.schema import Document
import tempfile
from streamlit_extras.switch_page_button import switch_page
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
###### injest footer

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
def get_latest_feedback(username):
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()
    get_query = """
    SELECT feedback FROM chat_logs 
    WHERE username ILIKE %s
    ORDER BY created_at DESC
    LIMIT 1;
"""

    cur.execute(get_query, (username,))  
    result = cur.fetchone() 
    feedback = result[0] if result else None

    return feedback
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
if 'username' not in st.session_state:
    st.session_state['username'] = ""
    
username_input = st.text_input("Enter your username to start:", value=st.session_state['username'])
if username_input:
    st.session_state['username'] = username_input
else:
    st.stop()  
# Initialize Streamlit application
 # Log the time after entering the username
logging.info(f"User logged in as: {st.session_state['username']}")
login_time = time.time()
logging.info(f"Time after login: {login_time - start_time:.2f} seconds")
st.title(f"🤖 Bodhee Bot | Welcome, {st.session_state['username']}!")
def save_chat_history_with_feedback():
    if st.session_state.get("messages"):
        # Initialize chat history string with the username
        chat_history = f"Chat History for {st.session_state['username']}:\n\n"
        
        # Append each message with proper role formatting (User or Bot) and feedback
        for i, msg in enumerate(st.session_state['messages']):
            if msg['role'].lower() == "user":
                chat_history += f"{st.session_state['username']}: {msg['content']}\n"
            else:
                chat_history += f"Bot: {msg['content']}\n"
                # Append feedback if available
                if 'feedback' in msg:
                    feedback = 'Positive' if msg['feedback'] else 'Negative'
                    chat_history += f"Feedback: {feedback}\n"
        
        # Define the file path with username included
        file_path = f"{st.session_state['username']}_chat_history_with_feedback.txt"
        
        # Write the chat history to the file
        with open(file_path, "w") as f:
            f.write(chat_history)
        
        return file_path
    return None
#might need to use the old promt template
# Define prompt templates
prompt_templates = {
    "detailed": """
    You are an assistant that provides comprehensive, detailed answers to questions based on
    a given context and try to answer from the documnet. Provide in-depth explanations and cover all aspects of the topic.
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



###BPS related prompt



# bps_prompt_templates = {
#     "detailed": """
#     You are an expert assistant with in-depth knowledge of Bodhee, a scheduling and rescheduling application designed for complex pharmaceutical operations covering Manufacturing, Quality and Maintenance scheduling. 
#     Your role is to provide comprehensive, expert-level guidance to users seeking to understand scheduling concepts and Bodhee’s specific features within a pharmaceutical context.
    
#     Offer detailed explanations and examples that illustrate how Bodhee’s tools support key scheduling needs, such as batch tracking, production timelines, and compliance management. 
#     Reference relevant sections from Bodhee's documentation to guide users on how to perform specific tasks, navigate features, or troubleshoot scheduling challenges within the application.

#     Whenever possible, illustrate how Bodhee can optimize workflows for drug manufacturing timelines, manage overlapping production stages. Include examples such as using Gantt charts within Bodhee for multi-stage production, scheduling time-sensitive compounds. 

#     If diagrams, charts, or feature-specific visuals from Bodhee are available, suggest these to enhance the user's understanding. For example, refer to Bodhee’s timeline view for visualizing production cycles or its compliance tracking tools for regulatory adherence.

#     **Context:**
#     {context}
    
#     **Question:**
#     {question}
    
#     **Guidance:**
#     Emphasize actionable steps using Bodhee’s features to address the user’s needs. For instance, detail how Bodhee can assist in rescheduling due to delays in production, optimizing inventory for batch production. Provide best practices and point out any limitations or considerations that users should be aware of when using Bodhee in pharmaceutical applications.
#     """,
    
#     "concise": """
#     You are a Bodhee expert providing direct, focused answers to questions on pharmaceutical scheduling within the Bodhee application. 
#     Your responses should quickly guide users by referencing Bodhee documentation and explaining relevant features that address their specific needs.

#     **Context:**
#     {context}
    
#     **Question:**
#     {question}
    
#     **Note:** Reference relevant sections in Bodhee’s documentation when they clarify the response. Use brief, specific examples from Bodhee’s pharma-focused features, such as batch production scheduling, timeline views, or compliance tracking tools, but keep answers concise and directly aligned with the question.
#     """
# }



# Keywords that trigger the 'detailed' prompt template
detailed_keywords = ["explain", "detail", "describe", "in-depth", "thorough", "elaborate"]
# Function to choose the appropriate prompt template
def select_prompt_template(question):
    if any(keyword in question.lower() for keyword in detailed_keywords):
        return prompt_templates["detailed"]
    else:
        return prompt_templates["concise"]
    
def generate_prompt_with_memory(selected_prompt_template):
    # Get conversation history
    conversation_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]
    )
    return selected_prompt_template.format(conversation=conversation_history)
vectorstore = None
file_embeddings = OllamaEmbeddings(model='mxbai-embed-large')
pdf_options = {
    "BPS": '/home/localstudio/rag-demo/release/BPS.pdf',
    "Portescap": '/home/localstudio/rag-demo/release/Porterscap.pdf',
    "IT Infrastructure" : '/home/localstudio/rag-demo/release/IT Infrastructure.pdf',
    "SCM" : '/home/localstudio/rag-demo/release/SCM.pdf'

}
vectorstore_paths = {
    "BPS": '/home/localstudio/rag-demo/release/embeddings/all_doc_vectorstore.pkl',
    "Portescap": '/home/localstudio/rag-demo/release/embeddings/all_doc_porter_vectorstore.pkl',
    "IT Infrastructure" : '/home/localstudio/rag-demo/release/embeddings/it_infra.pkl',
    "SCM" : '/home/localstudio/rag-demo/release/embeddings/scm.pkl'
}

# PDF file selection widget
selected_pdf = st.selectbox("Select a Divison", options=list(pdf_options.keys()))
PDF_FILE = pdf_options[selected_pdf]
VECTORSTORE_PATH = vectorstore_paths[selected_pdf]

# Initialize embeddings model
file_embeddings = OllamaEmbeddings(model='mxbai-embed-large')

# Function to embed documents in parallel using ProcessPoolExecutor
def embed_chunk(chunk):
    return file_embeddings.embed_documents([chunk.page_content])

if os.path.exists(VECTORSTORE_PATH):
    st.info("Loading existing vectorstore from disk...")
    logging.info("Loading existing vectorstore...")
    with open(VECTORSTORE_PATH, 'rb') as f:
        vectorstore = pickle.load(f)
    st.success("Vectorstore loaded successfully.")
    logging.info("Vectorstore loaded from disk.")
else:
    # Load PDF and split into chunks if no vectorstore exists
    st.info("No vectorstore found. Creating a new one...")
    logging.info("Loading the PDF...")
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()
    logging.info("PDF loaded.")
    
    # Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)  # Reduced chunk size
    chunks = splitter.split_documents(pages)
    
    #st.info("Creating embeddings for the document...")

    # Show spinner while embeddings are being created
    with st.spinner('Processing embeddings, please wait...'):
        # Create a Streamlit progress bar and initialize tqdm for terminal
        progress_bar = st.progress(0)  # Streamlit progress bar
        total_chunks = len(chunks)
        
        # Start tqdm progress bar in the terminal
        tqdm_bar = tqdm(total=total_chunks, desc="Embedding chunks", unit="chunk")
        
        embeddings_list = []
        
        # Use ProcessPoolExecutor for parallel embedding with max workers
        with ProcessPoolExecutor(max_workers=4) as executor:  # Limit workers
            futures = {executor.submit(embed_chunk, chunk): chunk for chunk in chunks}
            
            # As futures complete, update both progress bars and collect embeddings
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                embedding_result = future.result()
                embeddings_list.append(embedding_result)
                
                # Update tqdm and Streamlit progress bars
                tqdm_bar.update(1)  # Update tqdm bar
                progress_bar.progress((idx + 1) / total_chunks)  # Update Streamlit progress bar
                
                logging.info(f"Processed chunk {idx+1}/{total_chunks}")

        tqdm_bar.close()  # Close tqdm progress bar when done
    
    # Create vector store from the chunks and embeddings
    vectorstore = FAISS.from_documents(chunks, file_embeddings)
    
    # Save the vector store to disk
    st.info("Saving vectorstore to disk...")
    with open(VECTORSTORE_PATH, 'wb') as f:
        pickle.dump(vectorstore, f)
    st.success("Vectorstore created and saved successfully.")
    logging.info("Vectorstore saved to disk.")
# Image retrieval file paths
csv_file_path = '/home/localstudio/rag-demo/release/img_embeddings/all_doc_img_descriptions.csv'
faiss_index_path = '/home/localstudio/rag-demo/release/img_embeddings/bps_image_faiss_index.pkl'
faiss_data_path = '/home/localstudio/rag-demo/release/img_embeddings/bps_faiss_data.pkl'
mapping_file_path = '/home/localstudio/rag-demo/release/img_embeddings/bps_image_summary.csv'
# Load CSV file and verify contents
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error(f"File not found: {csv_file_path}")
    raise
# Check columns in CSV
required_columns = ['image_file', 'description']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        raise KeyError(f"Missing required column: {col}")
# Check if FAISS index and mapping file already exist
if os.path.exists(faiss_index_path) and os.path.exists(faiss_data_path) and os.path.exists(mapping_file_path):
    logging.info("Loading existing FAISS store...")
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
# Load the vectorstore
if VECTORSTORE_PATH is not None:
    if os.path.exists(VECTORSTORE_PATH):
        logging.info("Loading existing vectorstore...")
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
        logging.info("Vectorstore loaded from disk.")
    else:
        logging.info("Loading the PDF...")
        loader_start = time.time()
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        loader_end = time.time()
        logging.info(f"PDF loaded in {loader_end - loader_start:.2f} seconds")
        logging.info("Splitting the document into chunks...")
        split_start = time.time()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        split_end = time.time()
        logging.info(f"Document split into chunks in {split_end - split_start:.2f} seconds")
        logging.info("Creating embeddings and vectorstore...")
        embed_start = time.time()
        embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        vectorstore = FAISS.from_documents(chunks, embeddings)
        embed_end = time.time()
        logging.info(f"Embeddings and vectorstore created in {embed_end - embed_start:.2f} seconds")
        logging.info("Saving vectorstore to disk...")
        with open(VECTORSTORE_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)
        logging.info("Vectorstore saved to disk.")
    logging.info("Initializing retriever and model...")
    retriever = vectorstore.as_retriever()
    model = ChatOllama(model=model_name, temperature=0, streaming=True)
    parser = StrOutputParser()
MAX_MESSAGES = 10
MAX_CONTEXT_MEMORY = 5

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "Bot", "content": "Hello! How can I assist today?","image":None}]
    st.session_state["first_prompt"] = None  # To store the first prompt
    st.session_state["chat_name"] = "Chat-Default"  # Initialize chat_name

# Render chat history
for idx, message in enumerate(st.session_state["messages"]):
    if message["role"] == "user":
        st.chat_message("user").write(f"{message['content']}")
    else:
        st.chat_message("bot").write(f"🤖 {message['content']}")
        if message['image'] is not None:
            st.chat_message("bot").image(message['image'])

        # Check if it's the first message (index 0) and skip feedback buttons
        if idx > 0:  # Skip feedback buttons for the first message
            # Add thumbs up and thumbs down buttons for feedback
            feedback_col = st.columns(2)
            feedback_value = None  # Initialize feedback value
            with feedback_col[0]:
                if st.button("👍", key=f"thumbs_up_{idx}"):
                    if 'feedback' not in message:  # Prevent duplicate feedback storage
                        message['feedback'] = True  # Add feedback
                        feedback_value = True  # Set feedback value
                        print(f"Thumbs Up Feedback Sent for message {idx}")
                        # Update feedback in the database for this specific response
                        insert_chat_data(st.session_state['username'], st.session_state.messages[idx-1]['content'], message['content'], feedback_value)
            with feedback_col[1]:
                if st.button("👎", key=f"thumbs_down_{idx}"):
                    if 'feedback' not in message:  # Prevent duplicate feedback storage
                        message['feedback'] = False  # Add feedback
                        feedback_value = False  # Set feedback value
                        print(f"Thumbs Down Feedback Sent for message {idx}")
                        st.markdown("<p style='color:red;'>Thank you for your feedback! We're sorry to hear that. Please let us know how we can improve.</p>", unsafe_allow_html=True)
                        # Update feedback in the database for this specific response
                        insert_chat_data(st.session_state['username'], st.session_state.messages[idx-1]['content'], message['content'], feedback_value)


# Limit messages in session state
if len(st.session_state["messages"]) > MAX_MESSAGES:
    #st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]
    st.session_state.messages.pop(0)

if question := st.chat_input("Type your question here..."):
    
   
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.show_image = False


    results = img_vectorstore.similarity_search_with_score(question, k=5)
 
    scores_dict = {}
    result_ids = []
    for doc, score in results:
        if score < 600.00:
            scores_dict[doc.metadata['faiss_id']] = score
    sorted_dict_asc = dict(sorted(scores_dict.items(), key=lambda item: item[1]))
   
    for key in sorted_dict_asc.keys():
        result_ids.append(key)
   
    result_paths = mapping_df[mapping_df['faiss_id'].isin(result_ids)]['image_path'].to_list()
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    document_context = '\n'.join(map(lambda doc: doc.page_content, retrieved_docs))

    # Store the first prompt for the chat name
    if st.session_state["first_prompt"] is None:
        st.session_state["first_prompt"] = question[:5]  # Get the first 5 letters
        st.session_state["chat_name"] = f"Chat-{st.session_state['first_prompt']}"  # Create chat name

    # Display the chat name
    st.write(f"**Chat Name:** {st.session_state['chat_name']}")
    conversation_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"][-MAX_CONTEXT_MEMORY:])
    combined_context = f"{document_context}\n\n{conversation_history}"
    #prompt_text = select_prompt_template.format(context=combined_context, question=question)
    selected_prompt_template = select_prompt_template(question)
    prompt_text = selected_prompt_template.format(context=combined_context, question=question)

    logging.info("Retrieving context from vectorstore...")
    retrieve_start = time.time()
    print("Retrieved Document Context:", document_context)
    retrieve_end = time.time()
    logging.info(f"Context retrieved in {retrieve_end - retrieve_start:.2f} seconds")
    print("Combined Context:", combined_context)
    # Prepare the chain for the response generation
    chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
        }
        | PromptTemplate.from_template(prompt_text)
        | model
        | parser
    )

    # Display user question in chat
    st.chat_message("user").write(question)

    logging.info("Generating answer from the model...")
    model_start = time.time()

    try:
        response_placeholder = st.empty()
        response = ""
        
        # Prepare the chain result with combined context and question
        chain_result = {"context": combined_context, "question": question}  # Pass combined context and question
        
        for chunk in chain.stream(chain_result):
            response += chunk  # Append each chunk to the response string
            response_placeholder.write(response)

    except Exception as e:
        response = f"An error occurred: {str(e)}"
        logging.error(f"Error while generating response: {response}")


    img = None
    if result_paths:    
        img = Image.open(result_paths[0])
        st.image(img)    
    st.session_state.messages.append({"role": "BOT", "content": response, 'image': img})
    # Always save the chat history to the database after generating the response
    username = st.session_state['username']

    # Feedback for the current bot response (immediate feedback after response generation)
    feedback_col = st.columns(2)
    feedback_value = None  # Initialize feedback value for the new input
    with feedback_col[0]:
        if st.button("👍", key=f"thumbs_up_{len(st.session_state.messages) - 1}"):
            print("Thumbs Up button clicked")
            if 'feedback' not in st.session_state.messages[-1]:  # Prevent duplicate feedback storage
                st.session_state.messages[-1]['feedback'] = True  # Add feedback
                feedback_value = True
                print("Feedback set to True")
                # Update chat data with feedback
                update_chat_data(username, question, response, feedback_value)  # Save feedback as 'True'
    with feedback_col[1]:
        if st.button("👎", key=f"thumbs_down_{len(st.session_state.messages) - 1}"):
            print("Thumbs Down button clicked")
            if 'feedback' not in st.session_state.messages[-1]:  # Prevent duplicate feedback storage
                st.session_state.messages[-1]['feedback'] = False  # Add feedback
                feedback_value = False
                print("Feedback set to False")
                # Update chat data with feedback
                update_chat_data(username, question, response, feedback_value)  # Save feedback as 'False'
    model_end = time.time()
    logging.info(f"Model generated answer in {model_end - model_start:.2f} seconds")
# Download chat history button
if st.button("Download Chat History"):
    file_path = save_chat_history_with_feedback()  # Updated to include feedback
    if file_path:
        st.write(f"Chat history saved as {file_path}")
        st.download_button(
            label="Download Chat History",
            data=open(file_path, "r").read(),
            file_name=f"{st.session_state['username']}_chat_history_with_feedback.txt"  # Updated file name
        )

