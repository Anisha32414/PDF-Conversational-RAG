import os
import tempfile

import gradio as gr
import spaces

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader


# --------------------------------------------------
# HuggingFace Embeddings
# --------------------------------------------------

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# --------------------------------------------------
# ZeroGPU startup probe
# --------------------------------------------------

@spaces.GPU(duration=1)
def zerogpu_startup_probe():
    return None


# --------------------------------------------------
# Process PDFs
# --------------------------------------------------

def process_pdfs(api_key, uploaded_files):

    if not api_key or not api_key.strip():
        return None, "❌ Please enter your Groq API key."

    if not uploaded_files:
        return None, "❌ Please upload at least one PDF."

    try:

        documents = []

        # ------------------------------------------
        # Load all PDFs
        # ------------------------------------------

        for file_path in uploaded_files:

            loader = PyPDFLoader(file_path)

            docs = loader.load()

            documents.extend(docs)

        if not documents:
            return None, "❌ Could not extract any text from the PDFs."

        # ------------------------------------------
        # Split documents
        # ------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = text_splitter.split_documents(documents)

        # ------------------------------------------
        # Create Chroma vector store
        # ------------------------------------------

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        return (
            retriever,
            f"✅ Successfully processed {len(uploaded_files)} PDF(s) "
            f"and created {len(splits)} document chunks."
        )

    except Exception as e:

        return None, f"❌ Error while processing PDFs: {str(e)}"


# --------------------------------------------------
# Create Conversational RAG Chain
# --------------------------------------------------

def create_rag_chain(api_key, retriever, store):

    llm = ChatGroq(
        api_key=api_key,
        model_name="openai/gpt-oss-safeguard-20b",
        temperature=0
    )

    # ------------------------------------------
    # Contextualize question
    # ------------------------------------------

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. "
        "Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as it is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # ------------------------------------------
    # Answer question
    # ------------------------------------------

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    # ------------------------------------------
    # Session history
    # ------------------------------------------

    def get_session_history(
        session: str
    ) -> BaseChatMessageHistory:

        if session not in store:
            store[session] = ChatMessageHistory()

        return store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(
    api_key,
    session_id,
    user_input,
    retriever,
    store,
    chat_history
):

    if not api_key or not api_key.strip():
        return chat_history, "❌ Please enter your Groq API key.", store

    if retriever is None:
        return (
            chat_history,
            "❌ Please upload and process your PDF(s) first.",
            store
        )

    if not user_input or not user_input.strip():
        return chat_history, "", store

    if not session_id or not session_id.strip():
        session_id = "default_session"

    try:

        # ------------------------------------------
        # Create RAG chain
        # ------------------------------------------

        conversational_rag_chain = create_rag_chain(
            api_key,
            retriever,
            store
        )

        # ------------------------------------------
        # Ask question
        # ------------------------------------------

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        answer = response["answer"]

        # ------------------------------------------
        # Update Gradio chat
        # ------------------------------------------

        chat_history = chat_history or []

        chat_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return chat_history, "", store

    except Exception as e:

        return (
            chat_history,
            f"❌ Error: {str(e)}",
            store
        )


# --------------------------------------------------
# Clear Chat
# --------------------------------------------------

def clear_chat():

    return [], {}


# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

with gr.Blocks(
    title="Conversational RAG with PDF Uploads"
) as demo:

    gr.Markdown(
        """
        # 📚 Conversational RAG with PDF Uploads

        Upload one or more PDFs and ask questions about their content
        using **Groq + LangChain + ChromaDB**.
        """
    )

    # ------------------------------------------
    # API Key
    # ------------------------------------------

    api_key = gr.Textbox(
        label="Groq API Key",
        placeholder="Enter your Groq API key",
        type="password"
    )

    # ------------------------------------------
    # Session ID
    # ------------------------------------------

    session_id = gr.Textbox(
        label="Session ID",
        value="default_session",
        placeholder="Enter a session ID"
    )

    # ------------------------------------------
    # PDF Upload
    # ------------------------------------------

    uploaded_files = gr.File(
        label="Upload PDF Files",
        file_types=[".pdf"],
        file_count="multiple",
        type="filepath"
    )

    process_button = gr.Button(
        "Process PDFs",
        variant="primary"
    )

    process_status = gr.Markdown()

    # ------------------------------------------
    # Hidden state for retriever
    # ------------------------------------------

    retriever_state = gr.State(None)

    # ------------------------------------------
    # Chat History Store
    # ------------------------------------------

    store_state = gr.State({})

    # ------------------------------------------
    # Chatbot
    # ------------------------------------------

    chatbot = gr.Chatbot(
        label="Chat",
        type="messages",
        height=500
    )

    # ------------------------------------------
    # User question
    # ------------------------------------------

    user_input = gr.Textbox(
        label="Your Question",
        placeholder="Ask something about your PDFs...",
        lines=2
    )

    ask_button = gr.Button(
        "Ask",
        variant="primary"
    )

    response_status = gr.Markdown()

    clear_button = gr.Button(
        "Clear Chat"
    )

    # ------------------------------------------
    # Process PDFs
    # ------------------------------------------

    process_button.click(
        fn=process_pdfs,
        inputs=[
            api_key,
            uploaded_files
        ],
        outputs=[
            retriever_state,
            process_status
        ]
    )

    # ------------------------------------------
    # Ask question
    # ------------------------------------------

    ask_button.click(
        fn=ask_question,
        inputs=[
            api_key,
            session_id,
            user_input,
            retriever_state,
            store_state,
            chatbot
        ],
        outputs=[
            chatbot,
            response_status,
            store_state
        ]
    )

    # Press Enter to ask
    user_input.submit(
        fn=ask_question,
        inputs=[
            api_key,
            session_id,
            user_input,
            retriever_state,
            store_state,
            chatbot
        ],
        outputs=[
            chatbot,
            response_status,
            store_state
        ]
    )

    # ------------------------------------------
    # Clear chat
    # ------------------------------------------

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chatbot,
            store_state
        ]
    )


# --------------------------------------------------
# Launch
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch()
