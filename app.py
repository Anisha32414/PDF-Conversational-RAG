import os
import validators
import gradio as gr

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    YoutubeLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Prompt for summarization
prompt_template = """
Provide a concise summary of the following content.

Keep the summary within 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


def summarize_url(groq_api_key, generic_url):

    # Validate inputs
    if not groq_api_key or not groq_api_key.strip():
        return "❌ Please provide the Groq API Key."

    if not generic_url or not generic_url.strip():
        return "❌ Please provide a URL."

    if not validators.url(generic_url):
        return "❌ Please enter a valid URL. It can be a YouTube or Website URL."

    try:

        # Initialize Groq LLM
        llm = ChatGroq(
            api_key=groq_api_key,
            model="openai/gpt-oss-safeguard-20b",
            temperature=0
        )

        # Load YouTube or Website content
        if "youtube.com" in generic_url or "youtu.be" in generic_url:

            loader = YoutubeLoader.from_youtube_url(
                generic_url,
                add_video_info=False
            )

        else:

            loader = UnstructuredURLLoader(
                urls=[generic_url],
                ssl_verify=False,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/139.0.0.0 Safari/537.36"
                    )
                }
            )

        # Load documents
        docs = loader.load()

        if not docs:
            return "❌ No content could be extracted from the URL."

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200
        )

        final_docs = text_splitter.split_documents(docs)

        # Create summarization chain
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=True
        )

        # Generate summary
        output_summary = chain.invoke({
            "input_documents": final_docs
        })

        # Return summary
        return output_summary["output_text"]

    except Exception as e:

        return f"❌ Error: {str(e)}"


# -------------------------------
# Gradio Interface
# -------------------------------

with gr.Blocks(
    title="LangChain: Summarize Text from YouTube or Website"
) as demo:

    gr.Markdown(
        """
        # 🦜 LangChain: Summarize Text from YouTube or Website

        Enter your **Groq API Key** and a **YouTube or Website URL**
        to generate a concise summary.
        """
    )

    groq_api_key = gr.Textbox(
        label="Groq API Key",
        placeholder="Enter your Groq API key",
        type="password"
    )

    generic_url = gr.Textbox(
        label="URL",
        placeholder="Enter a YouTube or Website URL"
    )

    summarize_button = gr.Button(
        "Summarize",
        variant="primary"
    )

    output = gr.Markdown(
        label="Summary"
    )

    summarize_button.click(
        fn=summarize_url,
        inputs=[
            groq_api_key,
            generic_url
        ],
        outputs=output
    )


# Launch application
if __name__ == "__main__":
    demo.launch()
