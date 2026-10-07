import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    YoutubeLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


## Streamlit app
st.set_page_config(
    page_title="LangChain: Summarize text from YT or Website",
    page_icon="🦜"
)

st.title("LangChain: Summarize Text from YouTube or Website")
st.subheader("Summarize URL")


## Get the Groq API Key and URL (YouTube or Website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-safeguard-20b",
    temperature=0
)


## Prompt for summarization
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


generic_url = st.text_input(
    "URL",
    label_visibility="collapsed"
)


if st.button("Summarize the content from YT or Website"):

    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():

        st.error("Please provide the information")

    elif not validators.url(generic_url):

        st.error("Please enter a valid URL. It can be a YT or Website URL")

    else:

        try:

            with st.spinner("Loading content and generating summary..."):

                ## Loading the website or YT video data
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
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
                        }
                    )

                docs = loader.load()


                ## Check whether content was loaded
                if not docs:

                    st.error("No content could be extracted from the URL.")

                else:

                    ## Split documents into smaller chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=5000,
                        chunk_overlap=200
                    )

                    final_docs = text_splitter.split_documents(docs)


                    ## Chain for summarization
                    chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        map_prompt=prompt,
                        combine_prompt=prompt,
                        verbose=True
                    )


                    ## Generate summary
                    output_summary = chain.invoke({
                        "input_documents": final_docs
                    })


                    ## Display only the final summary
                    st.success(output_summary["output_text"])


        except Exception as e:

            st.exception(e)