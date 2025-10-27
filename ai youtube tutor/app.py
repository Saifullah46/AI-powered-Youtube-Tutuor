import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your Google Gemini API Key here or via .env
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", " your gemini api ")

# Initialize HuggingFace embeddings (lightweight and CPU-friendly)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Function: Extract transcript from YouTube
def get_youtube_transcript(url):
    """Extract transcript text safely from YouTube video URL"""
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_data = transcript.fetch()

        text_chunks = []
        for item in transcript_data:
            if isinstance(item, dict) and "text" in item:
                text_chunks.append(item["text"])
            elif hasattr(item, "text"):
                text_chunks.append(item.text)

        text = " ".join(text_chunks).strip()
        return text

    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("⚠️ No English transcript found.")
    except VideoUnavailable:
        st.error("🚫 This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("🌍 Could not retrieve transcript (may not be available in your region).")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    """Save transcript text to file"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


# Streamlit UI
st.set_page_config(page_title="AI YouTube Tutor", page_icon="🎓", layout="centered")
st.title("🎓 AI-Powered YouTube Tutor")
st.write("Extract knowledge from YouTube lectures and ask your own questions interactively!")

video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

# --- Process Video ---
if st.button("Process Video"):
    if not video_url:
        st.warning("⚠️ Please enter a valid YouTube URL.")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ Please set your GOOGLE_API_KEY in the .env file!")
            st.info("Get your free API key from: https://makersuite.google.com/app/apikey")
        else:
            with st.spinner("⏳ Extracting transcript..."):
                transcript_text = get_youtube_transcript(video_url)

            if transcript_text:
                with st.spinner("⚙️ Processing transcript and creating knowledge base..."):
                    try:
                        # Save transcript locally
                        save_transcript_to_file(transcript_text)

                        # Split transcript into chunks
                        loader = TextLoader("transcript.txt", encoding="utf-8")
                        documents = loader.load()

                        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        docs = splitter.split_documents(documents)

                        # Create FAISS vector store
                        vectorstore = FAISS.from_documents(docs, embeddings)
                        retriever = vectorstore.as_retriever()

                        # Initialize Gemini model (gemini-1.5-flash = fast + free)
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

                        # Create QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            return_source_documents=True
                        )

                        # Store in Streamlit session
                        st.session_state.qa_chain = qa_chain
                        st.session_state.transcript_processed = True
                        st.success("✅ Transcript processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                        st.info("Check your GOOGLE_API_KEY and ensure dependencies are installed.")


# --- Ask Questions ---
if "qa_chain" in st.session_state and st.session_state.get("transcript_processed"):
    st.divider()
    st.subheader("💬 Ask a Question about the Video")
    user_question = st.text_input("Your question:", placeholder="e.g., What are the main topics discussed?")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.qa_chain({"query": user_question})
                st.write("**Answer:**", result["result"])

                with st.expander("📄 View Source Excerpts"):
                    for i, doc in enumerate(result.get("source_documents", [])):
                        st.write(f"**Source {i+1}:**")
                        st.write(doc.page_content[:300] + "...")
                        st.divider()
            except Exception as e:
                st.error(f"Error generating answer: {e}")


# --- Footer ---
st.divider()
st.caption("🚀 Powered by Google Gemini AI & LangChain | Created by Saiff 🧠")
