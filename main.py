from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import whisper
import soundfile as sf
import numpy as np
from gtts import gTTS
import os
from dotenv import load_dotenv
import json
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Load Whisper model
model = whisper.load_model("base")

# Load course data
with open("courses.json", "r") as f:
    courses_data = json.load(f)["courses"]

# Prepare documents and FAISS vector store
documents = [
    Document(
        page_content=f"Course Name: {c['course_name']}\nCourse Description: {c['course_description']}\nCourse Learning: {c['course_learning']}\nDuration: {c['course_duration']}\nFee: {c['course_fee']}\nDiscount Fee: {c['discount_fee']}",
        metadata={"course_name": c["course_name"]}
    )
    for c in courses_data
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt
custom_prompt = PromptTemplate.from_template("""
You are an AI-powered Voice Sales Agent for a company named 'Interactive Cares'...
[TRUNCATED for brevity — keep full prompt from your original code]
{context}
Chat History:
{chat_history}

Human: {question}
AI:
""")

# Chain
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    verbose=False
)

# FastAPI app
app = FastAPI()

@app.post("/start-call")
def start_call():
    global memory
    memory.clear()
    return {"message": "Conversation started. Memory cleared."}

@app.get("/conversation")
def get_conversation():
    history = memory.chat_memory.messages
    conversation = [{"type": m.type, "content": m.content} for m in history]
    return {"conversation": conversation}

@app.post("/respond")
async def respond(audio: UploadFile = File(...)):
    try:
        # Read audio file
        data, samplerate = sf.read(audio.file)
        audio_data = np.mean(data, axis=1) if len(data.shape) > 1 else data
        audio_data = audio_data.astype(np.float32)  # 🔧 This line fixes the dtype issue


        # Transcription
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_data)).to(model.device)
        transcription = whisper.decode(model, mel, whisper.DecodingOptions(language="en", fp16=False)).text.strip()
        print(f"🗣️ User said: {transcription}")

        # Exit condition
        if transcription.lower() in ["exit", "quit", "bye"]:
            return JSONResponse(content={"response": "Thank you for your time! Goodbye.", "transcription": transcription})

        # Generate response
        response = chain.run(transcription)
        print(f"🤖 Assistant: {response}")

        # Generate TTS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reply_{timestamp}.mp3"
        tts = gTTS(response)
        tts.save(filename)

        return {
            "transcription": transcription,
            "response": response,
            "audio_file": filename
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
