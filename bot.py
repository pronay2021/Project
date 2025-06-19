import whisper
import sounddevice as sd
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
import keyboard

try:
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    
    try:
        model = whisper.load_model("base")          # Loading the Whisper model
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    samplerate = 16000
    channels = 1

    
    try:
        with open("courses.json", "r") as f:
            courses_data = json.load(f)["courses"]
    except Exception as e:
        raise RuntimeError(f"Failed to load courses.json: {str(e)}")

    
    try:
        documents = [
            Document(
                page_content=f"Course Name: {c['course_name']}\nCourse Description: {c['course_description']}\nCourse Learning: {c['course_learning']}\nDuration: {c['course_duration']}\nFee: {c['course_fee']}\nDiscount Fee: {c['discount_fee']}",
                metadata={"course_name": c["course_name"]}
            )
            for c in courses_data
        ]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")         # Loading the Whisper model
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        raise RuntimeError(f"Failed to initialize FAISS vector store: {str(e)}")

    # Defining the memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # The main Prompt
    custom_prompt = PromptTemplate.from_template("""
    You are an AI-powered Voice Sales Agent for a company named 'Interactive Cares' tasked with pitching online courses to potential customers in a natural, friendly, and professional manner.
    You should reply like you are talking in a call. Your goal is to engage the customer, qualify their needs, pitch the courses, handle objections and attempt to close the conversation by scheduling
    a follow-up or securing a commitment.

    **Conversation Stages**:
    1. **Introduction**: Greet the customer by name (if provided) and introduce yourself as an AI sales agent from the company offering online courses. Keep it friendly and concise.
    2. **Qualification**: Ask 2-3 open-ended questions to understand the customer's interests, goals, or experience (e.g., "Are you interested in advancing your career in AI or machine learning?" or "What kind of skills are you looking to develop?").
    3. **Pitch**: Based on the customer's responses, pitch the answer. Highlight the duration, special offer price, and key benefits tailored to their needs.
    4. **Objection Handling**: Address common objections (e.g., price: "I understand cost is a concern, but our special offer of 5000 Tk makes it affordable, and we offer flexible payment plans"; time: "The course is designed to fit busy schedules with self-paced learning"; relevance: "This course covers practical skills like LLMs that are in high demand"). Provide concise, empathetic responses.
    5. **Closing**: Attempt to secure a commitment (e.g., "Would you like to schedule a follow-up call to discuss enrollment details?") or guide them toward the next steps (e.g., "I can send you more information about the course—would that work for you?").

    **Instructions**:
    - Use the customer's name (if provided) to personalize the conversation.
    - Access course details from courses.json to provide accurate information. Default to pitching the Python course unless the customer's responses suggest another course is more relevant.
    - Maintain a friendly, professional tone and keep responses concise (2-3 sentences per response unless more detail is needed).
    - If the customer expresses disinterest or wants to end the conversation (e.g., says "exit," "not interested," or "bye"), respond politely (e.g., "Thank you for your time! Feel free to reach out if you change your mind.") and indicate the call should end.
    - If the customer's input is unclear, ask a clarifying question (e.g., "Could you share a bit more about what you're looking for in a course?").
    - Incorporate the conversation history to maintain context and avoid repetition.
    - If an objection is raised, address it directly and pivot back to the pitch or closing as appropriate.

    **Output**:
    - Respond with a natural, conversational reply that follows the current stage of the conversation (introduction, qualification, pitch, objection handling, or closing).

    {context}
    Chat History:
    {chat_history}

    Human: {question}
    AI:
    """)

    #LLM and retrieval chain
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            verbose=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM or chain: {str(e)}")

    #Welcome Message
    try:
        welcome_message = "Hello, I am speaking from Interactive Cares. Can I know your name sir?"
        welcome_filename = f"welcome_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        tts = gTTS(welcome_message)
        tts.save(welcome_filename)
        os.system(f"start {welcome_filename}")  
        print(welcome_message)
    except Exception as e:
        print(f"Welcome message error: {str(e)}")

    while True:
        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Callback status: {status}")
            audio_data.append(indata.copy())

        print("\n🎵 Press 'S' to start calling...")
        try:
            keyboard.wait('s')  
            print("\n🎙️ Press Enter to stop.")
            with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
                input()
        except Exception as e:
            print(f"Recording error: {str(e)}")
            continue

        if not audio_data:
            print("No audio data captured!")
            continue

        
        try:
            audio = np.concatenate(audio_data).flatten()
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(model.device)
            transcription = whisper.decode(model, mel, whisper.DecodingOptions(language="en", fp16=False)).text
            print("🗣️ You:", transcription)
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            continue

        # Exit condition
        if transcription.lower().strip() in ["exit", "quit", "bye"]:
            print("👋 Exiting. Goodbye!")
            break

       
        try:
            response = chain.run(transcription)
            print("AI Assistant:", response)
        except Exception as e:
            print(f"Chain response error: {str(e)}")
            continue
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reply_{timestamp}.mp3"
            tts = gTTS(response)
            tts.save(filename)
            os.system(f"start {filename}")  
        except Exception as e:
            print(f"TTS or playback error: {str(e)}")
            continue

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")