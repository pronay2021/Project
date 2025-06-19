# 🎙️ AI-powered Voice Sales Agent

This project is a voice calling agent that accepts user input as audio and responds with both audio and text. It uses OpenAI's Whisper model for speech-to-text transcription, GPT-4 as the large language model (LLM) to generate intelligent responses, and Google's gTTS for converting the text responses back into speech. The system leverages FAISS for efficient semantic search within the course data and uses LangChain as the framework to manage conversational retrieval and memory, enabling a seamless voice-based AI conversational experience.


---

## 🚀 Features

- 🎤 **Speech Recognition**: Converts spoken audio to text using `Whisper`.
- 🤖 **Smart AI Assistant**: Uses GPT-4 to provide course-related information.
- 🧠 **Contextual Memory**: Maintains conversation history during the session.
- 🔍 **Semantic Search**: Retrieves course info using FAISS + Sentence Transformers.
- 🔊 **Voice Response**: Converts text replies to downloadable `.mp3` files.
- 🌐 **FAST APIs**:
  - `POST /start-call` – Start/reset the conversation
  - `GET /conversation` – View conversation history
  - `POST /respond` – Upload an audio file and receive transcript, AI response and voice reply

---

## 📁 Project Structure

```
.
├── main.py               
├── courses.json          
├── .env                  
├── requirements.txt
├── bot.py        
```

---

## 🧪 Backend API

1. `POST /start-call` – Initializes the session and clears previous memory.
2. `POST /respond` – Upload a `.wav` or `.mp3` audio file.
3. The system will:
   - Transcribe the audio
   - Retrieve relevant course info
   - Generate a GPT-4-powered response
   - Save a `.mp3` file with the voice response
4. `GET /conversation` – Retrieve the full conversation history

---

## 📦 Setup Instructions


### 1. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate 
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your OpenAI API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the App 

```bash
python bot.py
```
### 5. Run the App with API

```bash
uvicorn main:app --reload
```
---


