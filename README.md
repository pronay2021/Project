# 🎙️ Voice-Based Course Assistant API

This project is a **FastAPI-powered AI Voice Assistant** designed to help users inquire about available courses via voice. It uses **OpenAI Whisper** for speech-to-text, **GPT-4** for intelligent responses, and **gTTS** for text-to-speech replies. It also supports contextual memory and semantic search over a course dataset using **FAISS** and **LangChain**.

---

## 🚀 Features

- 🎤 **Speech Recognition**: Converts spoken audio to text using `Whisper`.
- 🤖 **Smart AI Assistant**: Uses GPT-4 to provide course-related information.
- 🧠 **Contextual Memory**: Maintains conversation history during the session.
- 🔍 **Semantic Search**: Retrieves course info using FAISS + Sentence Transformers.
- 🔊 **Voice Response**: Converts text replies to downloadable `.mp3` files.
- 🌐 **REST APIs**:
  - `POST /start-call` – Start/reset the conversation
  - `GET /conversation` – View conversation history
  - `POST /respond` – Upload an audio file and receive transcript, AI response, and voice reply

---

## 📁 Project Structure

```
.
├── main.py               # FastAPI backend
├── courses.json          # Course data used for semantic search
├── .env                  # Stores your OpenAI API key
├── requirements.txt      # Python dependencies
```

---

## 🧪 Example Workflow

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

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/voice-course-assistant.git
cd voice-course-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your OpenAI API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
```

### 5. Run the App

```bash
uvicorn main:app --reload
```

---

## 🔗 API Endpoints

| Method | Endpoint         | Description                           |
|--------|------------------|---------------------------------------|
| POST   | `/start-call`    | Starts or resets the conversation     |
| GET    | `/conversation`  | Returns full chat history             |
| POST   | `/respond`       | Accepts audio and returns AI response |

---

## 📌 Notes

- Works with `wav` and `mp3` audio files.
- Uses `Whisper`'s base model – you can change it to `small`, `medium`, etc.
- `courses.json` must follow this structure:

```json
{
  "courses": [
    {
      "course_name": "Example Course",
      "course_description": "What the course covers...",
      "course_learning": "Skills gained...",
      "course_duration": "4 weeks",
      "course_fee": "$100",
      "discount_fee": "$80"
    }
  ]
}
```

---

## 📄 License

MIT License. Feel free to use and modify this project for your own needs.

---

## 💬 Credits

Built with ❤️ using FastAPI, Whisper, LangChain, and GPT-4.
