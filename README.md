# 🤖 AI Document Summariser — Powered by Groq API (FREE)

An intelligent document summarisation tool that uses **Groq's free API** with LLaMA3 model and RAG-style chunking to process and summarise documents instantly. Supports multiple summary types: comprehensive, brief, and bullet-point format.

## 🚀 Features
- 🔄 RAG-style document chunking for large documents
- 🤖 Groq API + LLaMA3 integration (completely FREE)
- ⚡ Ultra-fast inference (Groq is 10x faster than most LLM APIs)
- 📊 3 summary modes: Comprehensive, Brief, Bullet Points
- 🔑 Key phrase extraction using NLP heuristics
- 📋 Structured JSON output with metadata
- 💡 Demo mode (works without API key)

## 🛠️ Tech Stack
- **Python 3.x** — Core application
- **Groq API** — Free LLM inference engine
- **LLaMA3 (8B)** — Open source AI model
- **RAG Pipeline** — Chunk-based document processing
- **Prompt Engineering** — Structured output formatting

## 📂 Project Structure
```
ai-document-summariser/
├── app.py               # Main application
├── requirements.txt
├── output/              # Generated summaries (JSON)
└── README.md
```

## ⚙️ How to Run

### Step 1 — Get FREE Groq API Key
1. Go to **https://console.groq.com**
2. Sign up with your email (no credit card needed)
3. Go to **API Keys** → **Create Key**
4. Copy your key

### Step 2 — Install & Run
```bash
git clone https://github.com/varshini-br1/AI-Document-Summariser.git
cd AI-Document-Summariser
pip install -r requirements.txt

# Set your FREE API key
# Windows:
set GROQ_API_KEY=your-key-here

# Mac/Linux:
export GROQ_API_KEY=your-key-here

python app.py
```

## 📊 Sample Output
```
AI DOCUMENT SUMMARISER — Powered by Groq API (FREE)

Processing: tech_report...

EXECUTIVE SUMMARY:
AI integration in healthcare has dramatically improved outcomes,
with diagnostic accuracy up 34% and drug discovery time reduced
from 18 months to just 6 weeks...

KEY POINTS:
* Cancer detection improved by 34%
* Drug screening reduced to 6 weeks  
* ICU readmission rates down 22%

SENTIMENT: Positive
COMPLEXITY: Moderate

Total tokens used: 847
```

## 🔌 API Used
- **Groq API** (Free tier — no credit card required)
- **Model**: LLaMA3-8B-8192
- **Endpoint**: `/openai/v1/chat/completions`

## 👩‍💻 Author
**Varshini B R** — Computer Science Engineering Graduate
- 📧 varshinibr13@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/varshini-b-r-248767266)
