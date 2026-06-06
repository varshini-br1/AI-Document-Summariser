# 🤖 AI Document Summariser — Powered by Claude API

An intelligent document summarisation tool that uses Anthropic's Claude API with RAG-style chunking to process and summarise documents. Supports multiple summary types: comprehensive, brief, and bullet-point format.

## 🚀 Features
- 🔄 RAG-style document chunking for large documents
- 🤖 Claude API integration for intelligent summarisation
- 📊 3 summary modes: Comprehensive, Brief, Bullet Points
- 🔑 Key phrase extraction using NLP heuristics
- 📋 Structured JSON output with metadata
- 💡 Demo mode (works without API key)

## 🛠️ Tech Stack
- **Python 3.x** — Core application
- **Anthropic Claude API** — AI summarisation engine
- **RAG Pipeline** — Chunk-based document processing
- **Prompt Engineering** — Structured output formatting

## 📂 Project Structure
```
ai-document-summariser/
├── app.py               # Main application
├── requirements.txt
└── README.md
```

## ⚙️ How to Run
```bash
git clone https://github.com/Varshini-BR/ai-document-summariser.git
cd ai-document-summariser
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

python app.py
```

## 📊 Sample Output
```
EXECUTIVE SUMMARY:
AI integration in healthcare has improved diagnostic accuracy by 34% 
and reduced drug discovery time from 18 months to 6 weeks...

KEY POINTS:
• Cancer detection improved by 34%
• Drug screening reduced to 6 weeks
• ICU readmission rates down 22%

SENTIMENT: Positive
COMPLEXITY: Complex
```

## 🔌 API Used
- **Anthropic Claude** (`claude-sonnet-4-20250514`)
- Endpoint: `/v1/messages`

## 👩‍💻 Author
**Varshini B R** — Computer Science Engineering Graduate
- 📧 varshinibr13@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/varshini-b-r-248767266)
