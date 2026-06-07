"""
AI Document Summariser
Uses Groq API (FREE) for ultra-fast intelligent document summarisation with RAG-style chunking
"""
import os
import sys
import json
import re
from datetime import datetime

try:
    from groq import Groq
except ImportError:
    print("Installing groq... run: pip install groq")
    sys.exit(1)

SAMPLE_DOCS = {
    "tech_report": """
    Artificial Intelligence in Healthcare: 2025 Progress Report
    The integration of AI technologies in healthcare has accelerated dramatically in 2025,
    with machine learning models now assisting in diagnostics, drug discovery, and patient
    monitoring across 847 hospitals globally.
    Key Findings:
    1. Diagnostic Accuracy: AI-assisted tools improved early cancer detection by 34%.
    2. Drug Discovery: ML reduced drug screening from 18 months to 6 weeks, saving $2.3 billion.
    3. Patient Monitoring: AI systems reduced ICU readmission rates by 22%.
    Challenges: Data privacy concerns cited by 67% of institutions. Algorithmic bias in minority datasets remains critical.
    Recommendations: Invest in federated learning, diverse datasets, and explainability frameworks.
    """,
    "business_report": """
    Q2 2025 Business Performance Review - RetailCo India
    Total revenue: Rs.2,847 crore, 18.3% year-on-year growth. EBITDA margins improved to 14.2%.
    Online segment: Rs.1,247 crore (43.8% of revenue), growing 34% YoY.
    Physical retail: Rs.1,600 crore, same-store sales growth 8.4%.
    AI personalisation engine launch resulted in 23% increase in average order value.
    Supply chain digitisation reduced inventory costs by Rs.47 crore.
    Outlook: 20-22% revenue growth guided for H2 2025.
    """,
    "research_paper": """
    Abstract: NLP Advances in Low-Resource Languages
    MULTILANG-7B, a transformer model trained on 47 low-resource languages including Tamil and Kannada.
    Outperforms baselines by 12.4 BLEU points on translation benchmarks.
    Achieves 89.3% accuracy on Tamil sentiment classification vs 76.1% previous best.
    91.2 F1 score for Kannada Named Entity Recognition.
    Conclusion: Targeted architectural modifications advance NLP for underrepresented languages.
    """
}

def chunk_document(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 30):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def extract_key_phrases(text):
    sentences = re.split(r'[.!?]+', text)
    key = []
    indicators = ["improved", "reduced", "increased", "key", "critical", "significant"]
    for s in sentences:
        if any(ind in s.lower() for ind in indicators) and len(s.split()) > 5:
            key.append(s.strip())
    return key[:4]

def summarise_document(doc_text, doc_name, summary_type="comprehensive"):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    chunks = chunk_document(doc_text)
    key_phrases = extract_key_phrases(doc_text)

    prompt = f"""Analyse this document and provide a structured summary:

{doc_text}

Respond in this format:
EXECUTIVE SUMMARY: [2-3 sentences]
KEY POINTS:
- [point 1]
- [point 2]
- [point 3]
CRITICAL INSIGHTS: [2 sentences]
SENTIMENT: [Positive/Negative/Neutral]
COMPLEXITY: [Simple/Moderate/Complex]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert document analyst. Be concise and structured."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.3
    )

    return {
        "document": doc_name,
        "word_count": len(doc_text.split()),
        "chunks_processed": len(chunks),
        "key_phrases_detected": key_phrases,
        "summary": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens,
        "model": "llama-3.3-70b-versatile (Groq FREE)",
        "generated_at": datetime.now().isoformat()
    }

def main():
    print("=" * 60)
    print("   AI DOCUMENT SUMMARISER - Powered by Groq API (FREE)")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n No GROQ_API_KEY found.")
        print(" Get free key at: https://console.groq.com")
        print(" Then run: set GROQ_API_KEY=your-key-here")
        return

    print(f"\n API Key detected. Model: llama-3.3-70b-versatile")
    print(f" Processing {len(SAMPLE_DOCS)} documents...\n")
    os.makedirs("output", exist_ok=True)

    for name, text in SAMPLE_DOCS.items():
        print(f"Processing: {name}...")
        result = summarise_document(text, name)
        print(f"\n{'='*55}")
        print(f"Document  : {result['document']}")
        print(f"Words     : {result['word_count']}")
        print(f"Tokens    : {result['tokens_used']}")
        print(f"Model     : {result['model']}")
        print(f"\nSUMMARY:\n{result['summary']}\n")
        with open(f"output/summary_{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved → output/summary_{name}.json")

    print(f"\n{'='*60}")
    print("ALL DOCUMENTS PROCESSED SUCCESSFULLY!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
