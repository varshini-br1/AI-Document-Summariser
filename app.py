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

# ── Sample Documents ──────────────────────────────────────────────────────────
SAMPLE_DOCS = {
    "tech_report": """
    Artificial Intelligence in Healthcare: 2025 Progress Report
    
    Executive Summary:
    The integration of AI technologies in healthcare has accelerated dramatically in 2025, 
    with machine learning models now assisting in diagnostics, drug discovery, and patient 
    monitoring across 847 hospitals globally. This report examines the current landscape, 
    key achievements, and challenges faced by healthcare AI implementations.
    
    Key Findings:
    1. Diagnostic Accuracy: AI-assisted diagnostic tools have improved early cancer detection 
    rates by 34% compared to traditional methods, with false positive rates reduced by 28%.
    2. Drug Discovery: Machine learning algorithms have reduced the initial drug screening 
    phase from 18 months to approximately 6 weeks, saving pharmaceutical companies an 
    estimated $2.3 billion in R&D costs.
    3. Patient Monitoring: IoT-connected AI systems have reduced ICU readmission rates by 
    22% through predictive analytics and real-time anomaly detection.
    
    Challenges:
    Data privacy concerns remain the primary barrier to wider adoption, with 67% of 
    healthcare institutions citing HIPAA compliance as a significant challenge. 
    Additionally, algorithmic bias in datasets representing minority populations continues 
    to be a critical issue requiring immediate attention.
    
    Recommendations:
    Investment in federated learning approaches, diverse training datasets, and robust 
    explainability frameworks is essential for sustainable AI healthcare integration.
    """,

    "business_report": """
    Q2 2025 Business Performance Review — RetailCo India
    
    Financial Highlights:
    RetailCo India delivered strong Q2 2025 results with total revenue of Rs.2,847 crore, 
    representing 18.3% year-on-year growth. EBITDA margins improved to 14.2% from 11.8% 
    in Q2 2024, driven by operational efficiency initiatives and favourable raw material costs.
    
    Segment Performance:
    The Online segment contributed Rs.1,247 crore (43.8% of total revenue), growing 34% YoY 
    as mobile commerce adoption accelerated in Tier 2 and Tier 3 cities. Physical retail 
    delivered Rs.1,600 crore, with same-store sales growth of 8.4%.
    
    Strategic Initiatives:
    The company launched its AI-powered personalisation engine in Q2, resulting in a 23% 
    increase in average order value and 15% improvement in customer retention rates.
    Supply chain digitisation reduced inventory holding costs by Rs.47 crore.
    
    Outlook:
    Management guides for 20-22% revenue growth in H2 2025, supported by the festive 
    season and planned expansion into 3 new metro markets. Capital expenditure of Rs.180 
    crore is planned for fulfilment centre upgrades.
    """,

    "research_paper": """
    Abstract: Natural Language Processing Advances in Low-Resource Languages
    
    This paper presents MULTILANG-7B, a transformer-based language model trained on 
    47 low-resource languages including Tamil, Kannada, and Swahili. Our model achieves 
    state-of-the-art performance on machine translation tasks, outperforming existing 
    baselines by 12.4 BLEU points on average across evaluation benchmarks.
    
    Methodology:
    We employ a cross-lingual transfer learning approach with a modified attention mechanism 
    that preserves morphological structures unique to agglutinative languages. Training was 
    conducted on a distributed GPU cluster using 2.4 billion parameters over 6 weeks.
    
    Results:
    MULTILANG-7B achieves 89.3% accuracy on sentiment classification tasks for Tamil,
    compared to 76.1% for the previous best model. On Named Entity Recognition tasks,
    our model shows 91.2 F1 score for Kannada text.
    
    Conclusion:
    This work demonstrates that targeted architectural modifications combined with 
    carefully curated multilingual datasets can significantly advance NLP capabilities 
    for underrepresented languages, with important implications for digital inclusion.
    """
}

# ── RAG-style chunking ────────────────────────────────────────────────────────
def chunk_document(text: str, chunk_size: int = 500) -> list:
    """Split document into overlapping chunks for RAG-style processing."""
    words = text.split()
    chunks = []
    overlap = 50
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def extract_key_phrases(text: str) -> list:
    """Extract potential key phrases using simple heuristics."""
    sentences = re.split(r'[.!?]+', text)
    key = []
    indicators = ["key", "important", "significant", "main", "primary", "critical",
                  "achieved", "improved", "reduced", "increased", "major"]
    for s in sentences:
        if any(ind in s.lower() for ind in indicators) and len(s.split()) > 5:
            key.append(s.strip())
    return key[:5]

# ── AI Summarisation using Groq ───────────────────────────────────────────────
def summarise_document(doc_text: str, doc_name: str, summary_type: str = "comprehensive") -> dict:
    """Use Groq API to generate intelligent document summary."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

    chunks = chunk_document(doc_text)
    key_phrases = extract_key_phrases(doc_text)
    word_count = len(doc_text.split())

    prompts = {
        "comprehensive": f"""Analyse this document and provide a structured summary:

Document: {doc_text}

Provide your response in this exact format:
**EXECUTIVE SUMMARY** (2-3 sentences):
[summary here]

**KEY POINTS** (3-5 bullet points):
* [point 1]
* [point 2]
* [point 3]

**CRITICAL INSIGHTS**:
[2-3 sentences about the most important findings]

**ACTION ITEMS** (if applicable):
[recommendations or next steps]

**SENTIMENT**: [Positive/Negative/Neutral]
**COMPLEXITY**: [Simple/Moderate/Complex]""",

        "brief": f"Summarise this document in 3 sentences maximum: {doc_text}",

        "bullet": f"""Extract the 5 most important points from this document as bullet points:
{doc_text}

Format: * [Point]: [Brief explanation]"""
    }

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are an expert document analyst. Provide clear, structured, and insightful summaries."
            },
            {
                "role": "user",
                "content": prompts.get(summary_type, prompts["comprehensive"])
            }
        ],
        max_tokens=1000,
        temperature=0.3
    )

    return {
        "document": doc_name,
        "summary_type": summary_type,
        "word_count": word_count,
        "chunks_processed": len(chunks),
        "key_phrases_detected": key_phrases,
        "summary": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens,
        "model": "llama3-8b-8192 (via Groq)",
        "generated_at": datetime.now().isoformat()
    }

# ── Main CLI ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   AI DOCUMENT SUMMARISER — Powered by Groq API (FREE)")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️  GROQ_API_KEY not set.")
        print("\n📋 HOW TO GET FREE GROQ API KEY:")
        print("   1. Go to: https://console.groq.com")
        print("   2. Sign up with your email (FREE)")
        print("   3. Go to API Keys → Create Key")
        print("   4. Copy the key")
        print("\n   Then run:")
        print("   Windows : set GROQ_API_KEY=your-key-here")
        print("   Mac/Linux: export GROQ_API_KEY=your-key-here")
        print("\n   Then run: python app.py")
        print("\n[Running Demo Mode — showing pipeline structure]\n")
        demo_mode()
        return

    print(f"\n✅ API Key detected. Using model: llama3-8b-8192")
    print(f"\nAvailable documents:")
    for i, name in enumerate(SAMPLE_DOCS, 1):
        print(f"  {i}. {name}")

    print("\nProcessing all documents...\n")
    os.makedirs("output", exist_ok=True)

    all_results = []
    for name, text in SAMPLE_DOCS.items():
        print(f"🔄 Processing: {name}...")
        result = summarise_document(text, name, "comprehensive")
        all_results.append(result)

        print(f"\n{'─'*60}")
        print(f"📄 Document     : {result['document']}")
        print(f"📊 Word Count   : {result['word_count']}")
        print(f"🔢 Chunks       : {result['chunks_processed']}")
        print(f"🔑 Key Phrases  : {len(result['key_phrases_detected'])} detected")
        print(f"⚡ Tokens Used  : {result['tokens_used']}")
        print(f"🤖 Model        : {result['model']}")
        print(f"\n📝 SUMMARY:\n")
        print(result['summary'])
        print()

        # Save individual result
        with open(f"output/summary_{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved → output/summary_{name}.json")

    # Save combined report
    report = {
        "pipeline_run": datetime.now().isoformat(),
        "total_documents": len(all_results),
        "total_tokens": sum(r["tokens_used"] for r in all_results),
        "model": "llama3-8b-8192 (Groq - FREE)",
        "documents_processed": [r["document"] for r in all_results]
    }
    with open("output/pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ ALL DOCUMENTS PROCESSED!")
    print(f"   Total documents : {report['total_documents']}")
    print(f"   Total tokens    : {report['total_tokens']}")
    print(f"   Results saved   → output/ folder")
    print(f"{'='*60}")

def demo_mode():
    """Show pipeline structure without API call."""
    doc = list(SAMPLE_DOCS.values())[0]
    chunks = chunk_document(doc)
    phrases = extract_key_phrases(doc)

    print("📋 PIPELINE DEMONSTRATION (Demo Mode):")
    print(f"  ✅ Document loaded     : {len(doc.split())} words")
    print(f"  ✅ RAG chunks created  : {len(chunks)} chunks")
    print(f"  ✅ Key phrases found   : {len(phrases)}")
    for p in phrases[:3]:
        print(f"     • {p[:75]}...")
    print(f"\n  🤖 [Groq API call would go here → LLaMA3 generates structured summary]")
    print(f"\n  📊 Expected output structure:")
    sample = {
        "executive_summary": "AI in healthcare has improved diagnostic accuracy by 34%...",
        "key_points": ["Cancer detection +34%", "Drug screening 6 weeks", "ICU readmissions -22%"],
        "sentiment": "Positive",
        "complexity": "Moderate"
    }
    print(f"  {json.dumps(sample, indent=4)}")
    print(f"\n✅ Set GROQ_API_KEY to enable FREE live AI summarisation.")
    print(f"   Get free key at: https://console.groq.com")

if __name__ == "__main__":
    main()
