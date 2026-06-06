"""
AI Document Summariser
Uses Anthropic Claude API for intelligent document summarisation with RAG-style chunking
"""
import os
import sys
import json
import re
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
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
    RetailCo India delivered strong Q2 2025 results with total revenue of ₹2,847 crore, 
    representing 18.3% year-on-year growth. EBITDA margins improved to 14.2% from 11.8% 
    in Q2 2024, driven by operational efficiency initiatives and favourable raw material costs.
    
    Segment Performance:
    The Online segment contributed ₹1,247 crore (43.8% of total revenue), growing 34% YoY 
    as mobile commerce adoption accelerated in Tier 2 and Tier 3 cities. Physical retail 
    delivered ₹1,600 crore, with same-store sales growth of 8.4%.
    
    Strategic Initiatives:
    The company launched its AI-powered personalisation engine in Q2, resulting in a 23% 
    increase in average order value and 15% improvement in customer retention rates.
    Supply chain digitisation reduced inventory holding costs by ₹47 crore.
    
    Outlook:
    Management guides for 20-22% revenue growth in H2 2025, supported by the festive 
    season and planned expansion into 3 new metro markets. Capital expenditure of ₹180 
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

# ── AI Summarisation ─────────────────────────────────────────────────────────
def summarise_document(doc_text: str, doc_name: str, summary_type: str = "comprehensive") -> dict:
    """Use Claude API to generate intelligent document summary."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

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
• [point 1]
• [point 2]
• [point 3]

**CRITICAL INSIGHTS**:
[2-3 sentences about the most important findings]

**ACTION ITEMS** (if applicable):
[recommendations or next steps]

**SENTIMENT**: [Positive/Negative/Neutral]
**COMPLEXITY**: [Simple/Moderate/Complex]""",

        "brief": f"Summarise this in 3 sentences max: {doc_text}",

        "bullet": f"""Extract the 5 most important points from this document as bullet points:
{doc_text}

Format: • [Point]: [Brief explanation]"""
    }

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompts.get(summary_type, prompts["comprehensive"])}]
    )

    return {
        "document": doc_name,
        "summary_type": summary_type,
        "word_count": word_count,
        "chunks_processed": len(chunks),
        "key_phrases_detected": key_phrases,
        "summary": message.content[0].text,
        "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
        "generated_at": datetime.now().isoformat()
    }

# ── Main CLI ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   AI DOCUMENT SUMMARISER — Powered by Claude API")
    print("=" * 60)
    print("\nAvailable documents:")
    for i, name in enumerate(SAMPLE_DOCS, 1):
        print(f"  {i}. {name}")

    print("\nSummary types: comprehensive | brief | bullet")
    print("\n[Demo Mode — Using sample documents]")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not set. Running in demo mode (showing pipeline structure).")
        demo_mode()
        return

    for name, text in list(SAMPLE_DOCS.items())[:1]:
        print(f"\n🔄 Processing: {name}...")
        result = summarise_document(text, name, "comprehensive")
        print(f"\n{'─'*60}")
        print(f"📄 Document    : {result['document']}")
        print(f"📊 Word Count  : {result['word_count']}")
        print(f"🔢 Chunks      : {result['chunks_processed']}")
        print(f"🔑 Key Phrases : {len(result['key_phrases_detected'])} detected")
        print(f"⚡ Tokens Used : {result['tokens_used']}")
        print(f"\n📝 SUMMARY:\n{result['summary']}")

        with open(f"output_{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Full result saved → output_{name}.json")

def demo_mode():
    """Show pipeline structure without API call."""
    print("\n📋 PIPELINE DEMONSTRATION:")
    doc = list(SAMPLE_DOCS.values())[0]
    chunks = chunk_document(doc)
    phrases = extract_key_phrases(doc)
    print(f"  ✅ Document loaded: {len(doc.split())} words")
    print(f"  ✅ Chunked into: {len(chunks)} chunks for RAG processing")
    print(f"  ✅ Key phrases detected: {len(phrases)}")
    for p in phrases:
        print(f"     • {p[:80]}...")
    print("\n  🤖 [API call would go here → Claude generates structured summary]")
    print("\n  📊 Expected output structure:")
    sample = {"executive_summary": "...", "key_points": ["..."],
              "critical_insights": "...", "sentiment": "Positive", "complexity": "Moderate"}
    print(f"  {json.dumps(sample, indent=4)}")
    print("\n✅ Set ANTHROPIC_API_KEY to enable live AI summarisation.")

if __name__ == "__main__":
    main()
