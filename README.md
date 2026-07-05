# 🌐 PatentLens AI: Enterprise Research Copilot

PatentLens AI is an advanced, enterprise-grade automated system designed to analyze technical statements and claim phrases against global prior-art structures. The tool combines **BM25 Lexical (Keyword) Matching** with dense **ChromaDB Semantic (Vector) Embeddings** to isolate overlapping patent chunks.

---
Live Demo:  [https://patent-lens-copilot-dz2lmhk66htmw8jenely54.streamlit.app/]

## 🚀 Key Architectural Features
* **Two-Stage Retrieval Pipeline:** Integrates exact term matching (Lexical BM25) and contextual similarity analysis (ChromaDB Vector Space) into a unified search matrix.
* **Neural Re-ranking Engine:** Utilizes a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) model to evaluate and sort retrieved fragments based on semantic relevance scores.
* **Interactive Dynamic Workspace:** A multi-tab Streamlit dashboard offering deep-dive claim intersections, strategic alignment mapping, and alternative risk matrices.
* **On-the-fly Cloud Compilation:** Fully adapted for isolated cloud runtime deployments using native `pysqlite3` binary adjustments.

---

## 🛠️ File Structure & Repository Map

```text
patent-lens-copilot/
│
├── app.py                   # Main Streamlit UI engine with state persistence
├── ingest_patents.py        # Database ingestion pipeline & vector space builder
├── requirements.txt         # Environment dependencies manifest (version-locked)
├── patent_chunks.csv        # Prior-art reference technical raw data
└── PatentLens_DB/           # Automatically generated local vector database directory
⚙️ Local Installation & Development Setup
To run this platform locally on a Windows or Linux machine, execute the following steps line-by-line:

1. Clone the Repository
Bash
git clone [https://github.com/SAKSHI1243/patent-lens-copilot.git](https://github.com/SAKSHI1243/patent-lens-copilot.git)
cd patent-lens-copilot

2. Configure a Virtual Environment
Bash
python -m venv venv
# On Windows Activation:
venv\Scripts\activate
# On Linux/macOS Activation:
source venv/bin/activate
3. Install Required Dependencies
Bash
pip install -r requirements.txt
4. Run Vector Ingestion (Build local spaces)
Bash
python ingest_patents.py
5. Launch the Dashboard
Bash
streamlit run app.py
📊 App Operational Interface Workflows
🔍 Find Similar Patents: Input an original concept and inspect conflict risk metrics (🟢 LOW RISK, ⚡ MODERATE OVERLAP, ⚠️ HIGH COLLISION).

📊 Compare Ideas (Strategic Cross-Match): View live alignment scores along with structured AI workaround hints.

📄 Generate Research Reports: Review an automated summary manifest and download a .txt corporate compliance document natively.
