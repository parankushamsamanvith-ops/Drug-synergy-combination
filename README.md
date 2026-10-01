# SynRes-AI: Drug Synergy & Resistance Prediction Prototype

This prototype implements the concepts from the research paper:
> **"Improving synergistic drug combination prediction with signature-based gene expression features in oncology"**  
> *Frontiers in Pharmacology (2025)*  
> Authors: Mozhgan Mozaffarilegha and Sajjad Gharaghani

---

## 🌟 Key Concept

Traditional drug synergy models rely primarily on chemical structures (e.g. Morgan Fingerprints) or broad drug signatures (treated vs untreated). This platform integrates **Drug Resistance Signatures (DRS)**, which capture differential gene expression between **resistant and sensitive cell lines** ($\mu^R - \mu^S$) derived from GDSC $IC_{50}$ thresholds and LINCS L1000 transcriptomics.

---

## 🚀 How to Run the Frontend

### Option 1: Streamlit Dashboard (Python)

1. Activate your Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

### Option 2: Standalone Interactive Web Artifact (Browser)

Double-click or open the generated standalone HTML interface directly in your browser:
* Located in your artifact folder: `synergy_explorer_prototype.html`

---

## 📊 Features Included

- **Cell Line Context:** Select breast cancer (`MCF7`, `T47D`) or lung cancer cell lines.
- **Dynamic Synergy Scoring:** Displays predicted $S$-score and classification badge ($S > 10$ is synergistic).
- **Interactive Volcano Plot:** Highlights key resistance genes (*EIF4EBP1*, *TRIB3*, *SLC1A4*, *XBP1*).
- **Pathway Enrichment:** Compares DRS vs conventional Drug Signatures (DS) across cell cycle, p53, and PI3K/Akt pathways.
- **Benchmark Comparisons:** Live metrics comparison from the paper (Table 2 & Table 3).
