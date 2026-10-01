"""
Feature Engineering & Signature Extraction for Drug Synergy Prediction.
Implements:
1. Drug Resistance Signatures (DRS): differential expression between resistant and sensitive cell lines (mu^R - mu^S)
2. Conventional Drug Signatures (DS): treated vs untreated differential expression
3. Chemical Structure Descriptors: Morgan / physicochemical feature representations
4. Symmetric Pairwise Feature Fusion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Key cancer resistance & response genes identified in the paper (Figures 3 & 4)
MARKER_GENES = [
    "EIF4EBP1", "TRIB3", "SLC1A4", "MAPKAPK3", "CSNK2A2",
    "PDK2", "CHAC1", "FGF19", "XBP1", "TSC22D3",
    "GRB10", "FAT1", "TMEM87", "HMGCS1", "AURKA",
    "PRKAG1", "AKT1", "PIK3CA", "TP53", "EGFR",
    "CDK4", "CCND1", "MYC", "ESR1", "ERBB2",
    "DHFR", "TYMS", "BRCA1", "PARP1", "BCL2"
]

# Supported cell lines with baseline expression profiles
CELL_LINE_INFO = {
    "MCF7": {"tissue": "Breast", "type": "ER+ Breast Adenocarcinoma", "baseline_er": 2.4, "baseline_her2": -0.8},
    "T47D": {"tissue": "Breast", "type": "ER+ PR+ Ductal Carcinoma", "baseline_er": 2.8, "baseline_her2": -0.6},
    "A549": {"tissue": "Lung", "type": "Non-Small Cell Carcinoma (KRAS mutant)", "baseline_er": -1.2, "baseline_her2": 0.2},
    "HCC827": {"tissue": "Lung", "type": "EGFR-Mutated NSCLC (Exon 19 del)", "baseline_er": -1.0, "baseline_her2": 0.8},
    "PC3": {"tissue": "Prostate", "type": "Castration-Resistant Adenocarcinoma", "baseline_er": -1.5, "baseline_her2": -0.2}
}

# Supported oncology drugs with mechanisms
DRUG_DATABASE = {
    "Anastrozole": {"class": "Aromatase Inhibitor", "target": "CYP19A1", "mechanism": "Depletes estrogen synthesis, arresting ER+ breast cancer growth."},
    "Erlotinib": {"class": "EGFR TKI", "target": "EGFR", "mechanism": "Reversible EGFR tyrosine kinase inhibitor blocking proliferative signaling."},
    "Cyclophosphamide": {"class": "Alkylating Agent", "target": "DNA cross-linking", "mechanism": "Forms interstrand DNA crosslinks resulting in lethal replication stress."},
    "Letrozole": {"class": "Aromatase Inhibitor", "target": "CYP19A1", "mechanism": "Suppresses peripheral estrogen biosynthesis in hormone-dependent tumors."},
    "Methotrexate": {"class": "Antifolate (DHFR Inhibitor)", "target": "DHFR/TYMS", "mechanism": "Depletes folate pools and halts thymidylate/purine synthesis for DNA replication."},
    "Lapatinib": {"class": "Dual TKI", "target": "EGFR / ERBB2", "mechanism": "Dual inhibition of EGFR and HER2 tyrosine kinase survival cascades."},
    "Olaparib": {"class": "PARP Inhibitor", "target": "PARP1/2", "mechanism": "Traps PARP at single-strand breaks causing double-strand breaks in HR-deficient cells."},
    "Paclitaxel": {"class": "Taxane", "target": "TUBB (Tubulin)", "mechanism": "Stabilizes microtubule polymers preventing spindle disassembly in mitosis."},
    "Doxorubicin": {"class": "Anthracycline", "target": "TOP2A", "mechanism": "DNA intercalation and topoisomerase II poisoning triggering double-strand breaks."},
    "Tamoxifen": {"class": "SERM", "target": "ESR1", "mechanism": "Competitive estrogen receptor antagonist in breast tissue."}
}


def generate_synthetic_signatures(seed: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generates reproducible feature representations for each drug:
    - DRS: Drug Resistance Signatures (resistant vs sensitive cell lines: mu^R - mu^S)
    - DS: Conventional Drug Signatures (treated vs untreated)
    - Structure: Morgan-style chemical descriptor embeddings
    """
    rng = np.random.RandomState(seed)
    n_genes = len(MARKER_GENES)
    
    drs_dict = {}
    ds_dict = {}
    struct_dict = {}

    for drug_name in DRUG_DATABASE.keys():
        # DRS features have high variance on key resistance pathways
        drs_vec = rng.normal(loc=0.0, scale=0.8, size=n_genes)
        
        # Ground paper-specific signatures (Erlotinib, Anastrozole, etc.)
        if drug_name == "Erlotinib":
            # Upregulated in resistant: EIF4EBP1, TRIB3, SLC1A4, PDK2
            drs_vec[MARKER_GENES.index("EIF4EBP1")] = 1.85
            drs_vec[MARKER_GENES.index("TRIB3")] = 1.45
            drs_vec[MARKER_GENES.index("SLC1A4")] = 1.15
            drs_vec[MARKER_GENES.index("PDK2")] = 2.10
            # Downregulated in resistant: XBP1, TSC22D3, GRB10, FAT1
            drs_vec[MARKER_GENES.index("XBP1")] = -1.35
            drs_vec[MARKER_GENES.index("TSC22D3")] = -1.25
            drs_vec[MARKER_GENES.index("GRB10")] = -1.50
            drs_vec[MARKER_GENES.index("FAT1")] = -1.10
        elif drug_name == "Anastrozole":
            drs_vec[MARKER_GENES.index("ESR1")] = -1.8
            drs_vec[MARKER_GENES.index("DHFR")] = 1.5
            drs_vec[MARKER_GENES.index("CCND1")] = -1.2
            drs_vec[MARKER_GENES.index("AKT1")] = 1.2
        elif drug_name == "Methotrexate":
            drs_vec[MARKER_GENES.index("DHFR")] = 2.4
            drs_vec[MARKER_GENES.index("TYMS")] = 2.1
            drs_vec[MARKER_GENES.index("TP53")] = 1.1
            drs_vec[MARKER_GENES.index("BCL2")] = 0.9

        # DS is conventional treated vs control (less specific to resistance)
        ds_vec = rng.normal(loc=0.1, scale=0.5, size=n_genes)
        
        # Chemical Morgan-style descriptor (binary/continuous property vector)
        struct_vec = rng.uniform(low=0.0, high=1.0, size=64)

        drs_dict[drug_name] = drs_vec
        ds_dict[drug_name] = ds_vec
        struct_dict[drug_name] = struct_vec

    return drs_dict, ds_dict, struct_dict


def get_cell_line_vector(cell_line: str) -> np.ndarray:
    """Returns baseline expression vector for given cell line context."""
    cell_info = CELL_LINE_INFO.get(cell_line, CELL_LINE_INFO["MCF7"])
    n_genes = len(MARKER_GENES)
    rng = np.random.RandomState(abs(hash(cell_line)) % 100000)
    base = rng.normal(loc=0.0, scale=0.3, size=n_genes)
    base[MARKER_GENES.index("ESR1")] = cell_info["baseline_er"]
    base[MARKER_GENES.index("ERBB2")] = cell_info["baseline_her2"]
    return base


def fuse_features(vec_a: np.ndarray, vec_b: np.ndarray, cell_vec: np.ndarray) -> np.ndarray:
    """
    Symmetric Pairwise Feature Fusion:
    Phi(v_A, v_B) = [v_A + v_B,  v_A * v_B,  |v_A - v_B|,  cell_vec]
    Guarantees symmetry: predict(A, B) == predict(B, A).
    """
    add_feat = vec_a + vec_b
    mul_feat = vec_a * vec_b
    diff_feat = np.abs(vec_a - vec_b)
    return np.concatenate([add_feat, mul_feat, diff_feat, cell_vec])
