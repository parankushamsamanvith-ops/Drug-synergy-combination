"""
Real Benchmark Dataset Loader & Feature Extractor.
Loads real datasets from the Frontiers in Pharmacology (2025) study:
- Benchmarks: OncologyScreen (4,176 pairs), O'Neil (23,062 pairs), DrugComb (330,917 pairs), etc.
- Features: Real Drug Resistance Signatures (DRS) DEGsenWithMASK.npy (mu^R - mu^S)
- Multi-omics Cell Line Graph & Expression features (985 cell lines)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmarks")
SIGNATURE_DIR = os.path.join(DATA_DIR, "signatures")

# Mapping canonical SMILES in DEGsenWithMASK to standard oncology drug names
KNOWN_DRUG_NAMES = {
    "CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)c1cnccn1)B(O)O": "Bortezomib",
    "O=P1(N(CCCl)CCCl)NCCCO1": "Cyclophosphamide",
    "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1": "Dasatinib",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1": "Erlotinib",
    "Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1": "Gemcitabine",
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c4)n3)o1": "Lapatinib",
    "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1": "Methotrexate",
    "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C(O)(C(OC(=O)C(O)C(NC(=O)c2ccccc2)c2ccccc2)C1=O)C(C)(C)O": "Paclitaxel",
    "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1": "Sorafenib",
    "Cn1nnc2c(C(N)=O)ncn2c1=O": "Temozolomide",
    "CCC1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3c(CN(C)C)c(O)ccc3nc2-1": "Topotecan",
    "CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C)C3C(OC(C)=O)(C(=O)OC)C(O)C2N43)C1)CC": "Vinblastine",
    "CCC1=CC2CN(C1)Cc1c([nH]c3ccccc13)C(C(=O)OC)(c1cc3c(cc1OC)N(C=O)C1C(OC(C)=O)(C(=O)OC)C(O)C2N31)C": "Vincristine",
    "O=C(CCCCCCC(=O)Nc1ccccc1)NO": "Vorinostat (SAHA)",
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1": "Gefitinib",
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1": "Vandetanib",
    "Cc1ccc(C(=O)Nc2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1": "Imatinib",
    "O=C(c1ccc(N(CCCl)CCCl)cc1)c1ccc(N(CCCl)CCCl)cc1": "Chlorambucil Analog",
    "O=C(NC(Cc1ccccc1)C(=O)NC(Cc1ccccc1)C(=O)NO)c1ccccc1": "Histone Deacetylase Inhibitor",
    "CN1CCN(Cc2ccc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc2)CC1": "Nilotinib",
    "O=C(O)C1(Cc2cccc(Nc3nccs3)n2)CCC(Oc2cccc(Cl)c2F)CC1": "Seliciclib",
    "NC1(c2ccc(-c3nc4ccn5c(=O)[nH]nc5c4cc3-c3ccccc3)cc2)CCC1": "MK-2206 (Akt Inhibitor)"
}

NAME_TO_SMILES = {v: k for k, v in KNOWN_DRUG_NAMES.items()}

# Cache dictionaries in memory for performance
_DRS_CACHE = None
_DS_CACHE = None
_CELL_CACHE = None


def get_drug_name(smiles: str) -> str:
    """Returns friendly drug name if known, else shortened SMILES representation."""
    if smiles in KNOWN_DRUG_NAMES:
        return KNOWN_DRUG_NAMES[smiles]
    return smiles[:18] + "..."


def load_drug_signatures(feature_mode: str = "DRS") -> Dict[str, np.ndarray]:
    """
    Loads real precomputed drug transcriptomic signatures from disk.
    - DRS: DEGsenWithMASK.npy (Differential expression between resistant vs sensitive cells: mu^R - mu^S)
    - DS: DEGWithMASK.npy (Differential expression between treated vs untreated cells)
    Returns: Dict[smiles, 165-dim numpy vector]
    """
    global _DRS_CACHE, _DS_CACHE
    if feature_mode.upper() == "DRS" and _DRS_CACHE is not None:
        return _DRS_CACHE
    if feature_mode.upper() == "DS" and _DS_CACHE is not None:
        return _DS_CACHE

    file_name = "DEGsenWithMASK.npy" if feature_mode.upper() == "DRS" else "DEGWithMASK.npy"
    file_path = os.path.join(SIGNATURE_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Signature file {file_path} not found.")

    raw_dict = np.load(file_path, allow_pickle=True)[0]
    out_dict = {}

    for smiles, val in raw_dict.items():
        if isinstance(val, (tuple, list)) and len(val) >= 1:
            expr_vec = np.array(val[0], dtype=np.float32)
            mask_vec = np.array(val[1], dtype=np.float32) if len(val) > 1 else np.ones_like(expr_vec)
            # Combine expression and mask for 330-dim or use expression
            out_dict[smiles] = expr_vec * mask_vec
        else:
            out_dict[smiles] = np.array(val, dtype=np.float32)

    if feature_mode.upper() == "DRS":
        _DRS_CACHE = out_dict
    else:
        _DS_CACHE = out_dict

    return out_dict


def load_cell_line_features(dim_reduce: int = 64) -> Dict[str, np.ndarray]:
    """
    Loads multi-omics features for 985 cancer cell lines from
    '985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy'.
    Computes a compressed summary vector (dim: dim_reduce) for each cell line.
    """
    global _CELL_CACHE
    if _CELL_CACHE is not None:
        return _CELL_CACHE

    cell_file = os.path.join(SIGNATURE_DIR, "985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy")
    if not os.path.exists(cell_file):
        raise FileNotFoundError(f"Cell line multi-omics file {cell_file} not found.")

    raw_cells = np.load(cell_file, allow_pickle=True).item()
    cell_dict = {}

    # Extract pooled multi-omics profile for each cell line
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(4079, size=dim_reduce, replace=False)

    for cl_id, graph_nodes in raw_cells.items():
        # graph_nodes shape is (4079, 6)
        if isinstance(graph_nodes, np.ndarray) and graph_nodes.ndim == 2:
            # Subsample landmark genes and average across omic modalities
            sub_nodes = graph_nodes[sample_indices]
            cell_vec = np.mean(sub_nodes, axis=1).astype(np.float32)
        else:
            # fallback
            arr = np.array(graph_nodes)
            cell_vec = np.mean(arr.reshape(-1, 6), axis=1)[:dim_reduce].astype(np.float32)
        cell_dict[cl_id] = cell_vec

    _CELL_CACHE = cell_dict
    return cell_dict


def load_benchmark_dataset(dataset_name: str = "OncologyScreen") -> pd.DataFrame:
    """
    Loads raw benchmark synergy CSV:
    Options: 'OncologyScreen', 'Oneil', 'DrugCombDB', 'DrugComb', 'Almanac'
    """
    filename_map = {
        "oncologyscreen": "OncologyScreenLINCS_PRISM.csv",
        "oneil": "OneilLINCS_PRISM.csv",
        "drugcombdb": "DrugCombDBLINCS_PRISM.csv",
        "drugcomb": "DrugComb_LINCS_PRISM.csv",
        "almanac": "AlmanacLINCS_PRISM.csv"
    }

    key = dataset_name.lower().replace(" ", "").replace("_", "")
    target_file = filename_map.get(key, "OncologyScreenLINCS_PRISM.csv")
    csv_path = os.path.join(BENCHMARK_DIR, target_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark file {csv_path} not found.")

    df = pd.read_csv(csv_path)
    df.rename(columns={
        "drugname1": "drug_a",
        "drugname2": "drug_b",
        "cellline": "cell_line",
        "score": "synergy_score"
    }, inplace=True)
    return df


def prepare_real_feature_matrix(
    dataset_name: str = "OncologyScreen",
    feature_mode: str = "DRS",
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Builds symmetric feature matrix X and target array y from real benchmark data.
    X = [v_A + v_B,  v_A * v_B,  |v_A - v_B|,  cell_vec]
    """
    df = load_benchmark_dataset(dataset_name)
    drug_sigs = load_drug_signatures(feature_mode)
    cell_sigs = load_cell_line_features()

    # Filter to pairs where both drugs exist in the signatures
    valid_mask = df["drug_a"].isin(drug_sigs) & df["drug_b"].isin(drug_sigs)
    filtered_df = df[valid_mask].copy().reset_index(drop=True)

    if max_samples and len(filtered_df) > max_samples:
        filtered_df = filtered_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    X_list = []
    y_list = []
    valid_rows = []

    # Get sample cell vector for fallback
    sample_cl_vec = list(cell_sigs.values())[0]

    for idx, row in filtered_df.iterrows():
        vA = drug_sigs[row["drug_a"]]
        vB = drug_sigs[row["drug_b"]]
        cl_vec = cell_sigs.get(row["cell_line"], sample_cl_vec)

        # Symmetric fusion
        add_feat = vA + vB
        mul_feat = vA * vB
        diff_feat = np.abs(vA - vB)
        fused = np.concatenate([add_feat, mul_feat, diff_feat, cl_vec])

        X_list.append(fused)
        y_list.append(row["synergy_score"])
        valid_rows.append(row)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(valid_rows)

    return X, y, meta_df
