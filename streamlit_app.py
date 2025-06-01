# === STREAMLIT APP FOR IRAK4 SCREENING ===

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm
from rdkit.Chem import FilterCatalog
from rdkit.Chem import AllChem
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import os

st.title('IRAK4 SCREENING')
st.info('This application is designed to predict potent IRAK4 inhibitors')

# === 1. UPLOAD ===
st.header("Step 1: Upload and extract ID & SMILES")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
id_col = st.text_input("ID column (optional)", value="", placeholder="e.g. Molecule_Name")
smiles_col = st.text_input("SMILES column (required)", value="", placeholder="e.g. SMILES")

if st.button("Create Dataset"):
    if uploaded_file is None:
        st.warning("Please upload a file.")
    elif not smiles_col.strip():
        st.warning("Please enter SMILES column name.")
    else:
        df = pd.read_csv(uploaded_file)
        if smiles_col not in df.columns:
            st.error(f"Column '{smiles_col}' not found.")
        else:
            ids = df[id_col] if id_col and id_col in df.columns else [f"molecule{i+1}" for i in range(len(df))]
            df_new = pd.DataFrame({'ID': ids, 'SMILES': df[smiles_col]})
            st.session_state.df_new = df_new
            st.success("✅ Step 1 completed.")
            st.dataframe(df_new)

# === 2. STANDARDIZATION ===
st.header("Step 2: Standardize SMILES")

def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()
    reionizer = rdMolStandardize.Reionizer()
    result = []
    for smi in batch:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol)
                mol = rdMolStandardize.Cleanup(mol)
                mol = rdMolStandardize.Normalize(mol)
                mol = uc.uncharge(mol)
                mol = rdMolStandardize.FragmentParent(mol)
                mol = reionizer.reionize(mol)
                mol = md.Disconnect(mol)
                mol = te.Canonicalize(mol)
                result.append(Chem.MolToSmiles(mol))
            else:
                result.append(None)
        except:
            result.append(None)
    return result

if st.button("Standardize"):
    if "df_new" in st.session_state:
        df = st.session_state.df_new.copy()
        df["standardized"] = standardize_smiles(df.SMILES)
        st.session_state.df_standardized = df
        st.success("✅ Step 2 completed.")
        st.dataframe(df)
    else:
        st.warning("Please complete Step 1 first.")

# === 3. PAINS FILTER ===
st.header("Step 3: PAINS Filtering")

if st.button("Run PAINS Filter"):
    if "df_standardized" in st.session_state:
        df = st.session_state.df_standardized.copy()
        catalog = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams().AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS))
        clean, matches = [], []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['standardized'])
            if mol and catalog.GetFirstMatch(mol):
                matches.append(row)
            elif mol:
                clean.append(row)
        raw_pains = pd.DataFrame(clean)
        st.session_state.df_select = raw_pains.copy()
        st.success("✅ Step 3 completed.")
        st.dataframe(raw_pains)
    else:
        st.warning("Please complete Step 2 first.")

# === 4. ECFP4 FINGERPRINTS ===
st.header("Step 4: Compute ECFP4 Fingerprints")

if st.button("Generate ECFP4 Fingerprints"):
    if "df_select" in st.session_state:
        df = st.session_state.df_select.copy()

        def smiles_to_fp(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((2048,), dtype=int)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                return arr
            return np.full(2048, np.nan)

        fps = df['standardized'].apply(smiles_to_fp)
        df_fp = pd.DataFrame(fps.tolist(), columns=[f"bit_{i}" for i in range(2048)])
        df_out = pd.concat([df[['ID', 'standardized']].reset_index(drop=True), df_fp.reset_index(drop=True)], axis=1)
        st.session_state.df_split = df_out
        st.success("✅ Step 4 completed.")
        st.dataframe(df_out.head())
    else:
        st.warning("Please complete Step 3 first.")

# === 5. QSAR SCREENING ===
st.header("Step 5: IRAK4 QSAR Screening")

if "df_split" not in st.session_state:
    st.warning("⚠️ Please generate ECFP4 fingerprints in Step 4 first to unlock prediction.")
else:
    data = st.session_state.df_split.copy()

    if st.button("Run Prediction"):
        try:
            # (toàn bộ phần xử lý mô hình và hiển thị kết quả đặt trong đây)
            df = st.session_state.df_split.copy()

            with open('model1/rf_binary_813_tuned.pkl', 'rb') as f:
                clf = pickle.load(f)
            X_bin = df.drop(['ID', 'standardized'], axis=1)
            prob_bin = clf.predict_proba(X_bin)[:, 1]

            bin_df = pd.DataFrame({
                'ID': df['ID'],
                'standardized': df['standardized'],
                'label_prob': np.round(prob_bin, 4),
                'label': (prob_bin >= 0.5).astype(int)
            })

            with open('model1/xgb_regression_764_tuned.pkl', 'rb') as f:
                xgb = pickle.load(f)
            pred_reg = xgb.predict(X_bin)
            reg_df = pd.DataFrame({
                'ID': df['ID'],
                'standardized': df['standardized'],
                'predicted_pIC50': np.round(pred_reg, 4),
                'label': (pred_reg >= -np.log10(8e-9)).astype(int)
            })

            st.session_state.result = bin_df
            st.session_state.result_reg = reg_df

            consensus = bin_df[bin_df.label == 1].merge(reg_df[reg_df.label == 1], on=['ID', 'standardized'])
            consensus = consensus[['ID', 'standardized', 'label_prob', 'predicted_pIC50']]
            st.session_state.consensus = consensus
            st.success("✅ Step 5 completed.")

            st.subheader("Consensus Actives")
            AgGrid(consensus)
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please complete Step 4 first.")





