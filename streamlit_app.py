# === STREAMLIT APP FOR IRAK4 SCREENING ===

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import AllChem
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import os

custom_css = {
    ".ag-root-wrapper": {"border-radius": "12px"},
    ".ag-header": {"font-size": "16px", "background-color": "#F7D1BA"},
    ".ag-cell": {"font-size": "14px", "padding": "4px"},
    ".ag-row-hover": {"background-color": "#FFE2E2"},
}  

st.title('IRAK4 SCREENING')
st.info('This application is designed to predict potent IRAK4 inhibitors')

# === 1. UPLOAD ===
st.header("Step 1: Upload and extract ID & SMILES")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
id_col = st.text_input("ID column (optional)", value="", placeholder="e.g. Molecule_Name")
smiles_col = st.text_input("SMILES column (required)", value="", placeholder="e.g. SMILES")

if st.button("Create Dataset", type="primary"):
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
            
            # AgGrid hiển thị df_new
            gb = GridOptionsBuilder.from_dataframe(df_new)
            gb.configure_default_column(filterable=True, sortable=True)
            grid_options = gb.build()
            AgGrid(df_new, gridOptions=grid_options, height=300, theme="alpine", custom_css=custom_css)

# === 2. STANDARDIZATION ===
st.header("Step 2: Standardize")

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

if st.button("Standardize", type="primary"):
    if "df_new" in st.session_state:
        df = st.session_state.df_new.copy()
        df["standardized"] = standardize_smiles(df.SMILES)
        st.session_state.df_standardized = df
        st.success("✅ Step 2 completed.")
        
        # Hiển thị bằng AgGrid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filterable=True, sortable=True)
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, height=300, theme="alpine")
    else:
        st.warning("Please complete Step 1 first.")

# === 3. PAINS FILTER ===
st.header("Step 3: PAINS Filter")

if st.button("Run", type="primary"):
    if "df_standardized" in st.session_state:
        df = st.session_state.df_standardized.copy()
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
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

        # Hiển thị bảng raw_pains bằng AgGrid
        gb = GridOptionsBuilder.from_dataframe(raw_pains)
        gb.configure_default_column(filterable=True, sortable=True)
        grid_options = gb.build()
        AgGrid(raw_pains, gridOptions=grid_options, height=300, theme="alpine")
    else:
        st.warning("Please complete Step 2 first.")

# === 4. ECFP4 FINGERPRINTS ===
st.header("Step 4: Compute Fingerprints")

if st.button("Generate",type="primary"):
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
    else:
        st.warning("Please complete Step 3 first.")

# === 5. QSAR SCREENING ===
st.header("Step 5: IRAK4 QSAR Screening")

def run_qsar_prediction():
    data = st.session_state.df_split.copy()

    # === Binary Classification ===
    with open('model1/rf_binary_813_tuned.pkl', 'rb') as f:
        clf = pickle.load(f)
    X_bin = data.drop(['ID', 'standardized'], axis=1)
    prob_bin = clf.predict_proba(X_bin)[:, 1]

    bin_df = pd.DataFrame({
        'ID': data['ID'],
        'standardized': data['standardized'],
        'label_prob': prob_bin,
        'label': (prob_bin >= 0.5).astype(int)
    })
    bin_df['label_prob'] = bin_df['label_prob'].round(4)

    # === Regression Prediction ===
    with open('model1/xgb_regression_764_tuned.pkl', 'rb') as f:
        xgb = pickle.load(f)
    pred_reg = xgb.predict(X_bin)

    reg_df = pd.DataFrame({
        'ID': data['ID'],
        'standardized': data['standardized'],
        'predicted_pIC50': pred_reg,
        'label': (pred_reg >= -np.log10(8e-9)).astype(int)
    })
    reg_df['predicted_pIC50'] = reg_df['predicted_pIC50'].round(4)

    # === Consensus Actives ===
    consensus_df = bin_df[bin_df.label == 1].merge(
        reg_df[reg_df.label == 1], on=['ID', 'standardized']
    )[['ID', 'standardized', 'label_prob', 'predicted_pIC50']]

    # Lưu vào session_state
    st.session_state.result = bin_df
    st.session_state.result_reg = reg_df
    st.session_state.consensus = consensus_df
    st.session_state.qsar_done = True

# Nút chạy
if st.button("Run Prediction",type="primary"):
    if "df_split" not in st.session_state:
        st.warning("⚠️ Please complete Step 4 (ECFP4 Fingerprints) first.")
    else:
        try:
            run_qsar_prediction()
            st.success("✅ Step 5 completed.")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# Hiển thị kết quả nếu có
if st.session_state.get("qsar_done", False):
    # === Binary ===
    st.subheader("🧪 Binary Predicted Actives")
    df_binary_active = st.session_state.result.copy()
    df_binary_active = df_binary_active[df_binary_active['label'] == 1][['ID', 'standardized', 'label_prob']]
    gb_bin = GridOptionsBuilder.from_dataframe(df_binary_active)
    gb_bin.configure_default_column(filterable=True, sortable=True)
    gb_bin.configure_column("label_prob", type=["numericColumn"], valueFormatter="x.toFixed(4)")
    grid_options_bin = gb_bin.build()
    AgGrid(df_binary_active, gridOptions=grid_options_bin, height=300, theme='alpine')

    # === Regression ===
    st.subheader("📈 Regression Predicted Actives")
    df_reg_active = st.session_state.result_reg.copy()
    df_reg_active = df_reg_active[df_reg_active['label'] == 1][['ID', 'standardized', 'predicted_pIC50']]
    gb_reg = GridOptionsBuilder.from_dataframe(df_reg_active)
    gb_reg.configure_default_column(filterable=True, sortable=True)
    gb_reg.configure_column("predicted_pIC50", type=["numericColumn"], valueFormatter="x.toFixed(4)")
    grid_options_reg = gb_reg.build()
    AgGrid(df_reg_active, gridOptions=grid_options_reg, height=300, theme='alpine')

    # === Consensus ===
    st.subheader("📊 Consensus Actives")
    consensus_df = st.session_state.consensus.copy()
    gb_consensus = GridOptionsBuilder.from_dataframe(consensus_df)
    gb_consensus.configure_default_column(filterable=True, sortable=True)
    gb_consensus.configure_column("label_prob", type=["numericColumn"], valueFormatter="x.toFixed(4)")
    gb_consensus.configure_column("predicted_pIC50", type=["numericColumn"], valueFormatter="x.toFixed(4)")
    grid_options_consensus = gb_consensus.build()
    AgGrid(consensus_df, gridOptions=grid_options_consensus, height=400, theme='alpine')

    # Download CSV
    csv = consensus_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Result",
        data=csv,
        file_name='screening_result.csv',
        mime='text/csv'
    )
