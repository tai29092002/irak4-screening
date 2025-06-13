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
from st_flexible_callout_elements import flexible_callout

custom_css = {
    ".ag-root-wrapper": {"border-radius": "8px"},
    ".ag-header": {"font-size": "17px", "background-color": "#D29F80"},
    ".ag-cell": {"font-size": "15px", "padding": "4px"},
    ".ag-row-hover": {"background-color": "#FCE7C8"},
}  
CALLOUT_CONFIG = {
    "background_color": "#FFDAB3",
    "font_color": "#003366",
    "font_size": 17,
    "border_radius": 8,
    "padding": 10,
    "margin_bottom": 10
}

st.title('IRAK4 SCREENING')
flexible_callout(
    message="This application is designed to predict potent IRAK4 inhibitors",
    **CALLOUT_CONFIG  # <-- unpack dict
)
# === 1. UPLOAD ===
st.header("Step 1: Input")

# Inputs
uploaded_file = st.file_uploader("Upload a CSV file (optional)", type=['csv'])
id_col = st.text_input("ID column (optional)", value="", placeholder="e.g. Molecule_Name")
smiles_col = st.text_input("SMILES column (required if CSV)", value="", placeholder="e.g. SMILES")
st.markdown("**Or manually input SMILES below:** one per line or with optional ID prefix separated by comma")
manual_smiles_input = st.text_area("Manual SMILES input", height=150, placeholder="CCO\nmol1,CCN\nmol2,CCC")

# Standardization helper
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
                result.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            else:
                result.append(None)
        except Exception:
            result.append(None)
    return result

# Single button for both steps
if st.button("Process", type="primary"):
    # Step 1: build df_new
    data = []
    if manual_smiles_input.strip():
        for line in manual_smiles_input.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 2:
                data.append({'ID': parts[0], 'SMILES': parts[1]})
            elif len(parts) == 1:
                data.append({'ID': f"molecule{len(data)+1}", 'SMILES': parts[0]})
            else:
                st.warning(f"Line skipped: {line}")
    elif uploaded_file is not None and smiles_col:
        df = pd.read_csv(uploaded_file)
        if smiles_col not in df.columns:
            st.error(f"Column '{smiles_col}' not found in CSV.")
        else:
            ids = df[id_col] if id_col and id_col in df.columns else [f"molecule{i+1}" for i in range(len(df))]
            for i, smi in enumerate(df[smiles_col]):
                data.append({'ID': ids[i], 'SMILES': smi})
    else:
        st.error("Please upload a CSV or input SMILES manually.")

    # If dataset created, standardize
    if data:
        df_new = pd.DataFrame(data)
        df_new['standardized'] = standardize_smiles(df_new['SMILES'])
        st.session_state.df_standardized = df_new
        flexible_callout(message="🎯 Steps 1 completed.", **CALLOUT_CONFIG)
        # Display
        gb = GridOptionsBuilder.from_dataframe(df_new)
        gb.configure_default_column(filterable=True, sortable=True)
        AgGrid(df_new, gridOptions=gb.build(), height=350, theme='alpine', custom_css=custom_css)

# === 3. PAINS FILTER ===
st.header("Step 2: PAINS Filter")

if st.button("Process", type="primary"):
    if "df_standardized" in st.session_state:
        df = st.session_state.df_new.copy()
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
        flexible_callout(
            message="🎯 Step 3 completed.",
            **CALLOUT_CONFIG  # <-- unpack dict
        )

        # Hiển thị bảng raw_pains bằng AgGrid
        gb = GridOptionsBuilder.from_dataframe(raw_pains)
        gb.configure_default_column(filterable=True, sortable=True)
        grid_options = gb.build()
        AgGrid(raw_pains, gridOptions=grid_options, height=300, theme="alpine",custom_css=custom_css)
    else:
        flexible_callout(
            message="Please complete Step 2 first.",
            **CALLOUT_CONFIG  # <-- unpack dict
        )

# === Step 4+5: Fingerprints & QSAR Screening ===
st.header("Step 3: Screening")

def compute_fp_and_qsar():
    # 1) Compute ECFP4 fingerprints
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
    df_split = pd.concat(
        [
            df[['ID', 'standardized']].reset_index(drop=True),
            df_fp.reset_index(drop=True)
        ], axis=1
    )
    st.session_state.df_split = df_split

    # 2) Binary Classification
    X = df_split.drop(['ID', 'standardized'], axis=1)
    with open('model1/rf_binary_813_tuned.pkl', 'rb') as f:
        clf = pickle.load(f)
    prob = clf.predict_proba(X)[:, 1]
    bin_df = pd.DataFrame({
        'ID': df_split['ID'],
        'standardized': df_split['standardized'],
        'label_prob': np.round(prob, 4)
    })
    bin_df['active'] = np.where(bin_df['label_prob'] > 0.5, 'Strong', 'Weak')
    st.session_state.result = bin_df

    # 3) Regression Prediction → IC50 (nM)
    with open('model1/xgb_regression_764_tuned.pkl', 'rb') as f:
        xgb = pickle.load(f)
    pIC50 = np.round(xgb.predict(X), 4)
    ic50 = np.round((10 ** (-pIC50) * 1e9), 2)
    reg_df = pd.DataFrame({
        'ID': df_split['ID'],
        'standardized': df_split['standardized'],
        'IC50 (nM)': ic50
    })
    reg_df['active'] = np.where(reg_df['IC50 (nM)'] <= 8, 'Strong', 'Weak')
    st.session_state.result_reg = reg_df

    # 4) Consensus: both Strong
    consensus_df = (
        bin_df.loc[bin_df['active']=='Strong', ['ID','standardized','label_prob','active']]
        .merge(
            reg_df.loc[reg_df['active']=='Strong', ['ID','standardized','IC50 (nM)']],
            on=['ID','standardized']
        )
    )
    st.session_state.consensus = consensus_df
    st.session_state.qsar_done  = True

# single button for both steps
if st.button("Generate & Predict", type="primary"):
    if 'df_select' not in st.session_state:
        flexible_callout(
            message="Please complete Step 3 first.",
            **CALLOUT_CONFIG
        )
    else:
        try:
            compute_fp_and_qsar()
            flexible_callout(
                message="🎯 Steps 4+5 completed.",
                **CALLOUT_CONFIG
            )
        except Exception as e:
            flexible_callout(
                message=f"❌ Processing error: {e}",
                **CALLOUT_CONFIG
            )

# show results if done
if st.session_state.get('qsar_done', False):
    st.success("✅ Fingerprints & QSAR done — see results below.")

    # Binary
    st.subheader("🧪 Binary Predicted Actives (All)")
    dfb = st.session_state.result[['ID','standardized','active','label_prob']].reset_index(drop=True)
    gb = GridOptionsBuilder.from_dataframe(dfb)
    gb.configure_default_column(filterable=True, sortable=True)
    gb.configure_column('label_prob', type=['numericColumn'], valueFormatter='x.toFixed(4)')
    AgGrid(dfb, gridOptions=gb.build(), height=250, theme='alpine', custom_css=custom_css)

    # Regression
    st.subheader("📈 Regression Predicted Actives (All)")
    dfr = st.session_state.result_reg[['ID','standardized','active','IC50 (nM)']].reset_index(drop=True)
    gb = GridOptionsBuilder.from_dataframe(dfr)
    gb.configure_default_column(filterable=True, sortable=True)
    gb.configure_column('IC50 (nM)', type=['numericColumn'], valueFormatter='x.toFixed(2)')
    AgGrid(dfr, gridOptions=gb.build(), height=250, theme='alpine', custom_css=custom_css)

    # Consensus
    st.subheader("📊 Consensus Actives")
    dfc = st.session_state.consensus[['ID','standardized','label_prob','IC50 (nM)','active']].reset_index(drop=True)
    gb = GridOptionsBuilder.from_dataframe(dfc)
    gb.configure_default_column(filterable=True, sortable=True)
    gb.configure_column('label_prob', type=['numericColumn'], valueFormatter='x.toFixed(4)')
    gb.configure_column('IC50 (nM)', type=['numericColumn'], valueFormatter='x.toFixed(2)')
    AgGrid(dfc, gridOptions=gb.build(), height=350, theme='alpine', custom_css=custom_css)
    
    # Download CSV
    csv = consensus_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Result",
        data=csv,
        file_name='screening_result.csv',
        mime='text/csv'
    )
