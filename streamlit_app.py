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

st.title('IRAK4 QSAR SCREENING')
flexible_callout(
    message="This application is designed to predict potent IRAK4 inhibitors",
    **CALLOUT_CONFIG  # <-- unpack dict
)

# === 1. UPLOAD & STANDARDIZE ===
st.header("Step 1: Input Data")

st.markdown("**Upload a CSV file:**")
uploaded_file      = st.file_uploader("Upload here", type=['csv'])
id_col             = st.text_input("ID column (optional)", value="", placeholder="e.g. Molecule_Name")
smiles_col         = st.text_input("SMILES column (required)", value="", placeholder="e.g. SMILES")
st.markdown("**Or manually input SMILES below:** one per line and with ID (optional) separated by comma")
manual_smiles_input = st.text_area("Manual SMILES input", height=150, placeholder="CCO\nmol1,CCN\nmol2,")

def standardize_smiles(batch):
    uc        = rdMolStandardize.Uncharger()
    md        = rdMolStandardize.MetalDisconnector()
    te        = rdMolStandardize.TautomerEnumerator()
    reionizer = rdMolStandardize.Reionizer()
    result    = []
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
        except:
            result.append(None)
    return result

if st.button("Process", key="process_step1", type="primary"):
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
        df0 = pd.read_csv(uploaded_file)
        if smiles_col not in df0.columns:
            st.error(f"Column '{smiles_col}' not found in CSV.")
        else:
            ids = df0[id_col] if id_col and id_col in df0.columns else [f"molecule{i+1}" for i in range(len(df0))]
            for i, smi in enumerate(df0[smiles_col]):
                data.append({'ID': ids[i], 'SMILES': smi})
    else:
        st.error("Please upload a CSV or input SMILES manually.")

    if data:
        df_new = pd.DataFrame(data)
        df_new['standardized'] = standardize_smiles(df_new['SMILES'])
        st.session_state.df_standardized = df_new
        flexible_callout(message="🎯 Step 1 completed.", **CALLOUT_CONFIG)

# **Always** show the standardized table if it exists**
if 'df_standardized' in st.session_state:
    st.subheader("Input Data")
    df_std = st.session_state.df_standardized.reset_index(drop=True)
    gb = GridOptionsBuilder.from_dataframe(df_std)
    gb.configure_default_column(filterable=True, sortable=True)
    AgGrid(
        df_std,
        gridOptions=gb.build(),
        height=250,
        theme='alpine',
        custom_css=custom_css
    )

# Step 2: PAINS Filter
st.header("Step 2: PAINS Filter")
def run_pains_filter():
    df = st.session_state.df_standardized.copy()
    total = len(df)
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    clean = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['standardized'])
        if mol and not catalog.GetFirstMatch(mol):
            clean.append(row)
    raw_pains = pd.DataFrame(clean).reset_index(drop=True)
    st.session_state.df_select = raw_pains

    passed = len(raw_pains)
    failed = total - passed
    if failed == 0:
        st.success(f"No PAINS found. All compounds passed the filter.")
    else:
        st.success(f"{passed} compounds passed (no PAINS), {failed} flagged by PAINS.")

if st.button("Screening", key="process_step2", type="primary"):
    if "df_standardized" not in st.session_state:
        flexible_callout(
            message="Please complete Step 1 first.",
            **CALLOUT_CONFIG
        )
    else:
        try:
            run_pains_filter()
        except Exception as e:
            flexible_callout(
                message=f"❌ PAINS filter error: {e}",
                **CALLOUT_CONFIG
            )

if "df_select" in st.session_state:
    st.subheader("PAINS Filter Results")
    raw_pains = st.session_state.df_select
    gb = GridOptionsBuilder.from_dataframe(raw_pains)
    gb.configure_default_column(filterable=True, sortable=True)
    gb.configure_grid_options(suppressRowNumbers=True)
    AgGrid(
        raw_pains,
        gridOptions=gb.build(),
        height=250,
        theme="alpine",
        custom_css=custom_css
    )
        
# === Step 3: Fingerprints & QSAR Prediction ===
st.header("Step 3: QSAR Prediction")

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
        [df[['ID', 'standardized']].reset_index(drop=True), df_fp.reset_index(drop=True)],
        axis=1
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
    bin_df['Active'] = np.where(bin_df['label_prob'] > 0.5, 'Strong', 'Weak')
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
    reg_df['Active'] = np.where(reg_df['IC50 (nM)'] <= 8, 'Strong', 'Weak')
    st.session_state.result_reg = reg_df

    # 4) Consensus: both Strong
    consensus_df = (
        bin_df.loc[bin_df['Active']=='Strong', ['ID','standardized','Active']]
        .merge(
            reg_df.loc[reg_df['Active']=='Strong', ['ID','standardized','IC50 (nM)']],
            on=['ID','standardized']
        )
    )
    st.session_state.consensus = consensus_df
    st.session_state.qsar_done  = True

# single button for both steps
if st.button("Screening", key="run_fp_qsar", type="primary"):
    if 'df_select' not in st.session_state:
        flexible_callout(
            message="Please complete Step 3 first.",
            **CALLOUT_CONFIG
        )
    else:
        try:
            compute_fp_and_qsar()
            flexible_callout(
                message="🎯 Steps 3 completed.",
                **CALLOUT_CONFIG
            )
        except Exception as e:
            flexible_callout(
                message=f"❌ Processing error: {e}",
                **CALLOUT_CONFIG
            )

# show results and download
if st.session_state.get('qsar_done', False):
    st.success("✅ Screening done — see results below.")

    # --- Binary Prediction ---
    st.subheader("🧪 Binary Prediction")
    dfb = (
        st.session_state.result
        [['ID', 'standardized', 'Active', 'label_prob']]
        .reset_index(drop=True)
    )
    gb_bin = GridOptionsBuilder.from_dataframe(dfb)
    gb_bin.configure_default_column(filterable=True, sortable=True)
    gb_bin.configure_column(
        'label_prob',
        type=['numericColumn'],
        valueFormatter='x.toFixed(4)'
    )
    gb_bin.configure_grid_options(suppressRowNumbers=True)
    AgGrid(
        dfb,
        gridOptions=gb_bin.build(),
        height=350,
        fit_columns_on_grid_load=True,
        theme='alpine',
        custom_css=custom_css
    )

    # --- Regression Prediction ---
    st.subheader("📈 Regression Prediction")
    dfr = (
        st.session_state.result_reg
        [['ID', 'standardized', 'Active', 'IC50 (nM)']]
        .reset_index(drop=True)
    )
    gb_reg = GridOptionsBuilder.from_dataframe(dfr)
    gb_reg.configure_default_column(filterable=True, sortable=True)
    gb_reg.configure_column(
        'IC50 (nM)',
        type=['numericColumn'],
        valueFormatter='x.toFixed(2)'
    )
    gb_reg.configure_grid_options(suppressRowNumbers=True)
    AgGrid(
        dfr,
        gridOptions=gb_reg.build(),
        height=250,
        fit_columns_on_grid_load=True,
        theme='alpine',
        custom_css=custom_css
    )

    # --- Consensus Actives ---
    st.subheader("📊 Consensus Actives")
    dfc = (
        st.session_state.consensus
        [['ID', 'standardized', 'Active', 'IC50 (nM)']]
        .reset_index(drop=True)
    )
    gb_cons = GridOptionsBuilder.from_dataframe(dfc)
    gb_cons.configure_default_column(filterable=True, sortable=True)
    gb_cons.configure_column(
        'IC50 (nM)',
        type=['numericColumn'],
        valueFormatter='x.toFixed(2)'
    )
    gb_cons.configure_grid_options(suppressRowNumbers=True)
    AgGrid(
        dfc,
        gridOptions=gb_cons.build(),
        height=250,
        fit_columns_on_grid_load=True,
        theme='alpine',
        custom_css=custom_css
    )

    # Download results CSV (even if empty)
    df_download = st.session_state.get('consensus', pd.DataFrame(columns=['ID','standardized','IC50 (nM)','active']))
    csv = df_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Result",
        data=csv,
        file_name='IRAK4_screening_result.csv',
        mime='text/csv'
    )

