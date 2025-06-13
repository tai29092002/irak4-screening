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
st.header("Step 1: INPUT DATA")

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
        flexible_callout(
            message="🎯 Step 4 completed.",
            **CALLOUT_CONFIG  # <-- unpack dict
        )
    else:
        flexible_callout(
            message="Please complete Step 3 first.",
            **CALLOUT_CONFIG  # <-- unpack dict
        )

# Step 5: IRAK4 QSAR Screening
st.header("Step 5: IRAK4 QSAR Screening")

def run_qsar_prediction():
    data = st.session_state.df_split.copy()
    X = data.drop(['ID', 'standardized'], axis=1)

    # === Binary Classification ===
    with open('model1/rf_binary_813_tuned.pkl', 'rb') as f:
        clf = pickle.load(f)
    prob = clf.predict_proba(X)[:, 1]
    bin_df = pd.DataFrame({
        'ID': data['ID'],
        'standardized': data['standardized'],
        'label_prob': np.round(prob, 4)
    })
    bin_df['active'] = np.where(bin_df['label_prob'] > 0.5, 'Strong', 'Weak')

    # === Regression Prediction ===
    with open('model1/xgb_regression_764_tuned.pkl', 'rb') as f:
        xgb = pickle.load(f)
    pIC50 = np.round(xgb.predict(X), 4)
    ic50_nm = np.round((10 ** (-pIC50) * 1e9), 2)
    reg_df = pd.DataFrame({
        'ID': data['ID'],
        'standardized': data['standardized'],
        'IC50 (nM)': ic50_nm
    })
    reg_df['active'] = np.where(reg_df['IC50 (nM)'] <= 8, 'Strong', 'Weak')

    # === Consensus Actives ===
    consensus_df = (
        bin_df[bin_df['active'] == 'Strong'][['ID', 'standardized', 'label_prob', 'active']]
        .merge(
            reg_df[reg_df['active'] == 'Strong'][['ID', 'standardized', 'IC50 (nM)']],
            on=['ID', 'standardized']
        )
    )

    # Save to session state
    st.session_state.result     = bin_df
    st.session_state.result_reg = reg_df
    st.session_state.consensus  = consensus_df
    st.session_state.qsar_done  = True

# Always show the Run Prediction button
if st.button('Run Prediction', key='run_qsar', type='primary'):
    if 'df_split' not in st.session_state:
        flexible_callout(
            message='Please complete Step 4 first.',
            **CALLOUT_CONFIG
        )
    else:
        try:
            run_qsar_prediction()
            flexible_callout(
                message='🎯 Step 5 completed.',
                **CALLOUT_CONFIG
            )
        except Exception as e:
            flexible_callout(
                message=f'❌ Prediction error: {e}',
                **CALLOUT_CONFIG
            )

# Optionally show success message
if st.session_state.get('qsar_done', False):
    st.success("✅ Step 5 has been run — results updated below.")

# Display results
if st.session_state.get('qsar_done', False):
    # Binary Predicted Actives
    st.subheader('🧪 Binary Predicted Actives (All Compounds)')
    df_bin = st.session_state.result[['ID', 'standardized', 'active', 'label_prob']]
    gb1 = GridOptionsBuilder.from_dataframe(df_bin)
    gb1.configure_default_column(filterable=True, sortable=True)
    gb1.configure_column('label_prob', type=['numericColumn'], valueFormatter='x.toFixed(4)')
    AgGrid(df_bin, gridOptions=gb1.build(), height=300, theme='alpine', custom_css=custom_css)

    # Regression Predicted Actives
    st.subheader('📈 Regression Predicted Actives (All Compounds)')
    df_reg = st.session_state.result_reg[['ID', 'standardized', 'active', 'IC50 (nM)']]
    gb2 = GridOptionsBuilder.from_dataframe(df_reg)
    gb2.configure_default_column(filterable=True, sortable=True)
    gb2.configure_column('IC50 (nM)', type=['numericColumn'], valueFormatter='x.toFixed(2)')
    AgGrid(df_reg, gridOptions=gb2.build(), height=300, theme='alpine', custom_css=custom_css)

    # Consensus Actives
    st.subheader('📊 Consensus Actives')
    df_cons = st.session_state.consensus[['ID', 'standardized', 'label_prob', 'IC50 (nM)', 'active']]
    gb3 = GridOptionsBuilder.from_dataframe(df_cons)
    gb3.configure_default_column(filterable=True, sortable=True)
    gb3.configure_column('label_prob', type=['numericColumn'], valueFormatter='x.toFixed(4)')
    gb3.configure_column('IC50 (nM)', type=['numericColumn'], valueFormatter='x.toFixed(2)')
    AgGrid(df_cons, gridOptions=gb3.build(), height=400, theme='alpine', custom_css=custom_css)
    
    # Download CSV
    csv = consensus_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Result",
        data=csv,
        file_name='screening_result.csv',
        mime='text/csv'
    )
