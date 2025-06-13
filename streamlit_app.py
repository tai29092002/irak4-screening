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
st.header("Step 1: Upload and extract ID & SMILES")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
id_col = st.text_input("ID column (optional)", value="", placeholder="e.g. Molecule_Name")
smiles_col = st.text_input("SMILES column (required)", value="", placeholder="e.g. SMILES")

st.markdown("**Or manually input SMILES below (one per line, optional IDs separated by comma):**")
manual_smiles_input = st.text_area("Manual SMILES input", height=150, placeholder="CCO\nmolecule1,CCN\nmolecule2,CCC")

if st.button("Create Dataset", type="primary"):
    if manual_smiles_input.strip():  # Nếu có nhập tay thì ưu tiên xử lý
        data = []
        for line in manual_smiles_input.strip().splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 2:
                data.append({'ID': parts[0], 'SMILES': parts[1]})
            elif len(parts) == 1:
                data.append({'ID': f"molecule{len(data)+1}", 'SMILES': parts[0]})
            else:
                st.warning(f"Line skipped due to format: {line}")
        df_new = pd.DataFrame(data)
        st.session_state.df_new = df_new
        flexible_callout(message="🎯 Step 1 completed (manual input).", **CALLOUT_CONFIG)
    elif uploaded_file is None:
        flexible_callout(message="Please upload file or input SMILES manually.", **CALLOUT_CONFIG)
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
            flexible_callout(message="🎯 Step 1 completed (from file).", **CALLOUT_CONFIG)

    if "df_new" in st.session_state:
        gb = GridOptionsBuilder.from_dataframe(st.session_state.df_new)
        gb.configure_default_column(filterable=True, sortable=True)
        grid_options = gb.build()
        AgGrid(st.session_state.df_new, gridOptions=grid_options, height=300, theme="alpine", custom_css=custom_css)


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
        flexible_callout(
            message="🎯 Step 2 completed.",
            **CALLOUT_CONFIG  # <-- unpack dict
        )
        # Hiển thị bằng AgGrid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filterable=True, sortable=True)
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, height=300, theme="alpine",custom_css=custom_css)
    else:
        flexible_callout(
            message="Please complete Step 1 first.",
            **CALLOUT_CONFIG  # <-- unpack dict
        )

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
# Step 5: IRAK4 QSAR Screening
st.header("Step 5: IRAK4 QSAR Screening")

def run_qsar_prediction():
    data = st.session_state.df_split.copy()

    # Binary Classification
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
    # Add active column: Strong if >0.5, else Weak
    bin_df['active'] = np.where(bin_df['label_prob'] > 0.5, 'Strong', 'Weak')

    # Regression Prediction
    with open('model1/xgb_regression_764_tuned.pkl', 'rb') as f:
        xgb = pickle.load(f)
    pred_pIC50 = xgb.predict(X_bin)

    reg_df = pd.DataFrame({
        'ID': data['ID'],
        'standardized': data['standardized'],
        'predicted_pIC50': np.round(pred_pIC50, 4)
    })
    # Compute IC50 (M) then convert to nM and round
    reg_df['IC50 (nM)'] = (10 ** (-reg_df['predicted_pIC50']) * 1e9).round(2)
    # Add active based on pIC50 threshold for 8 nM: pIC50 >= -log10(8e-9)
    threshold = -np.log10(8e-9)
    reg_df['active'] = np.where(reg_df['predicted_pIC50'] >= threshold, 'Strong', 'Weak')

    # Consensus Actives: compounds strong in both binary and regression
    consensus_df = (
        bin_df.loc[bin_df['active'] == 'Strong', ['ID', 'standardized', 'label_prob', 'active']]
        .merge(
            reg_df.loc[reg_df['active'] == 'Strong', ['ID', 'standardized', 'IC50 (nM)']],
            on=['ID', 'standardized']
        )
    )

    # Save results
    st.session_state.result     = bin_df
    st.session_state.result_reg = reg_df[['ID','standardized','active','IC50 (nM)']]
    st.session_state.consensus  = consensus_df
    st.session_state.qsar_done  = True

# Run Prediction button (show once)
if not st.session_state.get('qsar_done', False):
    if st.button('Run Prediction', key='run_prediction_btn', type='primary'):
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
else:
    st.success("✅ You've already run Step 5 — no need to click again.")

# Display results
if st.session_state.get('qsar_done', False):
    # Binary Predicted Actives (All Compounds)
    st.subheader('🧪 Binary Predicted Actives (All Compounds)')
    df_binary = st.session_state.result[['ID','standardized','active','label_prob']]
    gb_bin = GridOptionsBuilder.from_dataframe(df_binary)
    gb_bin.configure_default_column(filterable=True, sortable=True)
    gb_bin.configure_column('label_prob', type=['numericColumn'], valueFormatter='x.toFixed(4)')
    AgGrid(df_binary, gridOptions=gb_bin.build(), height=300, theme='alpine', custom_css=custom_css)

    # Regression Predicted Actives (All Compounds)
    st.subheader('📈 Regression Predicted Actives (All Compounds)')
    df_reg = st.session_state.result_reg.copy()
    gb_reg = GridOptionsBuilder.from_dataframe(df_reg)
    gb_reg.configure_default_column(filterable=True, sortable=True)
    gb_reg.configure_column('IC50 (nM)', type=['numericColumn'], valueFormatter='x.toFixed(2)')
    AgGrid(df_reg, gridOptions=gb_reg.build(), height=300, theme='alpine', custom_css=custom_css)

    # Consensus Actives
    st.subheader('📊 Consensus Actives')
    df_cons = st.session_state.consensus[['ID','standardized','label_prob','IC50 (nM)','active']]
    gb_cons = GridOptionsBuilder.from_dataframe(df_cons)
    gb_cons.configure_default_column(filterable=True, sortable=True)
    gb_cons.configure_column('label_prob', type=['numericColumn'], valueFormatter='x.toFixed(4)')
    gb_cons.configure_column('IC50 (nM)', type=['numericColumn'], valueFormatter='x.toFixed(2)')
    AgGrid(df_cons, gridOptions=gb_cons.build(), height=400, theme='alpine', custom_css=custom_css)
    
    # Download CSV
    csv = consensus_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Result",
        data=csv,
        file_name='screening_result.csv',
        mime='text/csv'
    )
