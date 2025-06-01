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
from rdkit.Chem import AllChem
import subprocess
import os
st.title('IRAK4 SCREENING')
st.info('This application is designed to predict potent IRAK4 inhibitors')

# === 1. Upload and extract ===
st.header("Step 1: Upload and extract ID & SMILES")
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=['csv', 'txt'])
id_col = st.text_input("ID column (optional)", value="", placeholder="e.g. Molecule_Name")
smiles_col = st.text_input("SMILES column (required)", value="", placeholder="e.g. SMILES")

if st.button("Create"):
    if uploaded_file is None:
        st.warning("Please upload a file.")
    elif not smiles_col.strip():
        st.warning("Please enter SMILES column name.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"📄 File uploaded: {uploaded_file.name}")

            if smiles_col not in df.columns:
                st.error(f"Column '{smiles_col}' not found. Available columns: {list(df.columns)}")
            else:
                smiles_series = df[smiles_col]
                if id_col and id_col in df.columns:
                    id_series = df[id_col]
                    st.info(f"Using column '{id_col}' for ID.")
                else:
                    id_series = [f"molecule{i+1}" for i in range(len(smiles_series))]
                    st.info("ID column not found — generating molecule1, molecule2, ...")

                df_new = pd.DataFrame({'ID': id_series, 'SMILES': smiles_series})
                st.session_state.df_new = df_new
                st.success("✅ Extracted DataFrame:")
                st.dataframe(df_new)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# === 2. Standardize SMILES ===
def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()
    reionizer = rdMolStandardize.Reionizer()

    standardized_list = []
    for smi in tqdm(batch.to_list(), desc='Standardizing...'):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
                mol = rdMolStandardize.Cleanup(mol)
                mol = rdMolStandardize.Normalize(mol)
                mol = uc.uncharge(mol)
                mol = rdMolStandardize.FragmentParent(mol)
                mol = reionizer.reionize(mol)
                mol = md.Disconnect(mol)
                mol = te.Canonicalize(mol)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                standardized_list.append(smiles)
            else:
                standardized_list.append(None)
        except Exception:
            standardized_list.append(None)
    return standardized_list

# === 3. Run Standardization (no download) ===
st.header("Step 2: Standardize SMILES")

if "df_new" in st.session_state:
    if st.button("Standardize"):
        df_standardized = st.session_state.df_new.copy()

        if "SMILES" not in df_standardized.columns:
            st.error("❌ 'SMILES' column not found.")
        else:
            with st.spinner("⏳ Standardizing SMILES..."):
                standardized = standardize_smiles(df_standardized["SMILES"])
                df_standardized["Standardized_SMILES"] = standardized
                st.session_state.df_standardized = df_standardized
                st.success("✅ Standardization complete.")
                st.dataframe(df_standardized)
else:
    st.info("👉 Please complete Step 1 first.")

# === 4. PAINS-FILTER (no download) ===
st.header("Step 3: PAINS Filtering")

if "df_standardized" in st.session_state:
    df_pains = st.session_state.df_standardized.copy()

    # Đảm bảo cột chuẩn hóa tồn tại
    if "Standardized_SMILES" not in df_pains.columns:
        st.error("❌ 'Standardized_SMILES' column not found.")
    else:
        # Đổi tên cột để phù hợp xử lý
        df_pains.rename(columns={"Standardized_SMILES": "standardized"}, inplace=True)

        # Kiểm tra có cột ID
        if "ID" not in df_pains.columns:
            st.error("❌ 'ID' column not found.")
        else:
            # Tạo catalog PAINS
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)

            matches = []
            clean = []

            for index, row in tqdm(df_pains.iterrows(), total=df_pains.shape[0]):
                molecule = Chem.MolFromSmiles(row['standardized'])
                if molecule is None:
                    continue
                entry = catalog.GetFirstMatch(molecule)
                if entry is not None:
                    matches.append({
                        "ID": row['ID'],
                        "standardized": row['standardized'],
                        "PAINS": entry.GetDescription().capitalize(),
                    })
                else:
                    clean.append(index)

            matches_df = pd.DataFrame(matches)
            raw_pains = df_pains.loc[clean]

            # Hiển thị kết quả
            st.subheader("PAINS Matches")
            if not matches_df.empty:
                st.success(f"✅ {len(matches_df)} molecules matched PAINS filters.")
                st.dataframe(matches_df)
            else:
                st.success("✅ No PAINS alerts found.")

            st.subheader("Clean Molecules (No PAINS)")
            st.dataframe(raw_pains)
else:
    st.warning("⚠️ Please complete the 'Standardize' step first.")

# === 5. ECFP4-2048 ===
tqdm.pandas()

st.header("Step 4: Compute ECFP4 Fingerprints")

if "df_select" not in st.session_state:
    if "raw_pains" in locals():
        df_select = raw_pains.copy()
        st.session_state.df_select = df_select
    else:
        st.warning("⚠️ Please complete PAINS filtering first.")
        st.stop()
else:
    df_select = st.session_state.df_select

# Kiểm tra cột standardized
if "standardized" not in df_select.columns:
    st.error("❌ 'standardized' column not found in df_select.")
    st.stop()

# Hàm fingerprint
def smiles_to_ecfp4(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        else:
            return np.full(2048, np.nan)
    except:
        return np.full(2048, np.nan)

# Tính fingerprint khi nhấn nút
if st.button("Generate ECFP4 Fingerprints"):
    with st.spinner("🔬 Generating ECFP4 fingerprints..."):
        ecfp4_matrix = df_select['standardized'].progress_apply(smiles_to_ecfp4)
        ecfp4_df2048 = pd.DataFrame(ecfp4_matrix.tolist(),
                                     columns=[f'bit_{i}' for i in range(2048)])

        df_split = pd.concat([
            df_select[['ID', 'standardized']].reset_index(drop=True),
            ecfp4_df2048.reset_index(drop=True)
        ], axis=1)

        st.session_state.df_split = df_split  # Lưu vào session
        st.success("✅ ECFP4 fingerprints computed and stored.")

# === 6. SCREENING ===
st.header("Step 5: IRAK4 QSAR Screening")

# Đảm bảo fingerprint đã được tính
if "df_split" not in st.session_state:
    st.warning("⚠️ Please generate ECFP4 fingerprints first.")
    st.stop()
else:
    data = st.session_state.df_split.copy()

# Nút chạy cả hai mô hình
if st.button("Run Prediction"):
    try:
        # === Binary Classification ===
        with open('model1/rf_binary_813_tuned.pkl', 'rb') as file:
            rf_model = pickle.load(file)

        X_bin = data.drop(['ID', 'standardized'], axis=1)
        prob_bin = rf_model.predict_proba(X_bin)[:, 1]

        screening_bin = pd.DataFrame({
            'ID': data['ID'],
            'standardized': data['standardized'],
            'label_prob': np.round(prob_bin, 4),
            'label': np.where(prob_bin >= 0.5, 1, 0)
        })

        st.session_state.result = screening_bin.copy()

        # === Regression Prediction ===
        with open('model1/xgb_regression_764_tuned.pkl', 'rb') as file:
            xgb_model = pickle.load(file)

        X_reg = data.drop(['ID', 'standardized'], axis=1)
        predicted_pIC50 = xgb_model.predict(X_reg)

        screening_reg = pd.DataFrame({
            'ID': data['ID'],
            'standardized': data['standardized'],
            'predicted_pIC50': [f"{v:.4f}" for v in predicted_pIC50]
        })

        IC50_nM = 8
        IC50_M = IC50_nM * 1e-9
        base_pIC50 = -np.log10(IC50_M)
        screening_reg['predicted_pIC50'] = pd.to_numeric(screening_reg['predicted_pIC50'], errors='coerce')

        screening_reg['label'] = (screening_reg['predicted_pIC50'] >= base_pIC50).astype(int)

        st.session_state.result_reg = screening_reg.copy()

        # === Consensus Actives ===
        actives_bin = screening_bin[screening_bin['label'] == 1]
        actives_reg = screening_reg[screening_reg['label'] == 1]

        consensus_df = pd.merge(
            actives_bin[['ID', 'standardized']],
            actives_reg[['ID', 'standardized']],
            on=['ID', 'standardized'],
            how='inner'
        )

        consensus_df = pd.merge(
            consensus_df,
            screening_bin[['ID', 'label_prob']],
            on='ID', how='left'
        )

        consensus_df = pd.merge(
            consensus_df,
            screening_reg[['ID', 'predicted_pIC50']],
            on='ID', how='left'
        )

        st.session_state.consensus = consensus_df

        st.success("✅ Prediction complete. Scroll down to view results.")

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý dữ liệu prediction: {e}")

# === Hiển thị bảng kết quả nếu có ===
if "result" in st.session_state:
    st.subheader("🧪 Binary Predicted Actives")
    df_binary_active = st.session_state.result[st.session_state.result['label'] == 1][['ID', 'standardized', 'label_prob', 'label']]
    gb_bin = GridOptionsBuilder.from_dataframe(df_binary_active)
    gb_bin.configure_default_column(filterable=True, sortable=True)
    grid_options_bin = gb_bin.build()
    AgGrid(
        df_binary_active,
        gridOptions=grid_options_bin,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=300,
        theme='alpine'
    )

if "result_reg" in st.session_state:
    st.subheader("📈 Regression Predicted Actives")
    df_reg_active = st.session_state.result_reg[st.session_state.result_reg['label'] == 1][['ID', 'standardized', 'predicted_pIC50', 'label']]
    gb_reg = GridOptionsBuilder.from_dataframe(df_reg_active)
    gb_reg.configure_default_column(filterable=True, sortable=True)
    grid_options_reg = gb_reg.build()
    AgGrid(
        df_reg_active,
        gridOptions=grid_options_reg,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=300,
        theme='alpine'
    )

if "consensus" in st.session_state:
    st.subheader("📊 Consensus Actives")
    gb = GridOptionsBuilder.from_dataframe(st.session_state.consensus)
    gb.configure_default_column(filterable=True, sortable=True)
    grid_options = gb.build()
    AgGrid(
        st.session_state.consensus,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=400,
        theme='alpine'
    )

st.header("Step 6: Select Molecule and Prepare for Docking")

# Create folder to store output if not exists
LIGAND_DIR = "ligands_pdbqt"
os.makedirs(LIGAND_DIR, exist_ok=True)

if "consensus" in st.session_state:
    df = st.session_state.consensus.copy()

    if "selected_mol_id" not in st.session_state:
        st.subheader("📋 Step 6: Select 1 molecule from consensus table")

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filterable=True, sortable=True)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            height=400,
            theme='alpine'
        )

        selected_rows = grid_response["selected_rows"]

        if isinstance(selected_rows, list) and len(selected_rows) > 0:
            row = selected_rows[0]
            mol_id = row.get('ID')
            smiles = row.get('standardized')

            with st.expander("🔐 Confirm selection", expanded=True):
                st.markdown(f"You have selected molecule **{mol_id}**")
                if st.button("✅ Confirm this molecule"):
                    st.session_state.selected_mol_id = mol_id
                    st.session_state.selected_smiles = smiles
                    st.session_state.selected_df = pd.DataFrame([row])
                    st.success(f"✅ Molecule '{mol_id}' has been successfully selected and locked.")
                    st.experimental_rerun()
        else:
            st.info("👉 Please select one row from the table.")

    else:
        st.success(f"✅ Molecule '{st.session_state.selected_mol_id}' has already been selected and locked.")
        st.info("To select another molecule, reset your selection.")
else:
    st.warning("⚠️ consensus_df does not exist. Please run previous steps first.")


