import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm

st.title('🎈 IRAK4 SCREENING')
st.info('This is an app built for predicting IRAK4 inhibitors')

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
