import streamlit as st

st.title('🎈 IRAK4 SCREENING')

st.info('This is an app build for predicting IRAK4 inhibitors')

# streamlit_app.py

import streamlit as st
import pandas as pd
import io

st.title("Upload CSV/TXT and Extract ID & SMILES")

# === 1. File uploader ===
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=['csv', 'txt'])

# === 2. Input fields ===
id_col = st.text_input("ID column (optional)", value="", placeholder="Molecule_Name")
smiles_col = st.text_input("SMILES column (required)", value="", placeholder="SMILES")

# === 3. Process button ===
if st.button("Create"):
    if uploaded_file is None:
        st.warning("Please upload a file first.")
    elif not smiles_col.strip():
        st.warning("Please enter the SMILES column name.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded: {uploaded_file.name}")

            if smiles_col not in df.columns:
                st.error(f"SMILES column '{smiles_col}' not found. Available columns: {list(df.columns)}")
            else:
                smiles_series = df[smiles_col]
                if id_col and id_col in df.columns:
                    id_series = df[id_col]
                    st.info(f"Using column '{id_col}' for ID.")
                else:
                    id_series = [f"molecule{i+1}" for i in range(len(smiles_series))]
                    st.info("ID column not found — generating IDs: molecule1, molecule2, ...")

                df_new = pd.DataFrame({'ID': id_series, 'SMILES': smiles_series})
                st.success("✅ Extracted DataFrame with 'ID' and 'SMILES':")
                st.dataframe(df_new)

                # Option to download
                csv = df_new.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, file_name="id_smiles.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolStandardize
from tqdm.auto import tqdm

st.title("SMILES Standardizer")

# === Step 1: Upload file & get column names ===
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=['csv', 'txt'])

id_col = st.text_input("ID column (optional)", placeholder="e.g. Molecule_Name")
smiles_col = st.text_input("SMILES column (required)", placeholder="e.g. SMILES")

# === Step 2: Create df_new from uploaded file ===
if st.button("Create"):
    if uploaded_file is None:
        st.warning("Please upload a file.")
    elif not smiles_col:
        st.warning("Please enter SMILES column name.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            if smiles_col not in df.columns:
                st.error(f"Column '{smiles_col}' not found. Available: {list(df.columns)}")
            else:
                smiles_series = df[smiles_col]
                if id_col and id_col in df.columns:
                    id_series = df[id_col]
                else:
                    id_series = [f"molecule{i+1}" for i in range(len(smiles_series))]
                df_new = pd.DataFrame({'ID': id_series, 'SMILES': smiles_series})
                st.session_state.df_new = df_new
                st.success("✅ DataFrame created:")
                st.dataframe(df_new)
        except Exception as e:
            st.error(f"Error loading file: {e}")

# === Step 3: Define standardization function ===
def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()

    standardized = []
    for smi in tqdm(batch.to_list(), desc="Standardizing..."):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
                mol = rdMolStandardize.Cleanup(mol)
                mol = rdMolStandardize.Normalize(mol)
                mol = uc.uncharge(mol)
                mol = rdMolStandardize.FragmentParent(mol)
                mol = md.Disconnect(mol)
                mol = te.Canonicalize(mol)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                standardized.append(smiles)
            else:
                standardized.append(None)
        except Exception:
            standardized.append(None)
    return standardized

# === Step 4: Run standardization ===
if "df_new" in st.session_state:
    st.subheader("Step 2: Standardize SMILES")
    if st.button("Standardize"):
        with st.spinner("Standardizing SMILES..."):
            df_std = st.session_state.df_new.copy()
            df_std["Standardized_SMILES"] = standardize_smiles(df_std["SMILES"])
            st.session_state.df_standardized = df_std
            st.success("Standardization complete.")
            st.dataframe(df_std)

            # Download
            csv = df_std.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "standardized_smiles.csv", "text/csv")
else:
    st.info("Please complete the 'Create' step first.")



