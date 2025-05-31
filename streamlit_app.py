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

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm

def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()

    standardized_list = []
    for smi in tqdm(batch.to_list(), desc='Processing . . .'):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
                cleanup = rdMolStandardize.Cleanup(mol)
                normalized = rdMolStandardize.Normalize(cleanup)
                uncharged = uc.uncharge(normalized)
                fragment = uc.uncharge(rdMolStandardize.FragmentParent(uncharged))
                ionized = rdMolStandardize.Reionize(fragment)
                disconnected = md.Disconnect(ionized)
                tautomer = te.Canonicalize(disconnected)
                smiles = Chem.MolToSmiles(tautomer, isomericSmiles=False, canonical=True)
                standardized_list.append(smiles)
            else:
                standardized_list.append(None)
                print(f"Invalid SMILES: {smi}")
        except Exception as e:
            print(f"An error occurred with SMILES {smi}: {str(e)}")
            standardized_list.append(None)

    return standardized_list

df_standardized = st.session_state.df_new.copy()

standardized_list = standardize_smiles(df_standardized['SMILES'])
df_standardized['standardized'] = standardized_list





