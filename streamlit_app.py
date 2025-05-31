import streamlit as st

st.title('🎈 IRAK4 SCREENING')

st.info('This is an app build for predicting IRAK4 inhibitors')

import pandas as pd
import io
import ipywidgets as widgets
from IPython.display import display

# === 1. File Upload Widget ===
upload_label = widgets.Label("Input file:")
upload_button = widgets.FileUpload(
    accept='.csv,.txt',
    multiple=False
)

# Add padding on the left using Box layout
upload_row = widgets.HBox([upload_label, upload_button])
upload_row.layout.margin = '0 0 0 40px'  # top right bottom left

# === 2. Input Fields (with full label display and margin) ===
col1_input = widgets.Text(
    value='',
    placeholder='Molecule_Name',
    description='ID column (optional):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px', margin='0 0 0 40px')
)

col2_input = widgets.Text(
    value='',
    placeholder='SMILES',
    description='SMILES column (required):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px', margin='0 0 0 40px')
)

# Button
convert_button = widgets.Button(
    description="Create",
    layout=widgets.Layout(width='100px', margin='0 0 0 40px')
)

# Output
output_area = widgets.Output()

# === 3. Handler Function ===
def convert_columns(b):
    global df_new 
    output_area.clear_output()
    with output_area:
        if not upload_button.value:
            print("❗ No file uploaded.")
            return

        id_col = col1_input.value.strip()
        smiles_col = col2_input.value.strip()

        if not smiles_col:
            print("❗ Please enter the SMILES column name.")
            return

        for name, file_info in upload_button.value.items():
            content = file_info['content']
            try:
                df = pd.read_csv(io.BytesIO(content))
                print(f"📄 File loaded: {name}")
                
                if smiles_col not in df.columns:
                    print(f"❌ SMILES column '{smiles_col}' not found.")
                    print(f"Available columns: {list(df.columns)}")
                    return

                smiles_series = df[smiles_col]

                if id_col and id_col in df.columns:
                    id_series = df[id_col]
                    print(f"🔧 Using column '{id_col}' for ID.")
                else:
                    id_series = [f"molecule{i+1}" for i in range(len(smiles_series))]
                    print("ℹ️ ID column not provided or not found — generating default IDs (molecule1, molecule2, ...).")

                df_new = pd.DataFrame({'ID': id_series, 'SMILES': smiles_series})
                print("✅ New DataFrame with 'ID' and 'SMILES':")
                display(df_new)

            except Exception as e:
                print(f"⚠️ Error processing file: {e}")

# Wire event
convert_button.on_click(convert_columns)

# === 4. Display the UI with shifted layout and full labels ===
display(widgets.VBox([
    upload_row,
    col1_input,
    col2_input,
    convert_button,
    output_area
]))

