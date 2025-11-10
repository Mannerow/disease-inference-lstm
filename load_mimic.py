import pandas as pd
import os
import gzip
import collections

def load_mimic_data(data_dir):
    """
    Load MIMIC-III dataset from the specified directory.
    """
    tables = ['ADMISSIONS', 'NOTEEVENTS', 'DIAGNOSES_ICD', 'D_ICD_DIAGNOSES',]
    data = {}
    
    for table in tables:
        file_path = os.path.join(data_dir, f"{table}.csv.gz")
        if os.path.exists(file_path):
            with gzip.open(file_path, 'rt') as f:
                data[table] = pd.read_csv(f)
        else:
            raise FileNotFoundError(f"{file_path} not found.")
    
    return data

# Only keep category = discharge summary
# Section 2.1
def filter_discharges(data, noteevents_df):
    noteevents_df = noteevents_df[noteevents_df['CATEGORY'] == 'Discharge summary']
    noteevents_df = noteevents_df.dropna(subset=['TEXT'])
    # Remote dupes
    noteevents_df = noteevents_df.drop_duplicates(subset=['HADM_ID'], keep='last')
    return noteevents_df

# Should be  total of 931 distinct ICD9 codes
# 2.1
def truncate_icd9_codes(diag_codes):
    df = diag_codes.copy()
    df['ICD9_CODE_3DIGIT'] = df['ICD9_CODE'].fillna('').astype(str).str.slice(0, 3)
    df = df[df['ICD9_CODE_3DIGIT'] != '']
    print(f"Total unique 3-digit codes: {df['ICD9_CODE_3DIGIT'].nunique()} (expected: 931)")
    return df

# Section 3.1
def select_top_diseases(diagnoses_df, top_n=50):
    disease_counts = collections.Counter(diagnoses_df['ICD9_CODE_3DIGIT'])
    top_diseases = [disease for disease, count in disease_counts.most_common(top_n)]

    disease_to_index = {disease: idx for idx, disease in enumerate(top_diseases)}

    # calculate coverage
    total_admissions = diagnoses_df['HADM_ID'].nunique()
    admissions_with_top_diseases = diagnoses_df[
        diagnoses_df['ICD9_CODE_3DIGIT'].isin(top_diseases)
    ]['HADM_ID'].nunique()
    coverage = admissions_with_top_diseases / total_admissions

    print(f"Top {top_n} diseases cover {coverage:.2%} of all diagnoses.")
    
    return top_diseases, disease_to_index, coverage

# Filter for top dieases
# Section 3.1
def filter_top_diseases(diagnoses_df, noteevents_df, top_diseases):
    filtered = diagnoses_df[diagnoses_df['ICD9_CODE_3DIGIT'].isin(top_diseases)].copy()

    unique_hadm_ids = filtered['HADM_ID'].unique()
    
    filtered_notes = noteevents_df[noteevents_df['HADM_ID'].isin(filtered['HADM_ID'].unique())].copy()

    print(f"  Unique admissions with top diseases: {len(unique_hadm_ids)}")
    print(f"  Discharge summaries retained: {len(filtered_notes)}")
    print(f"  Expected (from paper): 46,364 for top-50 or 46,715 for top-100")
    
    return filtered, filtered_notes

if __name__ == "__main__":
    data_directory = "data"
    top_n_diseases = 50

    print("Loading MIMIC-III data...")
    mimic_data = load_mimic_data(data_directory)

    print('Dataset Summary...')
    for table_name, df in mimic_data.items():
        print(f"{table_name}: {df.shape[0]} rows, {df.shape[1]} columns")

    noteevents_df = mimic_data['NOTEEVENTS']
    print("Filtering Discharge Notes...")
    filtered_notes = filter_discharges(mimic_data, noteevents_df)

    print("Truncating ICD9 Codes...")
    diag_codes_df = mimic_data['DIAGNOSES_ICD']
    diagnoses = truncate_icd9_codes(diag_codes_df)

    print("Selecting Top Diseases...")
    top_diseases, disease_to_index, coverage = select_top_diseases(diagnoses, top_n=top_n_diseases)

    print("\n Top Diseases")
    for i, disease in enumerate(top_diseases[:10], 1):
        print(f"{i}. {disease}")

    print("Filtering for Top Diseases...")
    filtered_diagnoses, filtered_notes = filter_top_diseases(diagnoses, filtered_notes, top_diseases)








