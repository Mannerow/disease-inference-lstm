import pandas as pd
import os
import gzip
import collections
import re

# Symptom Lexicon (mocks UMLS 'sosy' terms). INstead of MetaMap
SYMPTOMS = {
    'fever', 'chills', 'night sweats', 'fatigue', 'malaise', 'weakness',
    'cough', 'shortness of breath', 'dyspnea', 'wheezing', 'chest pain',
    'palpitations', 'nausea', 'vomiting', 'diarrhea', 'constipation',
    'abdominal pain', 'bloating', 'heartburn', 'dysphagia', 'hematemesis',
    'melena', 'hematochezia', 'jaundice', 'pruritus', 'weight loss',
    'anorexia', 'dysuria', 'hematuria', 'frequency', 'urgency', 'incontinence',
    'headache', 'dizziness', 'vertigo', 'syncope', 'seizure', 'confusion',
    'memory loss', 'depression', 'anxiety', 'insomnia', 'back pain',
    'neck pain', 'joint pain', 'arthralgia', 'myalgia', 'edema', 'rash',
    'itching', 'bleeding', 'bruising', 'petechiae', 'cyanosis', 'pallor',
    'tremor', 'ataxia', 'numbness', 'tingling', 'paresthesia'
}

# Normalize: lower + handle variations
SYMPTOMS = {s.lower() for s in SYMPTOMS}

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
    print(f"  Expected: 46,364 for top-50 or 46,715 for top-100")
    
    return filtered, filtered_notes

# Section 2.1
# Need to find section headers in the discharge summaries
def identify_sections(text):
    # match section headers that may appear at start, middle, or end
    section_pattern = r'(^|\n)\s*([A-Z][A-Za-z\s/&-]+:)\s*(\n|\Z)'
    matches = re.findall(section_pattern, text)
    return [m[1].rstrip(':') for m in matches]

# Need to remove these
# Section 2.1
def filter_sections(text):
    remove_sections = [
        'Social History',
        'Medications on Admission',
        'Discharge Diagnosis',
    ]

    split_pattern = r'(\n\s*[A-Z][A-Za-z\s/&-]+:\s*\n)'
    parts = re.split(split_pattern, text)

    # keep only non-removed sections
    filtered_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if re.match(r'\n\s*[A-Z][A-Za-z\s/&-]+:\s*\n', part):

            # This is a section header
            header = part.strip().rstrip(':')
            if header not in remove_sections:
                # Keep header + next content block
                filtered_parts.append(part)
                if i + 1 < len(parts):
                    filtered_parts.append(parts[i + 1])
            else:
                # Skip header and content
                i += 1  
            i += 1 
        else:
            if not filtered_parts:
                filtered_parts.append(part)
            i += 1

    return ''.join(filtered_parts)

def process_section_filtering(input_file, output_file):
    df = pd.read_pickle(input_file)
    df['TEXT_FILTERED'] = df['TEXT_CLEANED'].apply(filter_sections)
    df.to_pickle(output_file)
    print(f"Filtered {len(df)} summaries")

# Section 2.1
# Mocking metamap. Replace with Metamap later
def extract_symptoms_from_text(text, symptom_set):
    if pd.isna(text):
        return []
    text_lower = text.lower()
    found_symptoms = []

    NEGATION_PATTERNS = [
        r'\b(denies?|denied|no\s+(history\s+of\s+)?|without|never\s+had|not\s+complaining|not\s+report|not\s+have|not\s+experiencing|absence\s+of)\b'
    ]

    for sym in symptom_set:
        escaped = re.escape(sym)
        # Handle multi-word symptoms
        pattern = r'\b' + escaped.replace(' ', r'\s+') + r'\b'
        matches = list(re.finditer(pattern, text_lower))

        for m in matches:
            start, end = m.span()
            # Look back ~50 chars
            window_start = max(0, start - 50)
            pre_text = text_lower[window_start:start]

            is_negated = False
            for neg_pat in NEGATION_PATTERNS:
                if re.search(neg_pat, pre_text):
                    is_negated = True
                    break

            found_symptoms.append({
                'symptom': sym,
                'is_negated': is_negated
            })

    # Deduplicate
    seen = set()
    unique = []
    for item in found_symptoms:
        key = (item['symptom'], item['is_negated'])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# Section 3.3
def filter_by_freq_and_length(notes_df, min_symptoms=10, min_symptoms_per_note=2, max_symptoms_per_note=50):
    all_symptoms = [s for lst in notes_df['SYMPTOMS_POS'] for s in lst]
    symptom_counts = collections.Counter(all_symptoms)
    valid_symptoms = {sym for sym, cnt in symptom_counts.items() if cnt >= min_symptoms}

    # Frequency filtering
    all_symptoms = [s for lst in notes_df['SYMPTOMS_POS'] for s in lst]
    symptom_counts = collections.Counter(all_symptoms)
    valid_symptoms = {sym for sym, cnt in symptom_counts.items() if cnt >= min_symptoms}

    # Keep only valid symptoms per note
    notes_df['SYMPTOMS_VALID'] = notes_df['SYMPTOMS_POS'].apply(
        lambda lst: [s for s in lst if s in valid_symptoms]
    )

    # Take first 50 symptoms
    notes_df['SYMPTOMS_TRUNCATED'] = notes_df['SYMPTOMS_VALID'].apply(
        lambda lst: lst[:max_symptoms_per_note]
    )

    # Filter out to few symptoms
    filtered_df = notes_df[
        notes_df['SYMPTOMS_TRUNCATED'].apply(len) >= min_symptoms_per_note
    ].copy()

    filtered_df['SYMPTOMS_FINAL'] = filtered_df['SYMPTOMS_TRUNCATED']
    print(f"  Final note count: {len(filtered_df)}")
    return filtered_df, valid_symptoms

def build_symptom_dict(notes_df):
    all_symptoms = [s for lst in notes_df['SYMPTOMS_FINAL'] for s in lst]
    unique_symptoms = sorted(set(all_symptoms))
    symptom_to_index = {symptom: idx for idx, symptom in enumerate(unique_symptoms)}
    print(f"Total unique symptoms after filtering: {len(unique_symptoms)}")
    print("Expected: 4200 symptoms")
    return symptom_to_index



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

    print("Identifying Sections in Discharge Summaries...")
    filtered_notes['TEXT_FILTERED'] = filtered_notes['TEXT'].apply(filter_sections)

    print("Extracting symptoms (mocking MetaMap)...")
    filtered_notes['RAW_SYMPTOMS'] = filtered_notes['TEXT_FILTERED'].apply(
        lambda x: extract_symptoms_from_text(x, SYMPTOMS)
    )

    print("Removing Negative Symptoms...")
    filtered_notes['SYMPTOMS_POS'] = filtered_notes['RAW_SYMPTOMS'].apply(
        # filters out negated symptoms
        lambda lst: [item['symptom'] for item in lst if not item['is_negated']]
    )

    print("Applying symptom frequency and length filtering...")
    final_notes_df, valid_symptoms = filter_by_freq_and_length(
        filtered_notes,
        min_symptoms=10,
        min_symptoms_per_note=2,
        max_symptoms_per_note=50
    )

    # map symtpoms to integers
    symptom_to_index = build_symptom_dict(final_notes_df)

