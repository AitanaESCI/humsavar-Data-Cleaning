import pandas as pd
import numpy as np
from collections import Counter


def clean_categorical_and_score(df, column, positive_labels, negative_labels, binary_labels=("P", "B")):
    # Extract category and score, ensuring score is float
    # Extract the string before '('
    df[f"{column}_cat"] = df[column].apply(lambda x: np.nan if pd.isna(x) or x == "-" else x.split('(')[0])

    # Extract the numeric value inside parentheses
    df[f"{column}_quant"] = df[column].apply(lambda x: np.nan if pd.isna(x) or x == "-" else x.split('(')[1].split(')')[0] if '(' in x else np.nan)
    df[f"{column}_quant"] = pd.to_numeric(df[f"{column}_quant"], errors='coerce')

    # Create binary column based on category
    binary_col = f"{column}_binary"
    df[binary_col] = np.nan
    df.loc[df[f"{column}_cat"].isin(positive_labels), binary_col] = binary_labels[0]
    df.loc[df[f"{column}_cat"].isin(negative_labels), binary_col] = binary_labels[1]

    # Move original and new columns to the end
    cols_to_move = [column, f"{column}_cat", f"{column}_quant", binary_col]
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move]

    print(f'{column} // {column}_cat // {column}_quant // {binary_col}\n')
    return df

def clean_numerical_with_threshold(df, score_column, threshold, binary_labels=("P", "B")):
    df = df.copy()
    # Convert to numeric to ensure score column is float
    df[f"{score_column}_parsed"] = pd.to_numeric(df[score_column], errors='coerce')

    # Apply threshold to create binary label
    if score_column in ['REVEL','Eigen-PC-raw_coding','MVP_score']:
        binary_col = f"{score_column}_binary_(thr_P>{threshold})"
        df[binary_col] = np.nan
        df.loc[df[f"{score_column}_parsed"] > threshold, binary_col] = binary_labels[0]
        df.loc[df[f"{score_column}_parsed"] <= threshold, binary_col] = binary_labels[1]
    else:
        binary_col = f"{score_column}_binary_(thr_P>={threshold})"
        df[binary_col] = np.nan
        df.loc[df[f"{score_column}_parsed"] >= threshold, binary_col] = binary_labels[0]
        df.loc[df[f"{score_column}_parsed"] < threshold, binary_col] = binary_labels[1]

    # Move original and new columns to the end
    cols_to_move = [score_column, f"{score_column}_parsed", binary_col]
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move]

    print(f'{score_column} // {score_column}_parsed // {binary_col}\n')
    return df


def clean_score_with_categorical_pred(df, score_column, pred_column, positive_labels, negative_labels, binary_labels=("P", "B")):
    df = df.copy()
    all_columns = list(df.columns)

    # Convert non-numeric values to NaN in the score column
    df[f"{score_column}_parsed"] = pd.to_numeric(df[score_column], errors='coerce')  # Ensures the score column is float
    #print(df[[pred_column]].value_counts())

    # Generate binary column based on prediction labels
    binary_col = f"{pred_column}_binary"
    df[binary_col] = np.nan
    df.loc[df[pred_column].isin(positive_labels), binary_col] = binary_labels[0]
    df.loc[df[pred_column].isin(negative_labels), binary_col] = binary_labels[1]
    #print(df[[binary_col]].value_counts(), '\n')

    # Move original and new columns to the end of the DataFrame
    cols_to_move = [score_column, pred_column, f"{score_column}_parsed", binary_col]
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move]

    print(f'{score_column} // {pred_column} // {score_column}_parsed // {binary_col}\n')

    return df



def clean_pred_score_columns(df, pred_column, score_column, binary_map={"D": "P", "T": "B", "H": "P", "M": "P", "L": "B", "N": "B"}):
    df = df.copy()
    all_columns = list(df.columns)

    # Process columns with alignment between pred and score values
    def process_columns(pred_str, score_str):
        # Parse the pred column: split and filter out non-informative values
        preds = [val for val in str(pred_str).split(',') if val not in ['.', '-']]

        # Parse the score column: split, filter, and convert to float if possible
        scores = [np.nan if val in ['.', '-'] else float(val) for val in str(score_str).split(',')]

        if not preds:  # If preds list is empty after filtering, return NaN
            return np.nan, np.nan, np.nan

        # Get the most common prediction(s) and check for a tie
        pred_counts = Counter(preds)
        most_common = pred_counts.most_common()

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie detected, take the first element in the preds list
            majority_pred = preds[0]
            majority_index = str(pred_str).split(',').index(majority_pred)  # Find the index in the original pred_str
        else:
            # No tie, take the most frequent prediction
            majority_pred = most_common[0][0]
            majority_index = str(pred_str).split(',').index(majority_pred)  # Get the index of the first occurrence

        corresponding_score = scores[majority_index] if scores[majority_index] is not np.nan else np.nan
        binary_label = binary_map.get(majority_pred, np.nan)  # Map the pred to P/B or NaN if unmapped

        return majority_pred, corresponding_score, binary_label

    # Apply the function to columns
    parsed_columns = [f"{pred_column}_parsed", f"{score_column}_parsed", f"{pred_column}_binary"]
    df[parsed_columns] = df.apply(lambda row: process_columns(row[pred_column], row[score_column]), axis=1, result_type="expand")

    print(f'{pred_column} // {score_column} // {pred_column}_parsed  // {score_column}_parsed  // {pred_column}_binary\n')

    # Preserve original column order and add new columns at the end
    cols_to_move = [score_column, pred_column] + parsed_columns
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move]

    return df

def clean_mutliple_score_with_categorical(df, score_column, threshold):
    df = df.copy()
    all_columns = list(df.columns)

    def process_scores(score_str):
        # Check if the input is a string; if not, return NaN
        if not isinstance(score_str, str):
            return np.nan

        # Split, filter, and keep only valid numerical values
        scores = [float(val) for val in score_str.split(',') if val not in ['.', '-', '', None]]

        if not scores:  # If list is empty after filtering, set to NaN
            return np.nan

        # Take the most common score; if there's a tie, take the first most common score
        score_counts = Counter(scores)
        most_common_score = score_counts.most_common(1)[0][0]  # First most common score in case of tie

        return most_common_score

    df[f'{score_column}_parsed'] = df[f'{score_column}'].apply(process_scores)

    # Create binary column based on threshold in MutPred_score_parsed
    binary_col = f"{score_column}_binary_(thr_P>={threshold})"
    df[binary_col] = np.nan
    df.loc[df[f"{score_column}_parsed"] >= threshold, binary_col] = 'P'
    df.loc[df[f"{score_column}_parsed"] < threshold, binary_col] = 'B'

    # Apply the function to columns
    parsed_columns = [f"{score_column}_parsed", binary_col]

    print(f'{score_column} // {score_column}_parsed // {binary_col}\n')

    # Preserve original column order and add new columns at the end
    cols_to_move = [score_column] + parsed_columns
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move]
    return df



# combine all in one function
def process_columns(pred_str, score_str, binary_map=None):
    """
    Process prediction and score columns
    """
    # Initialize default return values
    default_return = (np.nan, np.nan, np.nan)

    # Check for missing or invalid values
    if pd.isna(pred_str) or pd.isna(score_str) or pred_str == '-' or score_str == '-':
        return default_return

    try:
        # Split predictions and scores
        preds = pred_str.split('&')
        scores = score_str.split('&')

        # Handle empty strings or invalid formats
        if not preds or not scores or len(preds) != len(scores):
            return default_return

        # Get the majority prediction
        majority_pred = max(set(preds), key=preds.count)
        majority_index = preds.index(majority_pred)

        # Get corresponding score
        corresponding_score = scores[majority_index] if scores[majority_index] != '-' else np.nan
        try:
            corresponding_score = float(corresponding_score)
        except (ValueError, TypeError):
            corresponding_score = np.nan

        # Get binary label
        binary_label = np.nan
        if binary_map is not None and majority_pred in binary_map:
            binary_label = binary_map[majority_pred]

        return majority_pred, corresponding_score, binary_label

    except (IndexError, ValueError, AttributeError):
        return default_return


def clean_vep_data(df):
    """
    Clean VEP data using predefined configurations with improved error handling
    """

    # Configuration dictionaries
    CATEGORICAL_CONFIG = {
        "SIFT": {
            "positive": ["deleterious", "deleterious_low_confidence"],
            "negative": ["tolerated", "tolerated_low_confidence"]
        },
        "PolyPhen": {
            "positive": ["probably_damaging", "possibly_damaging"],
            "negative": ["benign"]
        }
    }

    NUMERICAL_THRESHOLDS = {
        "CADD_PHRED": 19,
        "REVEL": 0.5,
        "ClinPred": 0.5,
        'Eigen-PC-raw_coding': 0,
        'Eigen-raw_coding': 0,
        'GERP++_RS': 2,
        'DANN_score': 0.99,
        'MVP_score': 0.7
    }

    SCORE_PRED_CONFIG = {
        "BayesDel_noAF": {"score": "BayesDel_noAF_score", "pred": "BayesDel_noAF_pred", "positive": ["D"], "negative": ["T"]},
        "am": {"score": "am_pathogenicity", "pred": "am_class", "positive": ["likely_pathogenic"], "negative": ["likely_benign"]},
        "EVE": {"score": "EVE_SCORE", "pred": "EVE_CLASS", "positive": ["Pathogenic"], "negative": ["Benign"]},
        "MetaLR": {"score": "MetaLR_score", "pred": "MetaLR_pred", "positive": ["D"], "negative": ["T"]},
        "MetaSVM": {"score": "MetaSVM_score", "pred": "MetaSVM_pred", "positive": ["D"], "negative": ["T"]},
        "LRT": {"score": "LRT_score", "pred": "LRT_pred", "positive": ["D"], "negative": ["N"]},
        "M-CAP": {"score": "M-CAP_score", "pred": "M-CAP_pred", "positive": ["D"], "negative": ["T"]},
        "PrimateAI": {"score": "PrimateAI_score", "pred": "PrimateAI_pred", "positive": ["D"], "negative": ["T"]}
    }

    MULTIPLE_SCORE_THRESHOLDS = {
        "MutPred_score": 0.5,
        "VARITY_R_score": 0.5,
        "MPC_score": 2,
        "gMVP_score": 0.75
    }


    # Default binary map that was in the original function
    DEFAULT_BINARY_MAP = {"D": "P", "T": "B", "H": "P", "M": "P", "L": "B", "N": "B"}

    # Modified PRED_SCORE_COLUMNS to use the default binary map where none is specified
    PRED_SCORE_COLUMNS = {
    "FATHMM": {"pred": "FATHMM_pred", "score": "FATHMM_score", "binary_map": DEFAULT_BINARY_MAP},
    "MutationTaster": {"pred": "MutationTaster_pred", "score": "MutationTaster_score", "binary_map": DEFAULT_BINARY_MAP},
    "MutationAssessor": {
        "pred": "MutationAssessor_pred",
        "score": "MutationAssessor_score",
        "binary_map": {"H": "P", "M": "P", "L": "B", "N": "B"}  # This one keeps its specific map
        },
    "ESM1b": {"pred": "ESM1b_pred", "score": "ESM1b_score", "binary_map": DEFAULT_BINARY_MAP},
    "MetaRNN": {"pred": "MetaRNN_pred", "score": "MetaRNN_score", "binary_map": DEFAULT_BINARY_MAP},
    "PROVEAN": {"pred": "PROVEAN_pred", "score": "PROVEAN_score", "binary_map": DEFAULT_BINARY_MAP},
    "DEOGEN2": {"pred": "DEOGEN2_pred", "score": "DEOGEN2_score", "binary_map": DEFAULT_BINARY_MAP},
    "LIST-S2": {"pred": "LIST-S2_pred", "score": "LIST-S2_score", "binary_map": DEFAULT_BINARY_MAP}
    }


    # Select the columns of interest
    cols_select=[
    # informative
    '#Uploaded_variation','Location','Allele','Consequence','SYMBOL','Gene','Feature_type','Feature','BIOTYPE','Protein_position','Amino_acids',
    'ENSP','UNIPROT_ISOFORM','CLIN_SIG','aapos','genename','Ensembl_geneid','Ensembl_proteinid','Ensembl_transcriptid','Uniprot_acc','Uniprot_entry',
    'HGNC_ID','MANE','MANE_SELECT','TSL',
    # predictors
    'SIFT','PolyPhen',
    'BayesDel_addAF_pred','BayesDel_addAF_rankscore','BayesDel_addAF_score','BayesDel_noAF_pred','BayesDel_noAF_rankscore','BayesDel_noAF_score',
    'DANN_rankscore','DANN_score',
    'DEOGEN2_pred','DEOGEN2_rankscore','DEOGEN2_score',
    'ESM1b_pred','ESM1b_rankscore','ESM1b_score',
    'Eigen-PC-phred_coding','Eigen-PC-raw_coding','Eigen-PC-raw_coding_rankscore','Eigen-phred_coding','Eigen-raw_coding','Eigen-raw_coding_rankscore',
    'FATHMM_converted_rankscore','FATHMM_pred','FATHMM_score',
    'GERP++_NR','GERP++_RS','GERP++_RS_rankscore',
    #'GM12878_confidence_value','GM12878_fitCons_rankscore','GM12878_fitCons_score',
    #'HUVEC_confidence_value','HUVEC_fitCons_rankscore','HUVEC_fitCons_score',
    'LIST-S2_pred','LIST-S2_rankscore','LIST-S2_score',
    'LRT_Omega','LRT_converted_rankscore','LRT_pred','LRT_score',
    'M-CAP_pred','M-CAP_rankscore','M-CAP_score',
    'MPC_rankscore','MPC_score',
    'MVP_rankscore','MVP_score',
    'MetaLR_pred','MetaLR_rankscore','MetaLR_score',
    'MetaRNN_pred','MetaRNN_rankscore','MetaRNN_score',
    'MetaSVM_pred','MetaSVM_rankscore','MetaSVM_score',
    'MutPred_AAchange','MutPred_Top5features','MutPred_protID','MutPred_rankscore','MutPred_score',
    'MutationAssessor_pred','MutationAssessor_rankscore','MutationAssessor_score',
    'MutationTaster_AAE','MutationTaster_converted_rankscore','MutationTaster_model','MutationTaster_pred','MutationTaster_score',
    'PROVEAN_converted_rankscore','PROVEAN_pred','PROVEAN_score',
    'PrimateAI_pred','PrimateAI_rankscore','PrimateAI_score',
    'VARITY_ER_LOO_rankscore','VARITY_ER_LOO_score','VARITY_ER_rankscore','VARITY_ER_score','VARITY_R_LOO_rankscore','VARITY_R_LOO_score','VARITY_R_rankscore','VARITY_R_score',
    'fathmm-MKL_coding_group','fathmm-MKL_coding_pred','fathmm-MKL_coding_rankscore','fathmm-MKL_coding_score',
    'fathmm-XF_coding_pred','fathmm-XF_coding_rankscore','fathmm-XF_coding_score',
    'gMVP_rankscore','gMVP_score',
    'integrated_confidence_value','integrated_fitCons_rankscore','integrated_fitCons_score',
    'phastCons470way_mammalian','phastCons470way_mammalian_rankscore',
    'phyloP470way_mammalian','phyloP470way_mammalian_rankscore',
    'ClinPred',
    #'BLOSUM62',
    'REVEL',
    'am_class','am_pathogenicity',
    'EVE_CLASS','EVE_SCORE',
    'CADD_PHRED','CADD_RAW']

    df = df[cols_select].copy()
    # Convert to numeric and then to integers, handling NaN values
    df['position'] = pd.to_numeric(df['Protein_position'], errors='coerce').dropna().astype(int)
    df=df[df.position.notna()].reset_index(drop=True)

    # Clean categorical and score
    for col, config in CATEGORICAL_CONFIG.items():
        if col in df.columns:
            df = clean_categorical_and_score(
                df, col, config['positive'], config['negative']
            )

    # Clean numerical with threshold
    for col, threshold in NUMERICAL_THRESHOLDS.items():
        if col in df.columns:
            df = clean_numerical_with_threshold(df, col, threshold)

    # Clean score with categorical pred
    for config in SCORE_PRED_CONFIG.values():
        if config['score'] in df.columns and config['pred'] in df.columns:
            df = clean_score_with_categorical_pred(
                df, config['score'], config['pred'],
                config['positive'], config['negative']
            )

    # Clean multiple score categorical
    for col, threshold in MULTIPLE_SCORE_THRESHOLDS.items():
        if col in df.columns:
            df = clean_mutliple_score_with_categorical(df, col, threshold)

   # Clean pred score columns
    for config in PRED_SCORE_COLUMNS.values():
        if config['pred'] in df.columns and config['score'] in df.columns:
            df = clean_pred_score_columns(
                df,
                config['pred'],
                config['score'],
                binary_map=config['binary_map']  # Using the map directly from config
            )

    df=df.drop_duplicates().reset_index(drop=True)

    return df
