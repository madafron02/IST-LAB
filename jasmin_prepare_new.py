import csv
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 

# Plans for potential prettifying of this code
# - put the first parts of the JASMIN path in a separate field
# - prepare pairs of filepaths for the 2 styles to not do the whole thing twice

# Define function to read CSV files into dictionaries
def read_csv_to_dict(file_path, key_column, delimiter=" "):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            # if len(row) > 1:
            #     data[row[0]] = row[1:] if len(row) > 2 else row[1]
            data[row[0]] = [row[1]]
    return data

# Define function to read transcription CSV files into dictionaries
def read_csv_to_dict_transcript(file_path, key_column, delimiter=" "):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            data[row[0]] = " ".join(row[1:])
    return data

# Read Metadata CSV into dict that maps speaker ID to demographic group
def read_meta_csv_to_dict(file_path, spk_id_column_idx, info_column_idx, delimiter=","):
    metadata = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            key = row[spk_id_column_idx].upper()    # speaker ID, uppercase
            value = row[info_column_idx]          # demographic group as value
            metadata[key] = value
    return metadata

def prepare_dataset(dataset_name, output_format="csv"):
    ### File paths 
    
    # Demographic group per speaker
    spk2meta_read_file = "Rd_meta.csv"
    spk2meta_hmi_file = "HMI_meta.csv"

    # Speaker Gender
    spk2gender_read_file = "JASMIN/kaldi_procesed/nl_test_read_all_hires/spk2gender"
    spk2gender_hmi_file = "JASMIN/kaldi_procesed/nl_test_hmi_all_hires/spk2gender"

    # Speaker ID per utterance
    utt2spk_read_file = "JASMIN/kaldi_procesed/nl_test_read_all_hires/utt2spk"
    utt2spk_hmi_file = "JASMIN/kaldi_procesed/nl_test_hmi_all_hires/utt2spk"

    # Transcriptions
    text_read_file = "JASMIN/kaldi_procesed/nl_test_read_all_hires/text"
    text_hmi_file = "JASMIN/kaldi_procesed/nl_test_hmi_all_hires/text"

    # Utterance durations
    utt2dur_read_file = "JASMIN/kaldi_procesed/nl_test_read_all_hires/utt2dur"
    utt2dur_hmi_file = "JASMIN/kaldi_procesed/nl_test_hmi_all_hires/utt2dur"

    # Number of samples per utterance
    utt2num_samples_read_file = "JASMIN/kaldi_procesed/nl_test_read_all_hires/utt2num_samples"
    utt2num_samples_hmi_file = "JASMIN/kaldi_procesed/nl_test_hmi_all_hires/utt2num_samples"

    # Audio file path
    wav_scp_read_file = "JASMIN/kaldi_procesed/nl_test_read_all_hires/wav.scp"
    wav_scp_hmi_file = "JASMIN/kaldi_procesed/nl_test_hmi_all_hires/wav.scp"
    
    ### Read data from the files

    text_read = read_csv_to_dict_transcript(text_read_file, key_column=0)
    text_hmi = read_csv_to_dict_transcript(text_hmi_file, key_column=0)

    utt2dur_read = read_csv_to_dict(utt2dur_read_file, key_column=0)
    utt2dur_hmi = read_csv_to_dict(utt2dur_hmi_file, key_column=0)

    utt2num_samples_read = read_csv_to_dict(utt2num_samples_read_file, key_column=0)
    utt2num_samples_hmi = read_csv_to_dict(utt2num_samples_hmi_file, key_column=0)

    wav_scp_read = read_csv_to_dict(wav_scp_read_file, key_column=0)
    wav_scp_hmi = read_csv_to_dict(wav_scp_hmi_file, key_column=0)

    ###### Read 
    print("Start reading read")
    dataset_read = read_csv_to_dict(utt2spk_read_file, key_column=0)
    metadata_read = read_meta_csv_to_dict(spk2meta_read_file, spk_id_column_idx=1, info_column_idx=2, delimiter=",")
    print("Finish reading read")
    
    match_index = 0  # Index where the speaker ID resides, in the data dict 

    # Add demographic group to data list for each speech file
    for utt_id, data_list in dataset_read.items():
        speaker_id_to_match = data_list[match_index].upper() 
        if speaker_id_to_match in metadata_read:
            data_list.append(metadata_read.get(speaker_id_to_match))  # Append the demographic group
            
    gender = read_meta_csv_to_dict(spk2gender_read_file, spk_id_column_idx=0, info_column_idx=1, delimiter=" ")
    
    for utt_id, data_list in dataset_read.items():
        speaker_id_to_match = data_list[match_index].upper() 
        if speaker_id_to_match in gender:
            data_list.append(gender.get(speaker_id_to_match))  # Append the gender group

    # Convert to entry with utterance id, speaker id, demographic group, and gender
    result_read = [
        {'ID': utt_id, 'spk_id': data_list[0], 'demog': data_list[1], 'gender': data_list[2]}
        for utt_id, data_list in dataset_read.items()
    ]

    for entry in result_read:
        utt_id = entry['ID']

        entry['speaking_style'] = "read"
        entry['duration'] = utt2dur_read.get(utt_id)
        entry['num_samples'] = utt2num_samples_read.get(utt_id)
        entry['filepath'] = wav_scp_read.get(utt_id)
        entry['transcription'] = text_read.get(utt_id)
        
    print("Read data processed")
    ###### HMI 
    print("Start reading hmi")
    dataset_hmi = read_csv_to_dict(utt2spk_hmi_file, key_column=0)
    metadata_hmi = read_meta_csv_to_dict(spk2meta_hmi_file, spk_id_column_idx=1, info_column_idx=2, delimiter=",")
    print("Finish reading hmi")
    
    match_index = 0  # Index where the speaker ID resides, in the data dict 

    # Add demographic group to data list for each speech file
    for utt_id, data_list in dataset_hmi.items():
        speaker_id_to_match = data_list[match_index].upper() 
        if speaker_id_to_match in metadata_hmi:
            data_list.append(metadata_hmi.get(speaker_id_to_match))  # Append the demographic group
            
    gender_hmi = read_meta_csv_to_dict(spk2gender_hmi_file, spk_id_column_idx=0, info_column_idx=1, delimiter=" ")
    
    for utt_id, data_list in dataset_hmi.items():
        speaker_id_to_match = data_list[match_index].upper() 
        if speaker_id_to_match in gender_hmi:
            data_list.append(gender_hmi.get(speaker_id_to_match))  # Append the gender group

    # Convert to entry with utterance id, speaker id, demographic group, and gender
    result_hmi = [
        {'ID': utt_id, 'spk_id': data_list[0], 'demog': data_list[1], 'gender': data_list[2]}
        for utt_id, data_list in dataset_hmi.items()
    ]

    for entry in result_hmi:
        utt_id = entry['ID']

        entry['speaking_style'] = "hmi"
        entry['duration'] = utt2dur_hmi.get(utt_id)
        entry['num_samples'] = utt2num_samples_hmi.get(utt_id)
        entry['filepath'] = wav_scp_hmi.get(utt_id)
        entry['transcription'] = text_hmi.get(utt_id)
        
    print("HMI data processed")
    
    result = result_read + result_hmi
    
    print("Concatenated results")
    
    # Save to file
    output_file = f"{dataset_name}_formatted.{output_format}"
    if output_format == "csv":
        keys = result[0].keys()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result)
    elif output_format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
    
    print(f"Dataset formatted and saved as {output_file}")
    
def clean_dataset(dataset_name):
    df = pd.read_csv(f"{dataset_name}_formatted.csv")
    

    # Cleaning transcriptions
    # The following Regular Expressions were used to check for punctuation: 
    # .wav'\],.*[A-Z] -> None
    # .wav'\],.*[0-9] -> !!YES!! (mostly "mp3" but also 6 for 6 o' clock)
    # dashes are used in some places, but it seems to mean something (like a pause) so we leave it in

    digit_to_dutch = {
        '0': 'nul',
        '1': 'een',
        '2': 'twee',
        '3': 'drie',
        '4': 'vier',
        '5': 'vijf',
        '6': 'zes',
        '7': 'zeven',
        '8': 'acht',
        '9': 'negen'
    }
    
    def replace_digits_with_dutch(text):
        parts = []
        for char in text:
            parts.append(digit_to_dutch[char] if char.isdigit() else char)
        return ''.join(parts)

    df['transcription'] = df['transcription'].apply(replace_digits_with_dutch)

    # Remove rows where '[LAUGH]' is present in the transcription (NOT ANYMORE)
    # df = df[~df['transcription'].str.contains('\[LAUGH\]', regex=True, na=False)]

    # Keep only rows where demog is either 'NnT' or 'DT'
    df = df[df['demog'].isin(['NnT', 'DT'])]
    
    df["duration"] = df["duration"].str.strip("[]'").astype(float)
    df["num_samples"] = df["num_samples"].str.strip("[]'").astype(int)
    df["filepath"] = df["filepath"].str.strip("[]'")
    
    df.to_csv(f"{dataset_name}_cleaned.csv", index=False)
    
def split_dataset(dataset_name):
    df = pd.read_csv(f"{dataset_name}_cleaned.csv")
    
    # Split the dataset while maintaining balance
    # spk_groups = df.groupby('spk_id')
    # stratify_columns = spk_groups[['gender', 'demog']]
    # train_df, temp_df = train_test_split(spk_groups, test_size=0.2, stratify=stratify_columns, random_state=42)
    # val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[['gender', 'demog']], random_state=42)

    # Split at speaker level while maintaining balance
    def stratified_speaker_split(df, test_size=0.1, val_size=0.1, random_state=42):
        speaker_df = df.groupby('spk_id').first().reset_index()  # Get unique speakers with one row each
        train_speakers, temp_speakers = train_test_split(speaker_df, test_size=test_size + val_size, 
                                                        stratify=speaker_df[['demog', 'gender']],
                                                        random_state=random_state)
        val_speakers, test_speakers = train_test_split(temp_speakers, test_size=val_size / (test_size + val_size),
                                                    stratify=temp_speakers[['demog', 'gender']],
                                                    random_state=random_state)
        return train_speakers['spk_id'], val_speakers['spk_id'], test_speakers['spk_id']

    train_speakers, val_speakers, test_speakers = stratified_speaker_split(df)

    # Assign utterances based on speaker splits
    train_df = df[df['spk_id'].isin(train_speakers)]
    val_df = df[df['spk_id'].isin(val_speakers)]
    test_df = df[df['spk_id'].isin(test_speakers)]
    
    counts_train = train_df.groupby(['demog', 'gender', 'speaking_style']).size()
    print("Train counts:")
    print(counts_train)
    
    counts_val = val_df.groupby(['demog', 'gender', 'speaking_style']).size()
    print("Val counts:")
    print(counts_val)
    
    counts_test = test_df.groupby(['demog', 'gender', 'speaking_style']).size()
    print("Test counts:")
    print(counts_test)
    
    print("Check if any spk_id's overlap between the splits:")
    print(set(train_df['spk_id']).intersection(set(val_df['spk_id'])))
    print(set(train_df['spk_id']).intersection(set(test_df['spk_id'])))
    print(set(val_df['spk_id']).intersection(set(test_df['spk_id'])))
    
    # Save the splits to CSV
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset_name", help="Name of the dataset")
    # parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output file format")
    # args = parser.parse_args()
    
    prepare_dataset("Jasmin", "csv")
    clean_dataset("Jasmin")
    split_dataset("Jasmin")
