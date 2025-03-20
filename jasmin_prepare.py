import csv
import json
import argparse

# Define function to read CSV files into dictionaries
def read_csv_to_dict(file_path, key_column, delimiter=" "):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) > 1:
                data[row[0]] = row[1:] if len(row) > 2 else row[1]
    return data

#This has to be different for the files from Kayleigh
def read_csv_to_dict(file_path, key_column, delimiter=" "):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) > 1:
                data[row[0]] = row[2:] if len(row) > 2 else row[2]
    return data
# def filter_and_merge(read_csv, spontaneous_csv, output_csv):
#     selected_groups = {"NnT", "DT"}  # Groups to keep
    
#     def read_and_filter(file_path, speech_type):
#         filtered_data = []
#         with open(file_path, "r", encoding="utf-8") as f:
#             reader = csv.reader(f, delimiter="\t")
#             header = next(reader)  # Read header
#             for row in reader:
#                 if row[2] in selected_groups:  # Check group
#                     filtered_data.append(row + [speech_type])
#         return filtered_data
    
#     # Process files
#     read_data = read_and_filter(read_csv, "read")
#     spontaneous_data = read_and_filter(spontaneous_csv, "spontaneous")
    
#     # Merge data
#     merged_data = read_data + spontaneous_data
    
#     # Write to new CSV
#     with open(output_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f, delimiter="\t")
#         writer.writerow(["Filename", "SPKR", "Group", "Speech_Type"])  # New header
#         writer.writerows(merged_data)
    
#     print(f"Filtered and merged CSV saved as {output_csv}")


def prepare_dataset(dataset_name, output_format="csv"):
    # File paths 
    spk2meta_read_file = "Rd_meta.csv"
    spk2meta_hmi_file = "HMI_meta.csv"
    spk2gender_read_file = "JASMIN/kaldi_processed/nl_test_read_all_hires/spk2gender"
    spk2gender_hmi_file = "JASMIN/kaldi_processed/nl_test_hmi_all_hires/spk2gender"
    spk2utt_read_file = "JASMIN/kaldi_processed/nl_test_read_all_hires/spk2utt"
    spk2utt_hmi_file = "JASMIN/kaldi_processed/nl_test_hmi_all_hires/spk2utt"
    text_read_file = "JASMIN/kaldi_processed/nl_test_read_all_hires/text"
    text_hmi_file = "JASMIN/kaldi_processed/nl_test_hmi_all_hires/text"
    utt2dur_read_file = "JASMIN/kaldi_processed/nl_test_read_all_hires/utt2dur"
    utt2dur_hmi_file = "JASMIN/kaldi_processed/nl_test_hmi_all_hires/utt2dur"
    utt2num_samples_read_file = "JASMIN/kaldi_processed/nl_test_read_all_hires/utt2num_samples"
    utt2num_samples_hmi_file = "JASMIN/kaldi_processed/nl_test_hmi_all_hires/utt2num_samples"
    wav_scp_read_file = "JASMIN/kaldi_processed/nl_test_read_all_hires/wav.scp"
    wav_scp_hmi_file = "JASMIN/kaldi_processed/nl_test_hmi_all_hires/wav.scp"
    
    # Read data from the files
    spk2meta_read = read_csv_to_dict(spk2meta_read_file, key_column=1)
    spk2meta_hmi = read_csv_to_dict(spk2meta_hmi_file, key_column=0)
    spk2gender_read = read_csv_to_dict(spk2gender_read_file, key_column=0)
    spk2gender_hmi = read_csv_to_dict(spk2gender_hmi_file, key_column=0)
    text_read = read_csv_to_dict(text_read_file, key_column=0)
    text_hmi = read_csv_to_dict(text_hmi_file, key_column=0)
    utt2dur_read = read_csv_to_dict(utt2dur_read_file, key_column=0)
    utt2dur_hmi = read_csv_to_dict(utt2dur_hmi_file, key_column=0)
    utt2num_samples_read = read_csv_to_dict(utt2num_samples_read_file, key_column=0)
    utt2num_samples_hmi = read_csv_to_dict(utt2num_samples_hmi_file, key_column=0)
    wav_scp_read = read_csv_to_dict(wav_scp_read_file, key_column=0)
    wav_scp_hmi = read_csv_to_dict(wav_scp_hmi_file, key_column=0)
    
    # Filter utterance IDs based on DT and NnT groups
    valid_speakers = {spk_id.upper() for spk_id, group in spk2meta_read.items() if group in ["DT", "NnT"]} | \
                        {spk_id.upper() for spk_id, group in spk2meta_hmi.items() if group in ["DT", "NnT"]}
    

    # Prepare dataset entries
    dataset = []
    for utt_id, group in valid_utterances.items():
        speech_type = "read" if utt_id in spk2meta_read else "hmi"
        entry = {
            "utterance_id": utt_id,
            "speaker_id": spk2meta_read.get(utt_id, spk2meta_hmi.get(utt_id, "")),
            "gender": spk2gender_read.get(utt_id, spk2gender_hmi.get(utt_id, "")),
            "duration": utt2dur_read.get(utt_id, utt2dur_hmi.get(utt_id, "")),
            "num_samples": utt2num_samples_read.get(utt_id, utt2num_samples_hmi.get(utt_id, "")),
            "audio_path": wav_scp_read.get(utt_id, wav_scp_hmi.get(utt_id, "")),
            "transcription": text_read.get(utt_id, text_hmi.get(utt_id, "")),
            "speech_type": speech_type,
            "group": group
        }
        dataset.append(entry)
    
    # Save to file
    output_file = f"{dataset_name}_formatted.{output_format}"
    if output_format == "csv":
        keys = dataset[0].keys()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dataset)
    elif output_format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
    
    print(f"Dataset formatted and saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output file format")
    args = parser.parse_args()
    
    prepare_dataset(args.dataset_name, args.format)
