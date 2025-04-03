import speechbrain as sb
import pandas as pd

def split_into_demog_tests():
    data_folder = "/scratch/mfron/IST-LAB/data"

    test_data = pd.read_csv(f"{data_folder}/test.csv")

    # Separate speakers into two groups based on "demog" column
    test_datasets = {
        "NnT": test_data[test_data["demog"] == "NnT"],
        "DT": test_data[test_data["demog"] == "DT"],
    }

    # Sort datasets by "duration"
    test_datasets["NnT"] = test_datasets["NnT"].sort_values(by="duration")
    test_datasets["DT"] = test_datasets["DT"].sort_values(by="duration")

    # Save to CSV
    test_datasets["NnT"].to_csv("data/test_NnT.csv", index=False)
    test_datasets["DT"].to_csv("data/test_DT.csv", index=False)

if __name__ == "__main__":
    split_into_demog_tests()