import speechbrain as sb
import pandas as pd

def split_into_demog_tests():
    data_folder = "/scratch/mfron/IST-LAB/data"

    test_data = pd.read_csv(f"{data_folder}/test.csv")

    # Define possible values
    demog_values = ["NnT", "DT"]
    speaking_styles = ["read", "hmi"]

    # Generate and save each combination
    for demog in demog_values:
        for style in speaking_styles:
            subset = test_data[(test_data["demog"] == demog) & (test_data["speaking_style"] == style)]
            subset = subset.sort_values(by="duration")  # Sort by duration
            subset.to_csv(f"{data_folder}/test_{demog}_{style}.csv", index=False)

# test_DT_read
# test_DT_hmi
# test_NnT_read
# test_NnT_hmi

if __name__ == "__main__":
    split_into_demog_tests()