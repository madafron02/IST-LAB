import speechbrain as sb
import pandas as pd

def split_into_demog_tests(hparams):
    data_folder = hparams["data_folder"]

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams['test_csv'], replacements={"data_root": data_folder}
        )

    # Separate speakers into two groups based on "nativeness" column
    test_datasets = {
        "NnT": test_data.filtered(lambda x: x["demog"] == "NnT"),
        "DT": test_data.filtered(lambda x: x["demog"] == "DT"),
    }

    test_datasets["NnT"] = test_datasets["NnT"].filtered_sorted(sort_key="duration")
    test_datasets["DT"] = test_datasets["DT"].filtered_sorted(sort_key="duration")


    test_datasets["NnT"].to_csv("data/test_NnT.csv", index=False)
    test_datasets["DT"].to_csv("data/test_DT.csv", index=False)
