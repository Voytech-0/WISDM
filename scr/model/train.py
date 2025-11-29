from preprocessing.preprocess import preprocessing_pipeline_for_wisdm_raw_data, generate_splits
from configs.configs import ModelConfig, GeneralConfig, DataConfig


def main():
    data = preprocessing_pipeline_for_wisdm_raw_data(DataConfig.data_path_prefix)
    print(data.collect_schema())
    splits_dictionary = generate_splits(data)  # {train: X, y}, {validation: X, y}, {test: X, y}


if __name__ == "__main__":
    main()
