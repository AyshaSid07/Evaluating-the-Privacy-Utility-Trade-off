import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "folktables_public_coverage_RAW.csv"
TRAIN_FILE = "folktables_public_coverage_train.csv"
TEST_FILE = "folktables_public_coverage_test.csv"
TARGET_COLUMN = "PUBCOV"
RANDOM_SEED = 123

def main():
    df = pd.read_csv(INPUT_FILE)

    df_train, df_test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=RANDOM_SEED, 
        stratify=df[TARGET_COLUMN]
    )

    df_train.to_csv(TRAIN_FILE, index=False)
    df_test.to_csv(TEST_FILE, index=False)

if __name__ == "__main__":
    main()