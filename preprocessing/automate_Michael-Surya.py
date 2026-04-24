from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)

    # Clean missing values first so downstream operations are stable.
    df = df.dropna().copy()

    if "ocean_proximity" in df.columns:
        encoder = LabelEncoder()
        df["ocean_proximity"] = encoder.fit_transform(df["ocean_proximity"].astype(str))

    if "median_house_value" in df.columns:
        max_value = df["median_house_value"].max()
        df = df[df["median_house_value"] != max_value]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "housing_raw.csv"
    output_path = repo_root / "preprocessing" / "housing_preprocessing.csv"
    preprocess_dataset(input_path, output_path)
    print(f"Preprocessing selesai. Output: {output_path}")


if __name__ == "__main__":
    main()