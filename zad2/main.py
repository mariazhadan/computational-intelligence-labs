import csv
import re
import pandas as pd
import numpy as np

SRC_PATH = "iris_big_with_errors.csv"
OUT_PATH = "iris_big_clean.csv"


def load_with_structure_fixes(path: str) -> tuple[pd.DataFrame, dict]:
    """Load CSV, fixing known structural errors (split decimals) and reporting issues."""
    rows = []
    field_counts = {}
    bad_rows = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            field_counts[len(row)] = field_counts.get(len(row), 0) + 1

            # Fix rows where a decimal was split into two fields: ['4','75',...]
            if len(row) == 6:
                row = [f"{row[0]}.{row[1]}"] + row[2:]

            if len(row) != 5:
                bad_rows.append((line_no, row))
                # keep row as-is to avoid data loss

            rows.append(row)

    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)

    info = {
        "field_counts": field_counts,
        "bad_rows": bad_rows,
    }
    return df, info


def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Clean numeric columns: fix commas, coerce, range-check, fill with median."""
    stats = {}

    for col in cols:
        raw = df[col].astype(str).str.strip()
        raw = raw.str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(raw, errors="coerce")

        # Range check
        out_of_range = (numeric <= 0) | (numeric >= 10)
        out_of_range_count = int(out_of_range.sum())
        numeric[out_of_range] = np.nan

        missing_count = int(numeric.isna().sum())
        median = float(numeric.median())
        numeric = numeric.fillna(median)

        df[col] = numeric
        stats[col] = {
            "missing_or_invalid": missing_count,
            "out_of_range": out_of_range_count,
            "median_used": median,
        }

    return df, stats


def normalize_species(value: str) -> str:
    s = str(value).lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[\.?]", "", s)
    s = re.sub(r"\biris\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_species_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, dict]:
    valid = {"setosa", "versicolor", "virginica"}

    raw_unique = sorted(set(df[col].astype(str)))
    normalized = df[col].apply(normalize_species)
    invalid_before = sorted(set(df[col][~normalized.isin(valid)].astype(str)))

    typo_map = {
        "versicolr": "versicolor",
        "versi color": "versicolor",
        "setosa ": "setosa",
        "virginica ": "virginica",
    }

    normalized = normalized.replace(typo_map)

    # Replace remaining invalids with the mode of valid values
    if (normalized.isin(valid)).any():
        mode = normalized[normalized.isin(valid)].mode().iloc[0]
    else:
        mode = "setosa"

    normalized = normalized.where(normalized.isin(valid), mode)

    df[col] = normalized
    invalid_after = sorted(set(df[col][~df[col].isin(valid)]))

    stats = {
        "raw_unique": raw_unique,
        "invalid_before": invalid_before,
        "invalid_after": invalid_after,
        "mode_used": mode,
    }
    return df, stats


def main() -> None:
    df, info = load_with_structure_fixes(SRC_PATH)

    print("STRUCTURE CHECK")
    print(f"Field count distribution: {info['field_counts']}")
    if info["bad_rows"]:
        print(f"Rows with unexpected field counts: {len(info['bad_rows'])}")
        print(f"First 3 examples: {info['bad_rows'][:3]}")
    else:
        print("No structural errors after fixing split decimals.")

    # Missing/empty statistics before cleaning
    print("\nMISSING/EMPTY BEFORE CLEANING")
    empty_counts = (df == "").sum()
    print(empty_counts.to_string())

    numeric_cols = df.columns[:-1].tolist()
    species_col = df.columns[-1]

    # Basic stats with errors
    print("\nBASIC STATS WITH ERRORS (NUMERIC COERCE)")
    numeric_coerced = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    print(numeric_coerced.describe().to_string())

    # Clean numeric columns
    df, num_stats = clean_numeric_columns(df, numeric_cols)

    print("\nNUMERIC CLEANING SUMMARY")
    for col, s in num_stats.items():
        print(
            f"{col}: missing/invalid={s['missing_or_invalid']}, "
            f"out_of_range={s['out_of_range']}, median_used={s['median_used']:.4f}"
        )

    # Clean species column
    df, species_stats = clean_species_column(df, species_col)

    print("\nSPECIES CLEANING SUMMARY")
    print(f"Raw unique values: {species_stats['raw_unique']}")
    print(f"Invalid values before: {species_stats['invalid_before']}")
    print(f"Invalid values after: {species_stats['invalid_after']}")
    print(f"Mode used for remaining invalids: {species_stats['mode_used']}")

    # Save cleaned dataset
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {OUT_PATH}")

if __name__ == "__main__":
    main()
