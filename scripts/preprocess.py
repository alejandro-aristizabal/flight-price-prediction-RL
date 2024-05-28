import pandas as pd
import argparse


def load_and_preprocess_data(file_path):
    # Load data first 1_000_000_000 rows
    data = pd.read_csv(file_path, nrows=1_000_000_000)

    # Handle missing values
    data.fillna(method="ffill", inplace=True)

    # Create new features (example: days until flight)
    data["searchDate"] = pd.to_datetime(data["searchDate"])
    data["flightDate"] = pd.to_datetime(data["flightDate"])
    data["days_until_flight"] = (data["flightDate"] - data["searchDate"]).dt.days

    # Encode categorical variables
    data = pd.get_dummies(
        data,
        columns=["startingAirport", "destinationAirport", "segmentsAirlineName"],
        drop_first=True,
    )

    # Select relevant columns
    features = [
        "days_until_flight",
        "baseFare",
        "totalFare",
        "travelDuration",
        "isBasicEconomy",
        "isRefundable",
        "isNonStop",
    ]
    features += [
        col
        for col in data.columns
        if col.startswith("startingAirport_")
        or col.startswith("destinationAirport_")
        or col.startswith("segmentsAirlineName_")
    ]

    return data[features]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess flight prices data.")
    parser.add_argument("--file_path", type=str, help="Path to the raw data CSV file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save the preprocessed data CSV file"
    )
    args = parser.parse_args()

    data = load_and_preprocess_data(args.file_path)
    data.to_csv(args.output_path, index=False)
