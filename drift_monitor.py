
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset


def run_drift_detection(reference_df, current_df):

    # Convert Pandas DataFrames to Evidently datasets
    reference_data = Dataset.from_pandas(
        reference_df
    )

    current_data = Dataset.from_pandas(
        current_df
    )

    # Configure Evidently drift evaluation
    report = Report([
        DataDriftPreset()
    ])

    # Compare current data against reference
    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    return snapshot
