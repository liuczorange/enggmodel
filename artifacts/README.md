# Complete prediction exports

`predictions.csv.gz` and `predictions_wide.csv.gz` contain the complete saved final-test forecasts, not a sample. They are compressed for GitHub file-size limits. Run `python restore_predictions.py` from the repository root to restore them into `multihorizon_outputs/`.

The restore program verifies the uncompressed SHA-256 hashes recorded in `multihorizon_outputs/verification_summary.json`. It refuses to overwrite any existing file that differs from the published result.

Source attribution and transformations are described in the root README. Restoring predictions does not train models or open the source test split.
