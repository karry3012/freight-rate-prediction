# Freight Rate Prediction

Predicts the `posted_rate` of freight loads. Trained on Jan-Oct 2025 loads, used to
predict 12,000 Nov-Dec 2025 loads plus a fixed December 2025 lane (Lexington -> Fort Wayne).

## Setup & run

```bash
python -m pip install -r requirements.txt

# place the assessment files in data/ :
#   data/train_test.csv
#   data/validation.csv
#   data/validation_predictions_template.csv
#   data/december_chart_inputs.csv

python freight_rate_model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`freight_rate_model.py` prints the cross-validation results, writes
`validation_predictions.csv` (`load_id,predicted_rate`), fills the `predicted_rate`
column of `data/december_chart_inputs.csv`, and saves `validation_report.csv`.
`score.py` validates both files and creates `scorer_results/candidate_december.png`.
Python 3.10+ recommended.

## Repository contents

| Path | Description |
|---|---|
| `freight_rate_model.py` | Full pipeline: cleaning, features, validation, training, prediction |
| `score.py` | Provided scorer (unchanged) |
| `requirements.txt` | Dependencies |
| `validation_predictions.csv` | Final predictions for the 12,000 validation loads |
| `results/candidate_december.png` | December chart produced by `score.py` |
| `results/validation_report.csv` | Per-fold CV metrics |
| `results/december_chart_inputs_filled.csv` | Completed December predictions |

## Approach

**Validation / split.** Training data is Jan-Oct 2025 while the target loads are Nov-Dec 2025,
so the task is forecasting into the future. A random split would leak market conditions and
overstate accuracy. I use forward-chaining (expanding-window) CV: train on all months before
month *M*, test on month *M*, for M = June..October. A 2-months-ahead hold-out
(train Jan-Aug, test Sep-Oct) gives a stricter estimate.

**Data quality.**
- 292 negative weights (plausible magnitudes) treated as sign errors -> absolute value, flagged.
- 300 missing weights -> equipment median (train only); 374 missing `market_index` -> same-date mean. Missing flags kept.
- ~1.5% extreme rate outliers -> `log(rate)` target with Huber loss. Explicit trimming gave no gain.
- 8 cities in validation never appear in training -> no city/lane features; geography enters via lat/lon, distance and haversine-derived features.
- `market_index` drifts strongly by month -> used directly; month/day-of-year deliberately excluded because validation months are unseen.

**Model.** LightGBM regressor on `log(posted_rate)` (Huber loss), averaged over 5 seeds. Features:
distance (raw, log, inverse), geodesic distance and circuity, weight, equipment, coordinates,
lat/lon differences, `market_index`, `quote_signal`, day-of-week/month-day, missing flags.
Ridge regression is the baseline.

**Results (forward-chaining CV, mean of 5 folds).**

| Model | MAE | MAPE |
|---|---|---|
| Ridge baseline | $152.69 | 6.43% |
| LightGBM (Huber) | $126.49 | 5.32% |

2-months-ahead hold-out: MAE ~ $148, MAPE ~ 6.3%.

**December chart.** The December input rows contain only 7 columns. The model's other inputs
(`market_index`, `quote_signal`, coordinates) are filled from `validation.csv`: the mean of each
signal over that date's loads, and the known city coordinates. Predictions are therefore
fairly flat (~$800-$820) because only the date changes.
