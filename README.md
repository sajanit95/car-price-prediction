# Sri Lankan Used Car Price Prediction – Fair Price Checker

Predicts a fair asking price for a used car in Sri Lanka and flags listings that look over- or under-priced.
Trained on 18,938 car listings from a Sri Lankan classifieds site (Feb–Aug 2021).

**Live app:** https://car-price-prediction-2rlofumejge5u7gehmsev8.streamlit.app/

## Results

| | |
|---|---|
| Main model (new algorithm) | LightGBM (tuned) |
| Test MAE | Rs 866,035 |
| Test R² | 0.938 |
| Median error | 8.8% |

**New algorithm: LightGBM** (not covered in lectures). Models covered in lectures – Linear Regression, SVM (SVR), Decision Tree, Random Forest, XGBoost – plus Ridge and KNN are used as benchmarks. LightGBM ranked 1st after tuning. Explained with SHAP, LIME and partial dependence plots. Full analysis is in `Car_Price_Prediction_SriLanka.ipynb`.

## Project structure

```
├── Car_Price_Prediction_SriLanka.ipynb   # EDA, preprocessing, models, tuning, SHAP, LIME
├── data/dataset_vehicles.csv             # raw data
├── src/
│   ├── features.py      # cleaning + feature engineering
│   ├── validate.py      # checks the raw CSV before training
│   ├── models.py        # preprocessing pipeline, models, tuning search spaces
│   ├── train.py         # compare -> tune LightGBM + best benchmark -> evaluate -> save
│   ├── explain.py       # SHAP importance + partial dependence plots
│   └── inference.py     # single and batch prediction
├── run_pipeline.py      # runs the whole pipeline with one command
├── anonymise_data.py    # removes personal data from the raw CSV
├── config.yaml          # pipeline settings
├── config_ci.yaml       # quick settings for GitHub Actions
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
├── artifacts/           # saved model + metadata (used by the app)
├── app.py               # Streamlit app
└── requirements.txt
```

## 1. Setup

Requires Python 3.12.

```
git clone https://github.com/sajanit95/269168B_AML_E2E_car_price_predictor.git
cd 269168B_AML_E2E_car_price_predictor
python -m venv venv
```

Activate the environment:

- Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
  (if scripts are blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once)
- Mac/Linux: `source venv/bin/activate`

Install packages:

```
pip install -r requirements.txt
```

## 2. Run the pipeline

```
python run_pipeline.py
```

Takes about 12 minutes. Steps:

| Step | What happens |
|---|---|
| 1. Validate | Checks required columns exist and Price / Mileage / Capacity can be read as numbers. Stops if the data is broken. |
| 2. Clean | Converts text to numbers, removes duplicates and impossible prices, fixes engine size and mileage errors, creates features (age, brand_model, district, km_per_year, unregistered). |
| 3. Train & select | Compares all models with 5-fold CV, tunes LightGBM (main model) and the best benchmark with random search, keeps tuned settings only if they beat the defaults. |
| 4. Save | Saves the model and metadata to `artifacts/` with a version number. |
| 5. Explain | Saves SHAP importance and partial dependence plots to `artifacts/explain/`. |

Output files in `artifacts/`:

- `car_price_model.joblib` – full pipeline (preprocessing + model)
- `metadata.json` – version, test metrics, best settings, fair price range, app dropdown options
- `model_comparison.csv` – CV scores for all models
- `lime_background.csv` – sample data used by LIME in the app

To use different settings, edit `config.yaml` or pass another file:

```
python run_pipeline.py --config my_config.yaml
```

| Setting | Default | Meaning |
|---|---|---|
| `candidates` | Ridge, KNN, DecisionTree, SVR, XGBoost, LightGBM | models to compare |
| `cv_folds` | 5 | folds for comparing models |
| `main_model` | LightGBM | the new algorithm (always selected) |
| `tune_top_k` | 1 | number of benchmarks tuned for comparison |
| `tune_iterations` | 12 | random settings tried per model |
| `use_mlflow` | false | log runs to MLflow |

Random Forest is left out of the pipeline because it takes ~10 min to tune and was never the best model in the notebook.

### Retraining with new data

Replace `data/dataset_vehicles.csv` (same columns) and run `python run_pipeline.py` again. A new version is saved and the comparison table shows if it's better.

### Experiment tracking (optional)

```
pip install mlflow
```

Set `use_mlflow: true` in `config.yaml`, run the pipeline, then view runs with `mlflow ui`.

## 3. Predictions

Single car:

```python
from src.inference import predict

car = {"Brand": "Toyota", "Model": "Aqua", "Year": 2014, "mileage_km": 90000,
       "capacity_cc": 1500, "Fuel": "Hybrid", "Transmission": "Automatic",
       "Body": "Hatchback", "Condition": "Used", "Location": "Colombo"}
print(predict([car]))
```

Batch (CSV with the same columns, optional `listed_price` column for a verdict):

```
python -c "from src.inference import predict_batch; print(predict_batch('artifacts/sample_batch.csv', 'artifacts/predictions.csv'))"
```

## 4. Run the app locally

```
python -m streamlit run app.py
```

Opens at http://localhost:8501. Enter car details → estimated price + typical range → optional listed price check → LIME chart showing what pushed the price up or down.

## 5. Tests and CI

```
python -m pytest -q tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push:
1. installs requirements
2. runs the tests
3. runs a quick version of the pipeline (`config_ci.yaml`)

Check the **Actions** tab on GitHub for a green tick.

## 6. Deploy the app (Streamlit Community Cloud)

1. Make sure `artifacts/` is pushed to GitHub (the app needs the saved model)
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click **Create app**
4. Select the repo, branch `main`, main file `app.py`
5. **Advanced settings** → Python version **3.12**
6. Click **Deploy** (first build takes a few minutes)

To update the deployed app after retraining, push the new `artifacts/` folder – Streamlit redeploys automatically.

## Limitations

- Data is from 2021 (vehicle import restrictions) – prices have changed since
- Asking prices, not final sale prices
- Bigger errors for rare vehicles (EVs, vintage cars) and cars over Rs 25M

## Ethical data use

The raw ads contained seller names, ad URLs and phone numbers (184 descriptions). `anonymise_data.py` hashes names and URLs and masks phone numbers; none of these are used as model features. A test checks no phone numbers remain.

## Data source

`dataset_vehicles.csv` from github.com/DavidR0/Car-Price-Prediction (originally from Kaggle). Only the raw data is reused.
