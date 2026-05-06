# Indian Food Classifier

A small project that trains a MobileNetV2-based image classifier to recognize Indian dishes and provides a Streamlit app for inference.

## Data set collected from google images
https://drive.google.com/file/d/16fJVBFpF-SceLCmkWeCD_ni71RLpq7Dl/view?usp=sharing

## Contents

- `a3-smai.ipynb` — training notebook (builds and saves `indian_food_classifier.h5`)
- `app.py` — Streamlit app for uploading images and showing predictions
- `indian_food_classifier.h5` — saved trained model (expected in repo root for the app)
- `gimages_ds/` — training dataset (one folder per class)
- `Food Data/` — additional food image folders included in the workspace

## Dataset

The notebook expects `gimages_ds/` to contain one directory per class. Current classes in this workspace:

- bhature
- chapati
- daal_baati_churma
- lachha_paratha
- naan
- palak_roti
- ragi_roti
- rumali_rotti

If you retrain the model, make sure the class ordering used during training matches the label order used in `app.py`.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

## Training the Model

Open `a3-smai.ipynb` and run the cells in order. The notebook:

1. Loads images from `gimages_ds/`
2. Splits data into training and validation sets
3. Fine-tunes a MobileNetV2 base model
4. Saves the trained model as `indian_food_classifier.h5`

## Running the App

Place `indian_food_classifier.h5` in the repository root (or update `app.py` to point to its location), then run:

```bash
streamlit run app.py
```

Upload an image in the Streamlit UI to see the predicted dish, estimated calories, and common allergens.

## Notes

- The app expects the model file to be present in the project root by default.
- If you retrain with different classes, update the label mapping in `app.py` to match the model output order.

---

If you'd like, I can also add a short example screenshot, badges, or a `LICENSE` file next.
