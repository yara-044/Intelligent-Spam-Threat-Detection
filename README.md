# Intelligent Spam & Threat Detection System

A black-and-yellow cybersecurity interface based on the supplied project logo, connected to the original training and prediction scripts.

## Included files
- `app.py`: Streamlit interface and project dashboard.
- `project_logo.png`: the supplied project identity image.
- `train_and_evaluate.py`: original uploaded training script (copied unchanged).
- `predict_message.py`: original uploaded prediction script (copied unchanged).
- `spam_ham_dataset.csv`: supplied dataset.
- `requirements.txt`: dependencies.

## Run
Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first launch, click **TRAIN PROJECT MODELS**. The app executes the original `train_and_evaluate.py` with the supplied CSV, which trains and compares Naive Bayes, Logistic Regression, Linear SVM and Random Forest, then saves the best model, metrics and charts in `model_output/`.

The Message Scanner uses the saved best model and the same `model.predict([message])` operation as the original `predict_message.py`.

## Notes
- The uploaded training and prediction scripts are included unchanged.
- No probability/confidence score is displayed because the supplied prediction script returns a class label only.
- The dataset has HAM/SPAM labels only; the app does not independently verify malware, URLs, sender identity, or determine a specific threat subtype.
