# AAFAQ Arabic NLP Demo

Gradio app for Arabic question data (AAFAQ): Arabic→English translation, question classification, and question answering by retrieval.

## Run
```
pip install -r requirements.txt
python train.py AAFAQ_Dataset.csv   # put the original dataset in this folder
python app.py
```
In Google Colab, change the last line of `app.py` to `demo.launch(share=True)`.

## How it works
- `common.py`: light Arabic cleaning (no stop-word removal or stemming, so translation gets full sentences)
- `train.py`: trains the classifier (question text only) and builds the QA index
- `app.py`: translation (Helsinki-NLP/opus-mt-ar-en), classification, QA

## Notes
The dataset is not included. Add `AAFAQ_Dataset.csv` yourself.
