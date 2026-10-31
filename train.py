"""Run once:  python train.py AAFAQ_Dataset.csv
Uses the ORIGINAL dataset (not the stemmed one) and saves models to ./artifacts"""
import sys, joblib, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from common import light_clean, normalize_for_ml

csv = sys.argv[1] if len(sys.argv) > 1 else "AAFAQ_Dataset.csv"
out = Path("artifacts"); out.mkdir(exist_ok=True)

df = pd.read_csv(csv).dropna(subset=["QuestionText", "Answer", "Category"])
df = df.drop_duplicates(subset=["QuestionText"]).reset_index(drop=True)
df["q_raw"] = df["QuestionText"].map(light_clean)
df["a_raw"] = df["Answer"].map(light_clean)
df["q_norm"] = df["QuestionText"].map(normalize_for_ml)
print(df["Category"].value_counts(), "\n")

# ---- Classifier: trained on the QUESTION only (that's all the user types in the app)
def build():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return make_pipeline(feats, LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))

vc = df["Category"].value_counts()
ok = df[df["Category"].isin(vc[vc >= 2].index)]
Xtr, Xte, ytr, yte = train_test_split(ok["q_norm"], ok["Category"], test_size=0.2,
                                      random_state=42, stratify=ok["Category"])
clf = build().fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"Accuracy: {accuracy_score(yte, pred):.3f} | Macro-F1: {f1_score(yte, pred, average='macro'):.3f}")
print(classification_report(yte, pred, zero_division=0))

clf = build().fit(df["q_norm"], df["Category"])      # refit on everything for the app
joblib.dump(clf, out / "classifier.joblib")

# ---- QA retrieval index over the original questions
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
mat = vec.fit_transform(df["q_norm"])
joblib.dump({"vec": vec, "mat": mat,
             "q": df["q_raw"].tolist(), "a": df["a_raw"].tolist(),
             "cat": df["Category"].tolist()}, out / "qa_index.joblib")
print("Saved to", out.resolve())
