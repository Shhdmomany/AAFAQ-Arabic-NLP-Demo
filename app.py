"""Run:  python app.py   (after python train.py)"""
import joblib, numpy as np, gradio as gr
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from common import light_clean, normalize_for_ml

QA_THRESHOLD = 0.35          # below this the match is not trusted
clf = joblib.load("artifacts/classifier.joblib")
qa = joblib.load("artifacts/qa_index.joblib")

@lru_cache(maxsize=1)
def _mt():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "Helsinki-NLP/opus-mt-ar-en"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    return tok, model, torch

def translate(text: str) -> str:
    text = light_clean(text)           # raw sentence, NOT stemmed / stop-word-stripped
    if not text:
        return ""
    tok, model, torch = _mt()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**enc, num_beams=5, max_length=256, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)

def do_translate(text):
    return translate(text)

def do_classify(text):
    if not light_clean(text):
        return {}
    probs = clf.predict_proba([normalize_for_ml(text)])[0]
    return {c: float(p) for c, p in zip(clf.classes_, probs)}

def do_qa(question, show_english):
    if not light_clean(question):
        return "", "", "", ""
    sims = cosine_similarity(qa["vec"].transform([normalize_for_ml(question)]), qa["mat"]).ravel()
    top = np.argsort(-sims)[:3]
    best = top[0]
    similar = "\n".join(f"{sims[i]:.2f}  |  {qa['q'][i]}" for i in top)
    if sims[best] < QA_THRESHOLD:
        return "No confident match in the dataset for this question.", "", f"{sims[best]:.2f}", similar
    ans = qa["a"][best]
    if show_english:
        ans += "\n\n[EN] " + translate(ans)
    return ans, qa["cat"][best], f"{sims[best]:.2f}", similar

with gr.Blocks(title="AAFAQ Arabic NLP") as demo:
    gr.Markdown("# AAFAQ Arabic NLP demo\nTranslation, question classification and question answering.")
    with gr.Tab("Translate (AR→EN)"):
        i = gr.Textbox(label="Arabic text", rtl=True, lines=3)
        o = gr.Textbox(label="English translation")
        gr.Button("Translate").click(do_translate, i, o)
        gr.Examples([["أيفرز البنكرياس الأنسولين؟"], ["أليس القطن عماد الثروة في مصر؟"]], i)
    with gr.Tab("Classify"):
        i2 = gr.Textbox(label="Arabic question", rtl=True)
        o2 = gr.Label(label="Category", num_top_classes=5)
        gr.Button("Classify").click(do_classify, i2, o2)
    with gr.Tab("Question answering"):
        i3 = gr.Textbox(label="Arabic question", rtl=True)
        en = gr.Checkbox(label="Also show English translation of the answer")
        a = gr.Textbox(label="Answer", rtl=True, lines=3)
        c = gr.Textbox(label="Category")
        s = gr.Textbox(label="Match score")
        sim = gr.Textbox(label="Closest dataset questions", lines=3, rtl=True)
        gr.Button("Answer").click(do_qa, [i3, en], [a, c, s, sim])

demo.launch()
