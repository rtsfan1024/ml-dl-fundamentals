# encoding=utf-8

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from rake_nltk import Rake
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ======== 配置 ========
BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
DATA_CSV = str(BASE_DIR / "assets" / "data" / "cnn_dailymail_full.csv")       # CNN/DailyMail数据集路径
MAX_TRAIN_DOCS = 2000                     # 为了更快收敛，先用2000
TOPK_SENT = 2                             # 摘要输出句子数
KEYWORDS_K = 3                            # 关键词数量

# ======== 简易工具 ========
def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    # 优先用 NLTK
    try:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt_tab/english")
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        if len(sents) >= 2:
            return [s.strip() for s in sents if s.strip()]
    except Exception:
        pass
    # 兜底：正则按标点切分
    sents = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def topk_diverse(indices_scores: List[Tuple[int, float]], sent_vecs, k=3, sim_th=0.6):
    picked = []
    for idx, sc in indices_scores:
        if not picked:
            picked.append((idx, sc))
        else:
            ok = True
            for j, _ in picked:
                sim = cosine_similarity(sent_vecs[idx], sent_vecs[j])[0, 0]
                if sim >= sim_th:
                    ok = False
                    break
            if ok:
                picked.append((idx, sc))
        if len(picked) >= k:
            break
    return [i for i, _ in picked]

# ======== 训练数据制作（用gold摘要弱监督打标） ========
def make_training_pairs(df: pd.DataFrame, max_docs=2000):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    X_sent_text, X_doc_text, y = [], [], []
    used = 0
    for _, row in df.iterrows():
        art, hig = str(row['article']), str(row['highlights'])
        if not art or not hig:
            continue
        sents = split_sentences(art)
        if len(sents) < 2:
            continue
        labels = []
        for s in sents:
            r = scorer.score(hig, s)['rougeL'].fmeasure
            labels.append(1 if r >= 0.4 else 0)
        if sum(labels) == 0:
            # 选与 highlights 最相近的一句当正样本
            best_idx, best = 0, -1
            for i, s in enumerate(sents):
                r = scorer.score(hig, s)['rougeL'].fmeasure
                if r > best:
                    best, best_idx = r, i
            labels[best_idx] = 1
        X_sent_text.extend(sents)
        X_doc_text.extend([art] * len(sents))
        y.extend(labels)
        used += 1
        if used >= max_docs:
            break
    return X_sent_text, X_doc_text, y

# ======== 训练主流程 ========
def train_models(data_csv=DATA_CSV, max_docs=MAX_TRAIN_DOCS):
    df = pd.read_csv(data_csv)
    df = df[['article', 'highlights']].dropna().sample(frac=1.0, random_state=42).reset_index(drop=True)

    Xs, Xd, y = make_training_pairs(df, max_docs=max_docs)

    # 按“每文句数”聚合，供位置特征与相似度特征使用
    sent_per_doc, doc_texts = [], []
    i = 0
    for _, row in df.iterrows():
        art = str(row['article'])
        sents = split_sentences(art)
        if len(sents) < 2:
            continue
        sent_per_doc.append(len(sents))
        doc_texts.append(art)
        i += 1
        if i >= max_docs:
            break

    tfidf_sent = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), stop_words='english')

    # 用同一套词表：先在句子上拟合，再把整篇文档也映射到同一空间
    S_mat = tfidf_sent.fit_transform(Xs)
    D_mat = tfidf_sent.transform(doc_texts)

    # 为了兼容后面的summarize()，把tfidf_doc指向同一个向量器
    tfidf_doc = tfidf_sent

    # 句-文相似度（逐文计算）
    sims, start = [], 0
    for d_i, cnt in enumerate(sent_per_doc):
        end = start + cnt
        s_block = S_mat[start:end]
        d_vec   = D_mat[d_i]
        sims.append(cosine_similarity(s_block, d_vec))  # (cnt, 1)
        start = end
    sim_feat = np.vstack(sims)  # (n_sent, 1)

    # 位置特征
    hand = []
    for cnt in sent_per_doc:
        for s_i in range(cnt):
            pos = s_i / max(1, cnt - 1)
            pos2 = (0.5 - abs(pos - 0.2))
            hand.append([pos, pos2])
    hand = np.array(hand)

    # 特征拼接
    X_feat = np.hstack([sim_feat, hand])  # (n_sent, 3)

    # 分类器
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    clf.fit(X_feat, y)

    # 关键词TF-IDF（在highlights上拟合）
    tfidf_kw = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=10)
    tfidf_kw.fit(df['highlights'].astype(str).head(max_docs))

    artifacts = {
        'tfidf_sent': tfidf_sent,
        'tfidf_doc': tfidf_doc,
        'sent_per_doc': sent_per_doc,
        'clf': clf,
        'tfidf_kw': tfidf_kw
    }
    return artifacts

# ======== 推理：摘要（抽取式） ========
def summarize(text: str, artifacts, topk=TOPK_SENT) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    tfidf_sent = artifacts['tfidf_sent']; tfidf_doc = artifacts['tfidf_doc']; clf = artifacts['clf']

    S = tfidf_sent.transform(sents)
    d = tfidf_doc.transform([text])
    sim = cosine_similarity(S, d)

    # 位置特征
    hand = []
    for i in range(len(sents)):
        pos = i / max(1, len(sents) - 1)
        pos2 = (0.5 - abs(pos - 0.2))
        hand.append([pos, pos2])
    hand = np.array(hand)

    X = np.hstack([sim, hand])
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)

    # 多样性去重
    indices_scores = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)
    if len(sents) <= topk:
        picked_idx = list(range(len(sents)))
    else:
        chosen = []
        for idx, sc in indices_scores:
            ok = True
            for j in chosen:
                sim_ = cosine_similarity(S[idx], S[j])[0, 0]
                if sim_ >= 0.6:
                    ok = False
                    break
            if ok:
                chosen.append(idx)
            if len(chosen) >= topk:
                break
        picked_idx = sorted(chosen)

    return " ".join([sents[i] for i in picked_idx])

# ======== 推理：关键词（TF-IDF简单版 + RAKE兜底） ========
def extract_keywords(text: str, artifacts, k=KEYWORDS_K) -> List[str]:
    tfidf_kw = artifacts['tfidf_kw']
    vec = tfidf_kw.transform([text])
    if vec.nnz > 0:
        inds = vec.toarray().ravel().argsort()[::-1]
        feature_names = tfidf_kw.get_feature_names_out()
        words = [feature_names[i] for i in inds if i < len(feature_names)]
        words = [w for w in words if len(w) > 2][:k]
        if len(words) >= k // 2:
            return words[:k]
    # 兜底：RAKE
    rk = Rake(min_length=1, max_length=3)
    rk.extract_keywords_from_text(text)
    cands = [phrase for score, phrase in rk.get_ranked_phrases_with_scores()]
    return cands[:k]

# ======== CLI 快速试跑 ========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict"], default="train")
    ap.add_argument("--data_csv", default=DATA_CSV)
    ap.add_argument("--text", default="")
    ap.add_argument("--topk", type=int, default=TOPK_SENT)
    args = ap.parse_args()

    if args.mode == "train":
        arts = train_models(args.data_csv, MAX_TRAIN_DOCS)
        ckpt_dir = BASE_DIR / "assets" / "models" / "quick_ckpt"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        #
        pd.to_pickle(arts, str(ckpt_dir / "artifacts.pkl"))
        print(f"✅ Done. Saved to {ckpt_dir / 'artifacts.pkl'}")
    else:
        arts = pd.read_pickle(str(BASE_DIR / "assets" / "models" / "quick_ckpt" / "artifacts.pkl"))
        txt = args.text or ("The sun is shining and the sky is blue."
                            "Children are playing football in the park."
                            "A man is reading a newspaper on the bench."
                            "Two dogs are running and chasing each other."
                            "Everyone feels happy on this warm day.")
        print("---- Summary ----")
        print(summarize(txt, arts, topk=args.topk))
        print("---- Keywords ----")
        print(extract_keywords(txt, arts, k=KEYWORDS_K))



