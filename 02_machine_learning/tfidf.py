# encoding=utf-8

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

doc   = "苹果手机维修服务，提供屏幕更换、电池更换、主板维修与进水处理"

top_k = 5   # 设置取几个关键词
tok   = lambda s: [w for w in jieba.lcut(s) if len(w) > 1 and w.strip()]

v     = TfidfVectorizer(tokenizer=tok, token_pattern=None)
X     = v.fit_transform([doc])  # 如果有多篇文档，就用docs列表替换[doc]即可
vocab = v.get_feature_names_out()

# 提取TF（scikit-learn的TfidfVectorizer默认是词频/文档长度）
tf = (X > 0).astype(int).toarray()[0] * 0  # 先占位
term_counts = np.array(X.toarray()[0]) / v.idf_  # TF（因为TF-IDF / IDF = TF）
tf = term_counts / term_counts.sum()             # 归一化后就是TF

idf = v.idf_  # 每个词的IDF
tfidf = X.toarray()[0]

# 取权重最高的top_k
idx = tfidf.argsort()[::-1][:top_k]

for i in idx:
    print(f"{vocab[i]}\tTF: {tf[i]:.3f}\tIDF: {idf[i]:.3f}\tTF-IDF: {tfidf[i]:.3f}")


