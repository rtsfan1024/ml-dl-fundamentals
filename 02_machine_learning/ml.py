import pandas as pd
import numpy as np
from pathlib import Path

# 引入 sklearn 的核心架构组件
from sklearn.compose import ColumnTransformer # 列转换器：用于将不同的预处理逻辑分发到不同的列
from sklearn.preprocessing import OneHotEncoder, StandardScaler # 编码器与标准化工具
from sklearn.pipeline import Pipeline # 管道：工程化极客的最爱，防数据泄漏 (Data Leakage) 核心利器
from sklearn.linear_model import LinearRegression, Ridge, Lasso # 线性模型家族
from sklearn.impute import SimpleImputer # 缺失值填充器
from sklearn.metrics import mean_absolute_error, root_mean_squared_error # 评估指标计算器
from sklearn.model_selection import KFold, cross_val_score # 交叉验证组件
from sklearn.metrics import make_scorer

# ==========================================
# ⚙️ 架构配置区 (CONFIG) - 统一管理实验超参
# ==========================================
BASE_DIR   = Path(__file__).resolve().parent.parent  # 项目根目录
DATA_PATH  = str(BASE_DIR / "assets" / "data" / "ecommerce_sales_dataset.csv")  # 原始数据挂载路径
MODEL_TYPE  = "none"     # 算法选型: "none"=普通线性回归, "l2"=岭回归(防过拟合), "l1"=Lasso(自带特征选择)
ALPHA       = 1.0        # 正则化惩罚力度（仅在选了 l1 或 l2 时生效，值越大对复杂模型的惩罚越狠）
EVAL_MODE   = "rmse"      # 验证模式: "mae"(平均绝对误差), "rmse"(均方根误差), "cv"(交叉验证)
CV_FOLDS    = 5          # 交叉验证的折数（把数据切5份，轮流做测试，最抗数据倾斜）
SHUFFLE_TRAIN_VAL = True # 切分训练/验证集前是否洗牌（打乱顺序，防止原数据按时间或地域聚集引发偏差）
RANDOM_STATE = 42        # 随机种子：AI 工程化的底线，保证你我跑出来的结果绝对一致 (Reproducibility)
DO_PREDICT  = False       # 业务开关：True=执行最终预测并落盘(True=启动预测导出CSV), False=仅做线下评估
PRED_OUTPATH = str(BASE_DIR / "assets" / "data" / "predictions_linear_regression.csv")  # 最终战果的输出路径


# ==========================================
# 🛠️ 核心算子定义区 (Operators)
# ==========================================

def make_ohe():
    """
    向下兼容的独热编码器生成工厂。
    [架构意图]: 解决 Scikit-Learn 1.2+ 版本破坏性更新把 sparse 参数改名为 sparse_output 的工程痛点。
    handle_unknown="ignore" 极度重要：如果预测集中出现了训练集没见过的新类别，直接全填0，防止线上宕机崩溃。
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # 适配新版 sklearn
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # 适配老版 sklearn


def build_pipeline(model_type: str, alpha: float, numeric_cols, categorical_cols) -> Pipeline:
    """
    构建端到端的 ML 管道 (End-to-End ML Pipeline)。
    [架构意图]: 强制规范数据流，先清洗再进模型。用 Pipeline 封装可以 100% 杜绝特征在训练集和验证集之间串户泄漏。
    """
    # 1. 连续型数值特征管道 (如：价格、重量)
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # 缺失值填充策略：中位数（比平均数更抗极端离群值干扰）
        ("scaler", StandardScaler()),                  # 标准化：把数据拉扁到均值0方差1，防止数值极大的列（如千万级销售额）权重吞噬其他列
    ])
    
    # 2. 离散型类别特征管道 (如：颜色、尺码)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), # 缺失值填充策略：众数（取出现频次最高的类目）
        ("ohe", make_ohe()),                                  # 转变为 One-Hot 向量（使得线性模型能理解非数字字符）
    ])
    
    # 3. 列转换器拼合 (分治策略)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),       # 对数值列应用数值管道
            ("cat", categorical_pipeline, categorical_cols), # 对类别列应用类别管道
        ],
        remainder="drop", # 没被指定的列（如一些垃圾特征）直接丢弃
    )

    # 4. 挂载回归预测头 (Predictive Head)
    if model_type == "none":
        reg = LinearRegression()
    elif model_type == "l2":
        reg = Ridge(alpha=alpha, random_state=RANDOM_STATE) # L2正则，削弱多重共线性 (Multicollinearity)
    elif model_type == "l1":
        reg = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=10000) # L1正则，会自动把没用的特征权重直接归零
    else:
        raise ValueError("MODEL_TYPE 必须是 'none', 'l2', 'l1' 中的一个")

    # 5. 返回终极管道对象：预处理引擎 -> 回归预测器
    return Pipeline(steps=[("prep", preprocessor), ("reg", reg)])


# ==========================================
# 📊 线下评估模块 (Evaluation)
# ==========================================

def evaluate_fixed_split(model: Pipeline, X_train, y_train, X_val, y_val, mode: str) -> float:
    """在固定的 train(1000) / val(500) 划分上评估。"""
    model.fit(X_train, y_train) # 训练：模型开始学习权重
    pred = model.predict(X_val) # 验证：模型对没见过的数据进行闭卷考试

    if mode == "mae":
        score = mean_absolute_error(y_val, pred) # 绝对误差
    elif mode == "rmse":
        score = root_mean_squared_error(y_val, pred)  # 均方根误差 (RMSE)，对极端的大预测错误惩罚更狠
    else:
        raise ValueError("固定切分模式下，EVAL_MODE 必须是 'mae' 或 'rmse'。")

    return score


def evaluate_cross_validation(model: Pipeline, X, y, mode: str, folds: int) -> float:
    """
    高级兵器：K折交叉验证 (K-Fold Cross Validation)。
    [架构意图]: 把全部 1500 条数据切 5 块，每次拿 4 块训练 1 块测试，循环 5 次求平均。指标极其稳定，是最真实的实力体现。
    """
    if mode == "mae":
        scoring = "neg_mean_absolute_error"
    elif mode == "rmse":
        try:
            scoring = "neg_root_mean_squared_error"
            _ = cross_val_score(model, X, y, cv=2, scoring=scoring) # 探针：先用2折试探一下当前 sklearn 版本支不支持此指标
        except Exception:
            # 如果不支持，就手写一个打分器注入进去
            scoring = make_scorer(lambda yt, yp: -root_mean_squared_error(yt, yp))
    else:
        scoring = "neg_root_mean_squared_error" # 默认 fallback

    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring) # 启动 5 轮高强度模型炼丹评估
    
    # 因为 sklearn 内部约定所有指标必须“越大越好”，所以算误差时它偷偷加了负号。最后我们取负号还原本尊。
    return float(-scores.mean())


def run_prediction(model: Pipeline, X_pred: pd.DataFrame, pred_df: pd.DataFrame, out_path: str):
    """业务落地：对未来的预测集生成最终预测并落盘 CSV。"""
    y_pred = model.predict(X_pred)
    out = pred_df.copy()
    out["y_pred"] = y_pred # 注入预测列
    Path(out_path).parent.mkdir(parents=True, exist_ok=True) # 极客防御：确保输出目录的文件夹存在，防止 IO 报错
    out.to_csv(out_path, index=False)
    print(f"✅ 已导出预测结果到: {out_path}")


# ==========================================
# 🚀 主控逻辑 (Main Data Flow)
# ==========================================
def main():
    # 1. 读入数据大盘
    df = pd.read_csv(DATA_PATH)

    # 2. 自动定位目标预测列 (Label)
    target_col = "y" if "y" in df.columns else df.columns[-1]

    # 保留最原始的行号，方便后续如果发现数据异常，能顺藤摸瓜找回原表是哪一行
    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})

    # 3. 物理隔绝数据集：有目标值的(前1500条) vs 没目标值等我们去猜的(后500条)
    is_labeled = ~df[target_col].isna() # 生成布尔掩码 (~ 表示取反，即 not isna)
    labeled_df = df[is_labeled].copy()  # 已标注的数据底座
    pred_df = df[~is_labeled].copy()    # 待预测的未来业务数据

    # 防御性编程：万一数据源根本没有缺失的列，强制把最后 500 条切走当成待预测数据
    if pred_df.empty:
        pred_df = df.tail(500).copy()
        labeled_df = df.drop(pred_df.index).copy()

    # 4. 从标注集中提纯出特征矩阵 (X)
    drop_cols = {target_col, "row_index", "id", "ID", "Id"} # 杀掉业务无关/无特征意义的主键
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if SHUFFLE_TRAIN_VAL: # 如果配置了洗牌
        labeled_df = labeled_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # [切分训练/验证集数据块]
    train_df = labeled_df.iloc[:1000].copy()
    val_df   = labeled_df.iloc[1000:1500].copy()

    X_train = train_df[feature_cols]             # 训练特征集
    y_train = train_df[target_col].astype(float) # 训练目标集 (Label)
    X_val   = val_df[feature_cols]               # 验证特征集
    y_val   = val_df[target_col].astype(float)   # 验证目标集
    X_pred  = pred_df[feature_cols]              # 最终业务待预测特征集

    # 动态嗅探特征类型，实现预处理的自动路由
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # 5. 装配组装好的 ML Pipeline
    model = build_pipeline(MODEL_TYPE, ALPHA, numeric_cols, categorical_cols)

    # 6. 线下评估流水线启动
    print(f"=== 实验配置流水线 ===")
    print(f"MODEL_TYPE={MODEL_TYPE}, ALPHA={ALPHA}, EVAL_MODE={EVAL_MODE}")
    print(f"Train数据量={len(train_df)}, Val数据量={len(val_df)}, Predict待推断量={len(pred_df)}")

    if EVAL_MODE in ("mae", "rmse"):
        score = evaluate_fixed_split(model, X_train, y_train, X_val, y_val, EVAL_MODE)
        print(f"[Fixed 1000/500] {EVAL_MODE.upper()} = {score:.6f} (越小越好)")
    elif EVAL_MODE == "cv":
        X_all = labeled_df[feature_cols]
        y_all = labeled_df[target_col].astype(float)
        cv_score = evaluate_cross_validation(model, X_all, y_all, "rmse", CV_FOLDS)
        print(f"[{CV_FOLDS}-Fold CV on 1500 labeled] RMSE = {cv_score:.6f} (越小越好)")
    else:
        raise ValueError("EVAL_MODE must be one of: 'mae', 'rmse', 'cv'.")

    # 7. 终极一战：开启全部 1500 条数据的二次重训，压榨最后一丝数据价值，最后输出预测！
    if DO_PREDICT:
        X_all = labeled_df[feature_cols]             # 这里合并了前面的 1000+500
        y_all = labeled_df[target_col].astype(float) 
        
        model.fit(X_all, y_all) # 在最全量的数据上重新拟合（这在生产环境叫 Retrain）
        run_prediction(model, X_pred, pred_df, PRED_OUTPATH) # 输出那缺失的 500 个推断结果

if __name__ == "__main__":
    main()