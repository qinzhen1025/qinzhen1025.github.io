---
title: "中国 COVID-19 疫情指标统计分析与 ARIMA 滚动预测（2020–2022）"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/china-covid19-arima-forecast
date: 2026-01-18
excerpt: "基于公开疫情数据完成中国疫情指标洞察，并用 ARIMA 实现 7 天滚动预测与误差评估，为趋势研判提供可量化基线模型。"
header:
  teaser: /images/portfolio/china-covid19-arima-forecast/teaser.png
tags:
- 时间序列
- ARIMA
- 疫情分析
- 统计分析
- 滚动预测
- 模型评估
tech_stack:
- name: Python
- name: Pandas
- name: Statsmodels
- name: Scikit-learn
- name: Matplotlib
- name: Seaborn
---

## 项目背景（Background）
本项目基于公开 COVID-19 数据（国家维度统计），聚焦中国在 **2020-01-23 至 2022-12-31** 时间段内的疫情关键指标变化，包括新增确诊、死亡、政策严格度指数（stringency index）与再生数（reproduction rate）。

项目目标分为两部分：
1. **统计分析与可视化**：通过描述性统计、相关性分析与滞后相关分析，刻画指标间的联动关系及其随时间变化的特征。
2. **时间序列预测基线**：以 **new_cases_smoothed** 为目标序列，构建 **ARIMA 7 天滚动预测（Rolling Forecast）**流程，并用误差指标评估预测效果与稳定性。

---

## 数据说明（Dataset）
- 数据来源：公开 COVID-19 国家级时间序列数据（OWID 等常用公开数据源）
- 分析对象：China
- 时间范围：2020-01-23 ~ 2022-12-31
- 关键字段：
  - `new_cases` / `new_cases_smoothed`
  - `new_deaths` / `new_deaths_smoothed`
  - `stringency_index`
  - `reproduction_rate`

---

## 核心实现（Implementation）

### 1）数据筛选与预处理（中国 + 时间范围 + 关键字段）
核心步骤包括：国家筛选、字段筛选、日期格式转换、时间窗裁剪。

```python
china_data = df[df["country"] == "China"][[
    "date",
    "total_cases", "new_cases", "new_cases_smoothed",
    "total_deaths", "new_deaths", "new_deaths_smoothed",
    "stringency_index", "reproduction_rate"
]].copy()

china_data["date"] = pd.to_datetime(china_data["date"])
china_data = china_data[
    (china_data["date"] >= "2020-01-23") &
    (china_data["date"] <= "2022-12-31")
]
```

---

### 2）描述性统计（提炼关键阶段与极值）
通过均值、中位数、四分位数、极值及其发生日期，快速定位疫情的关键波动阶段。

```python
analysis_cols = ["new_cases", "new_deaths", "stringency_index", "reproduction_rate"]

stats = {}
for col in analysis_cols:
    s = china_data[col]
    stats[col] = {
        "mean": s.mean(),
        "median": s.median(),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
        "min": s.min(),
        "max": s.max(),
        "max_date": china_data.loc[s.idxmax(), "date"] if s.notna().any() else None
    }

stats_df = pd.DataFrame(stats).T
stats_df
```

---

### 3）相关性分析与滞后效应探索（Correlation + Lag）
为了更深入理解指标间关系，本项目进一步从三个角度进行分析：
- **整体相关性**：观察指标间是否存在同步变化趋势；
- **滚动相关性**：分析相关关系是否随时间漂移（结构变化）；
- **滞后相关性**：探索“领先/滞后”效应，为解释机制与建模提供依据。

> 该部分主要以可视化结果呈现（见下方 Results）。

---

### 4）ARIMA：7 天滚动预测（Rolling Forecast）
本项目将 `new_cases_smoothed` 作为目标序列，采用滚动预测策略：
- 每次用当前训练集拟合 ARIMA
- 预测未来 7 天
- 将真实值并入历史序列，继续下一轮预测
- 最终汇总所有预测窗口，计算整体误差

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

ts = china_data[["date", "new_cases_smoothed"]].dropna().copy()
ts = ts.set_index("date")["new_cases_smoothed"]

def is_stationary(series):
    p_value = adfuller(series.dropna(), autolag="AIC")[1]
    return p_value < 0.05

# 自动差分直到平稳（最多差分 3 次）
d = 0
diff = ts.copy()
while (not is_stationary(diff)) and d < 3:
    diff = diff.diff().dropna()
    d += 1

# ARIMA 参数（可进一步用 AIC/BIC 优化）
p, q = 2, 2
horizon = 7

y_true_all, y_pred_all = [], []
start_idx = 0

while start_idx + horizon < len(ts):
    train = ts.iloc[:start_idx]

    # 训练序列过短则跳过
    if len(train) < 30:
        start_idx += horizon
        continue

    model = ARIMA(train, order=(p, d, q)).fit()
    forecast = model.forecast(steps=horizon)

    test = ts.iloc[start_idx:start_idx + horizon]
    y_true_all.extend(test.values.tolist())
    y_pred_all.extend(forecast.values.tolist())

    start_idx += horizon

rmse = mean_squared_error(y_true_all, y_pred_all, squared=False)
mae  = mean_absolute_error(y_true_all, y_pred_all)
mape = mean_absolute_percentage_error(y_true_all, y_pred_all)

rmse, mae, mape
```

---

## 结果与分析（Results & Analysis）

### 1）疫情关键指标总览（趋势观察）
> 该图建议作为封面图（teaser），用于展示项目整体分析范围与核心变量。

![疫情指标总览](/images/portfolio/china-covid19-arima-forecast/teaser.png)

**结论要点（建议写法）：**
- 不同阶段新增确诊与死亡呈现明显波动，反映疫情发展与防控策略变化；
- `stringency_index` 与 `reproduction_rate` 在关键阶段出现明显变化，体现政策与传播强度的动态关系。

---

### 2）相关性分析（整体关系）
![相关性总览](/images/portfolio/china-covid19-arima-forecast/correlation_overview.png)

**结论要点（建议写法）：**
- 通过相关性矩阵可以快速判断指标之间的同步变化关系；
- 在不同阶段（例如疫情早期与后期），相关结构可能出现变化，提示数据存在非平稳特征。

---

### 3）滚动相关性（结构漂移）
![滚动相关性](/images/portfolio/china-covid19-arima-forecast/rolling_correlation.png)

**结论要点（建议写法）：**
- 滚动相关性用于观察“指标关系是否稳定”；
- 若曲线波动明显，说明疫情传播机制与政策响应之间的关系可能随阶段变化而漂移。

---

### 4）滞后相关分析（领先/滞后效应）
![滞后相关曲线](/images/portfolio/china-covid19-arima-forecast/lag_correlation_curves.png)

![滞后相关热图](/images/portfolio/china-covid19-arima-forecast/lag_correlation_heatmap.png)

**结论要点（建议写法）：**
- 滞后相关用于探索一个变量对另一个变量的“延迟影响”；
- 例如政策变化或再生数变化，可能在若干天后反映到新增确诊/死亡的变化上；
- 该分析为后续引入外生变量（ARIMAX/SARIMAX）提供了可解释依据。

---

### 5）ARIMA 7 天滚动预测效果（基线模型）
![ARIMA滚动预测](/images/portfolio/china-covid19-arima-forecast/arima_rolling_forecast.png)

![滚动窗口误差](/images/portfolio/china-covid19-arima-forecast/rolling_window_errors.png)

**结论要点（建议写法）：**
- ARIMA 作为经典统计模型，适合作为时间序列预测的 baseline；
- 在趋势相对平稳阶段预测效果较好，但在突发波动阶段误差容易增大；
- 滚动窗口误差可以反映模型在不同阶段的稳定性，为模型升级提供方向。

---

## 我的贡献（What I did）
- 构建中国疫情数据分析数据集：完成筛选、清洗、时间窗裁剪；
- 完成多指标趋势分析与可视化，输出可解释的阶段性结论；
- 通过相关性、滚动相关与滞后相关分析，探索指标联动机制；
- 实现 ARIMA 7 天滚动预测流程，并通过 RMSE/MAE/MAPE 进行量化评估。

---

## 可优化方向（Next Steps）
- 采用 AIC/BIC 或网格搜索优化 ARIMA 的 `(p,d,q)` 参数；
- 引入季节性与结构变化：尝试 SARIMA、Prophet 等方法；
- 引入外生变量：构建 ARIMAX/SARIMAX，将政策指数与再生数纳入模型；
- 针对波动剧烈阶段进行分段建模或变点检测，提高鲁棒性。
