"""XGBoost time-series demo: create 30 days of sales and forecast 7 days.

Run:
    pip install -r requirements-xgboost-demo.txt
    python xgboost_forecast_demo.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# Keep matplotlib's cache in a writable location when the home directory is locked.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_SEED = 42
HISTORY_DAYS = 30
FORECAST_DAYS = 7
START_DATE = "2026-10-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgboost_demo_output"
DATA_PATH = Path(__file__).resolve().parent / "xgboost_demo_data.csv"
FEATURE_COLUMNS = [
    "day_index",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_7",
    "rolling_mean_3",
]


def create_demo_data() -> pd.DataFrame:
    """Create reproducible daily sales with trend and weekly seasonality."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, periods=HISTORY_DAYS, freq="D")
    day_index = np.arange(HISTORY_DAYS)
    weekly_effect = np.array([0, 2, 4, 6, 10, 18, 12])
    noise = rng.normal(loc=0, scale=3.5, size=HISTORY_DAYS)

    sales = 120 + 1.8 * day_index + weekly_effect[dates.dayofweek] + noise
    sales = np.round(sales, 1)

    return pd.DataFrame({"date": dates, "sales": sales})


def load_demo_data() -> pd.DataFrame:
    """Load the included 30-day CSV so it can also be edited by hand."""
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if len(data) != HISTORY_DAYS:
        raise ValueError(f"示例数据必须正好包含 {HISTORY_DAYS} 行，当前为 {len(data)} 行")
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build calendar, lag, and rolling features using only earlier values."""
    featured = data.copy()
    featured["day_index"] = np.arange(len(featured))
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    featured["lag_1"] = featured["sales"].shift(1)
    featured["lag_2"] = featured["sales"].shift(2)
    featured["lag_7"] = featured["sales"].shift(7)
    featured["rolling_mean_3"] = featured["sales"].shift(1).rolling(3).mean()
    return featured.dropna().reset_index(drop=True)


def build_model() -> XGBRegressor:
    """Return a small regression model suitable for this demo dataset."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def evaluate_model(featured: pd.DataFrame) -> tuple[float, float]:
    """Use the last five known days as a chronological holdout set."""
    train = featured.iloc[:-5]
    test = featured.iloc[-5:]
    model = build_model()
    model.fit(train[FEATURE_COLUMNS], train["sales"])
    predictions = model.predict(test[FEATURE_COLUMNS])

    mae = mean_absolute_error(test["sales"], predictions)
    rmse = np.sqrt(mean_squared_error(test["sales"], predictions))
    return float(mae), float(rmse)


def make_next_day_features(
    next_date: pd.Timestamp, history_values: list[float], day_index: int
) -> pd.DataFrame:
    """Create one feature row from actual or previously predicted history."""
    day_of_week = next_date.dayofweek
    return pd.DataFrame(
        [
            {
                "day_index": day_index,
                "day_of_week": day_of_week,
                "is_weekend": int(day_of_week >= 5),
                "lag_1": history_values[-1],
                "lag_2": history_values[-2],
                "lag_7": history_values[-7],
                "rolling_mean_3": float(np.mean(history_values[-3:])),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def forecast_next_week(
    history: pd.DataFrame, model: XGBRegressor
) -> pd.DataFrame:
    """Recursively forecast seven days without using unknown future values."""
    history_values = history["sales"].astype(float).tolist()
    last_date = history["date"].max()
    forecasts: list[dict[str, object]] = []

    for step in range(1, FORECAST_DAYS + 1):
        next_date = last_date + pd.Timedelta(days=step)
        feature_row = make_next_day_features(
            next_date=next_date,
            history_values=history_values,
            day_index=len(history_values),
        )
        prediction = max(0.0, float(model.predict(feature_row)[0]))
        prediction = round(prediction, 1)
        history_values.append(prediction)
        forecasts.append({"date": next_date, "predicted_sales": prediction})

    return pd.DataFrame(forecasts)


def save_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> Path:
    """Save a line chart that connects historical and forecast values."""
    chart_path = OUTPUT_DIR / "sales_forecast.png"
    forecast_line = pd.concat(
        [
            history.tail(1).rename(columns={"sales": "predicted_sales"}),
            forecast,
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(history["date"], history["sales"], marker="o", label="Historical sales")
    ax.plot(
        forecast_line["date"],
        forecast_line["predicted_sales"],
        marker="o",
        linestyle="--",
        label="7-day forecast",
    )
    ax.axvline(history["date"].max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title("XGBoost Daily Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def main() -> None:
    history = load_demo_data()
    featured = add_features(history)
    mae, rmse = evaluate_model(featured)

    final_model = build_model()
    final_model.fit(featured[FEATURE_COLUMNS], featured["sales"])
    forecast = forecast_next_week(history, final_model)

    OUTPUT_DIR.mkdir(exist_ok=True)
    history_path = OUTPUT_DIR / "historical_sales.csv"
    forecast_path = OUTPUT_DIR / "future_7_days_forecast.csv"
    history.to_csv(history_path, index=False, date_format="%Y-%m-%d")
    forecast.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")
    chart_path = save_chart(history, forecast)

    print("\n=== 30天模拟销量（最后10天）===")
    print(history.tail(10).to_string(index=False))
    print("\n=== 时间顺序验证集指标（最后5个已知日）===")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("\n=== 未来7天预测 ===")
    print(forecast.to_string(index=False))
    print("\n输出文件：")
    print(f"- {history_path}")
    print(f"- {forecast_path}")
    print(f"- {chart_path}")


if __name__ == "__main__":
    main()
