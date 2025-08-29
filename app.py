from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import io, base64

from core_v22pro import analyze_symbol as analyze_signals  # کد اصلی سیگنال‌ها از این فایل ایمپورت میشه

app = FastAPI(title="Trading Signal API",
              description="API برای تولید سیگنال با ابزارهای مختلف تحلیل تکنیکال",
              version="1.0.0")

# 📌 ورودی کاربر
class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "15m"   # کاربر می‌تواند 15m, 1h, 4h, 1d انتخاب کند
    indicators: list = ["rsi", "macd", "ema"]
    combine: bool = False    # اگر True باشد، ابزارها ترکیب می‌شوند

# 📌 خروجی تست ساده
@app.get("/")
def home():
    return {"message": "Trading Signal API is running 🚀"}

# 📌 تولید سیگنال
@app.post("/signal")
def get_signal(req: SignalRequest):
    # 🔹 اینجا داده‌های کندل (OHLCV) را باید از صرافی بگیری یا شبیه‌سازی کنی
    # برای تست فعلاً داده‌ی ساختگی تولید می‌کنیم
    np.random.seed(42)
    df = pd.DataFrame({
        "close": np.random.rand(100) * 100,
        "high": np.random.rand(100) * 105,
        "low": np.random.rand(100) * 95,
    })

    # 🔹 ارسال به هسته اصلی تحلیل (core_v22pro)
    result, chart = analyze_signals(df, req.indicators, req.combine)

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "indicators": req.indicators,
        "combine": req.combine,
        "signal": result,
        "chart": chart   # نمودار به صورت base64
    }
