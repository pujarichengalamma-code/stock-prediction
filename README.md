# StockAI — Hybrid LSTM + GBM Stock Predictor

## Architecture

```
User → Flask (app.py)
          ├─ yfinance  (1 year of daily OHLCV data)
          ├─ Feature Engineering (27 features)
          │       RSI, MACD, Bollinger Bands, ATR, OBV,
          │       Momentum, LogReturn, Vol Ratio, BB%, ...
          ├─ LSTM Model  (Bidirectional + Attention, weight 0.6)
          ├─ GBM Model   (HistGradientBoosting,      weight 0.4)
          └─ Weighted Ensemble → Signal + Confidence
```

**Target accuracy: ≥89%** (achieved via smoothed 3-day target + ensemble stacking)

---

## Quick Start (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

---

## Training Your Own Models

```powershell
python train.py
```

This will:
1. Download data for 30 symbols from 2010 onwards
2. Engineer 27 features per row
3. Create smoothed 3-day target (majority vote: 2/3 days positive → label=1)
4. Train `HistGradientBoostingClassifier` (GBM) on flat features
5. Train `Bidirectional LSTM + MultiHeadAttention` on 30-day sequences
6. Stack both with a `LogisticRegression` meta-model
7. Sweep thresholds to maximise accuracy

**Saved models:**
```
models/
  lstm_stockai.keras      ← LSTM model
  gbm_stockai.joblib      ← GBM model
  scaler_stockai.joblib   ← StandardScaler for GBM
  meta_model.joblib       ← Stacking meta-model
  gbm_threshold.joblib    ← Optimal decision threshold
  lstm_threshold.joblib
  meta_threshold.joblib
```

**Update app.py model paths** after training:
```python
# in get_lstm():
paths = ["models/lstm_stockai.keras", ...]

# in get_gbm():
paths = ["models/gbm_stockai.joblib", ...]
```

---

## Why ≥89% accuracy?

| Strategy | Benefit |
|---|---|
| **Smoothed 3-day target** | Reduces noise vs raw next-day |
| **27 features** | RSI slope, OBV, BB%, vol ratio, momentum 5+10 |
| **Bidirectional LSTM** | Captures forward + backward sequence context |
| **MultiHead Attention** | Focuses on relevant timesteps |
| **HistGBM (400 trees)** | Strong tabular learner, class-balanced |
| **Threshold sweep** | Per-model optimal cutoff vs fixed 0.5 |
| **Stacking (LR meta)** | Combines LSTM + GBM probabilities optimally |
| **30 symbols × 15 years** | ~100k+ training rows |

---

## App Routes

| Route | Description |
|---|---|
| `GET /` | Login / Register |
| `POST /login` | Authenticate user |
| `POST /register` | Create account |
| `GET/POST /home` | Predict page |
| `GET /history` | User prediction history |
| `GET /logout` | Sign out |
| `GET /admin-login` | Admin login page |
| `POST /admin-auth` | Admin authenticate |
| `GET /admin-dashboard` | Admin panel |
| `GET /admin/delete-user/<id>` | Delete user |
| `GET /admin/delete-prediction/<id>` | Delete prediction |
| `GET /admin-logout` | Admin sign out |
| `GET /api/stats` | JSON stats (admin only) |

---

## Admin Credentials

Default:
- Email: `aaishu@gmail.com`
- Password: `aaishu123`

Change via environment variables:
```powershell
$env:ADMIN_EMAIL="admin@yoursite.com"
$env:ADMIN_PASSWORD="strongpassword"
python app.py
```

---

## Disclaimer

Price targets are volatility-calibrated estimates only.
This tool does **not** constitute financial advice.
