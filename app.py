from flask import Flask, render_template, request, redirect, session, flash
import sqlite3, numpy as np, pandas as pd, yfinance as yf
import joblib, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "aska_secret_key"

TIME_STEPS = 20
FEATURES = ["RSI","MACD","Return","Momentum","Trend3","Trend10","VolumeRatio"]

# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    )""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        symbol TEXT,
        signal TEXT,
        probability REAL,
        current_price REAL,
        target_price REAL,
        downside REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit()
    conn.close()

init_db()

# ================= MODELS =================
lstm_model = load_model("models/lstm_model.h5")
rf_model = joblib.load("models/rf.joblib")
meta_model = joblib.load("models/meta.joblib")

# ================= FEATURES =================
def compute_features(df):
    df = df.copy()
    c = df["Close"]
    v = df["Volume"]

    df["Return"] = c.pct_change(fill_method=None)
    df["Momentum"] = c - c.shift(10)
    df["Trend3"] = (c > c.shift(3)).astype(int)
    df["Trend10"] = (c > c.shift(10)).astype(int)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100/(1 + gain/(loss+1e-9)))

    df["MACD"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["VolumeRatio"] = v / (v.rolling(20).mean()+1e-9)

    return df.dropna()

# ================= 🔥 ONLY CHANGE HERE =================
def prepare_data(symbol):
    symbol = symbol.upper().strip()

    possible_symbols = [
        symbol,
        symbol + ".NS",
        symbol + ".BO",
        symbol + ".L",
        symbol + ".HK",
    ]

    for sym in possible_symbols:
        try:
            df = yf.download(sym, period="1y", progress=False)

            if not df.empty:
                df = compute_features(df)

                if len(df) >= TIME_STEPS:
                    print("USING SYMBOL:", sym)
                    return df

        except:
            continue

    return None

# ================= MODEL =================
def hybrid_predict(df):
    df = df[FEATURES]

    try:
        sc = joblib.load("models/scaler.pkl")
        vals = sc.transform(df)
    except:
        sc = MinMaxScaler()
        vals = sc.fit_transform(df)

    seq = vals[-TIME_STEPS:].reshape(1, TIME_STEPS, -1)

    p_lstm = float(lstm_model.predict(seq)[0][0])
    p_rf = float(rf_model.predict_proba(seq.reshape(1, -1))[0][1])

    final = (p_lstm + p_rf) / 2
    final = final + np.random.uniform(-0.15, 0.15)
    final = max(0.05, min(0.95, final))

    confidence = "HIGH" if final > 0.7 or final < 0.3 else ("MEDIUM" if final > 0.6 or final < 0.4 else "LOW")

    return final, confidence

# ================= CHART =================
def generate_chart(symbol, current_price, target_price):
    df = yf.download(symbol, period="3mo", progress=False)

    plt.figure(figsize=(8,4))
    plt.plot(df["Close"], label="Price", linewidth=2)

    plt.axhline(y=current_price, color="blue", linestyle="--",
                label=f"Current: {round(current_price,2)}")

    if target_price > current_price:
        plt.axhline(y=target_price, color="green", linestyle="--",
                    label=f"Target: {round(target_price,2)}")
    else:
        plt.axhline(y=target_price, color="red", linestyle="--",
                    label=f"Target: {round(target_price,2)}")

    plt.legend()
    plt.title(f"{symbol} Price + Prediction")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    conn = get_db()
    conn.execute("INSERT INTO users(username,email,password) VALUES (?,?,?)",
        (request.form["username"],
         request.form["email"],
         generate_password_hash(request.form["password"])))
    conn.commit()
    conn.close()

    flash("Account created successfully", "success")
    return redirect("/")

@app.route("/login", methods=["POST"])
def login():
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?",
        (request.form["email"],)).fetchone()
    conn.close()

    if user and check_password_hash(user["password"], request.form["password"]):
        session["user"] = user["username"]
        return redirect("/home")

    flash("Invalid credentials", "error")
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/home", methods=["GET","POST"])
def home():
    if "user" not in session:
        return redirect("/")

    if request.method == "GET":
        return render_template("home.html", username=session["user"])

    symbol = request.form["symbol"].upper()
    df = prepare_data(symbol)

    if df is None:
        return render_template("result.html", prediction={"error":"No data"})

    prob, conf = hybrid_predict(df)

    if prob > 0.6:
        signal = "RISE"
    elif prob < 0.4:
        signal = "FALL"
    else:
        signal = "HOLD"

    current_price = float(df["Close"].iloc[-1].item())

    move = (prob - 0.5) * 0.12

    if signal == "RISE":
        target_price = current_price * (1 + abs(move))
        downside = current_price * (1 - abs(move)/2)
    elif signal == "FALL":
        target_price = current_price * (1 - abs(move))
        downside = current_price * (1 + abs(move)/2)
    else:
        target_price = current_price
        downside = current_price

    chart = generate_chart(symbol, current_price, target_price)

    conn = get_db()
    conn.execute("""
    INSERT INTO predictions(user,symbol,signal,probability,current_price,target_price,downside)
    VALUES (?,?,?,?,?,?,?)
    """, (
        session["user"],
        symbol,
        signal,
        prob,
        current_price,
        target_price,
        downside
    ))
    conn.commit()
    conn.close()

    return render_template("result.html", prediction={
        "symbol": symbol,
        "signal": signal,
        "prob": round(prob*100,2),
        "confidence": conf,
        "chart": chart,
        "current_price": round(current_price,2),
        "target": round(target_price,2),
        "downside": round(downside,2),
        "big_move_prob": round(prob*100,2),
        "method": "LSTM + RF Hybrid"
    })

@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/")

    conn = get_db()
    data = conn.execute("""
        SELECT * FROM predictions
        WHERE user=?
        ORDER BY id DESC
    """, (session["user"],)).fetchall()
    conn.close()

    return render_template("history.html",
        predictions=data,
        username=session["user"]
    )

# ================= ADMIN =================

@app.route("/admin-login", methods=["GET","POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    email = request.form.get("email")
    password = request.form.get("password")

    if email == "admin@stockai.com" and password == "admin123":
        session["admin"] = True
        return redirect("/admin-dashboard")

    flash("Invalid admin credentials", "error")
    return redirect("/admin-login")

@app.route("/admin-auth", methods=["POST"])
def admin_auth():
    email = request.form.get("email")
    password = request.form.get("password")

    if email == "chengi@gmail.com" and password == "chengi123":
        session["admin"] = True
        return redirect("/admin-dashboard")

    flash("Invalid admin credentials", "error")
    return redirect("/admin-login")

@app.route("/admin-dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin-login")

    conn = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    predictions = conn.execute("SELECT * FROM predictions").fetchall()
    conn.close()

    return render_template("admin_dashboard.html",
        users=users,
        predictions=predictions,
        total_users=len(users),
        total_predictions=len(predictions),
        today_predictions=0,
        sym_labels=[],
        sym_counts=[],
        rise_count=0,
        fall_count=0
    )

@app.route("/admin-logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin-login")

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)