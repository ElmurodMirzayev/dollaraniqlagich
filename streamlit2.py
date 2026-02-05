import streamlit as st
import joblib
import pandas as pd

# ---------- Заголовок ----------
st.title("💱 USD → UZS BUY / SELL Predictor")

# ---------- Загрузка модели ----------
@st.cache_resource
def load_model():
    return joblib.load("usduzs_buysell.pkl")

model = load_model()

# ---------- Загрузка данных ----------
@st.cache_data
def load_data():
    df = pd.read_csv("USD_UZS_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

df = load_data()

st.success("Model and data loaded ✅")

# ---------- Ввод данных ----------
st.subheader("📥 Enter today's data")

close = st.text_input(
    "Today's USD/UZS close price",
    value="12200.30"
)

dayofweek = st.selectbox(
    "Day of week",
    options=[
        (0, "Monday"),
        (1, "Tuesday"),
        (2, "Wednesday"),
        (3, "Thursday"),
        (4, "Friday"),
        (5, "Saturday"),
        (6, "Sunday")
    ],
    format_func=lambda x: x[1]
)[0]

month = st.selectbox(
    "Month",
    list(range(1, 13))
)

# ---------- Предсказание ----------
if st.button("🔮 Predict"):
    try:
        close = float(close)

        close_yesterday = df['close'].iloc[-1]
        change = (close - close_yesterday) / close_yesterday * 100

        close_lag1 = df['close'].iloc[-1]
        close_lag7 = df['close'].iloc[-7]
        close_ma7 = df['close'].iloc[-7:].mean()
        change_lag1 = df['change'].iloc[-1]

        my_data = pd.DataFrame([{
            'close_lag1': close_lag1,
            'close_lag7': close_lag7,
            'close_ma7': close_ma7,
            'change_lag1': change_lag1,
            'change': change,
            'dayofweek': dayofweek,
            'month': month
        }])

        prediction = model.predict(my_data)[0]

        if prediction == 1:
            st.success("🟢 **Predicted action: BUY**")
        else:
            st.error("🔴 **Predicted action: SELL**")

    except ValueError:
        st.error("❌ Please enter a valid number for close price")
