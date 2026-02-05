import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="USD → UZS Sotib olish / Sotish",
    layout="wide"
)

st.title("💱 USD → UZS Sotib olish / Sotish ва Прогноз")

# ===============================
# Модель ва маълумотларни юклаш
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("usduzs_buysell.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("USD_UZS_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

model = load_model()
df = load_data()

st.success("Модель ва маълумотлар юкланди ✅")

st.subheader("📊 Охирги маълумотлар")
st.dataframe(df.tail())

# ===============================
# Ён панель — режим танлаш
# ===============================
mode = st.sidebar.radio(
    "Режимни танланг",
    ["📌 Бир кунлик BUY / SELL", "📈 Бир неча кунлик прогноз"]
)

# =====================================================
# 🔹 1-РЕЖИМ: Бир кунлик BUY / SELL
# =====================================================
if mode == "📌 Бир кунлик BUY / SELL":

    st.subheader("📥 Кунлик маълумотларни киритинг")

    yesterday_close_str = st.text_input(
        "Кечаги USD/UZS курси",
        value=f"{df['close'].iloc[-1]:.2f}"
    )

    today_close_str = st.text_input(
        "Бугунги USD/UZS курси",
        value=f"{df['close'].iloc[-1]:.2f}"
    )

    dayofweek = st.selectbox(
        "Ҳафта куни",
        [
            (0, "Душанба"),
            (1, "Сешанба"),
            (2, "Чоршанба"),
            (3, "Пайшанба"),
            (4, "Жума"),
            (5, "Шанба"),
            (6, "Якшанба")
        ],
        format_func=lambda x: x[1]
    )[0]

    month = st.selectbox(
        "Ой",
        list(range(1, 13)),
        index=df['date'].iloc[-1].month - 1
    )

    if st.button("🔮 BUY / SELL аниқлаш"):
        try:
            yesterday_close = float(yesterday_close_str.replace(",", "."))
            today_close = float(today_close_str.replace(",", "."))
        except ValueError:
            st.error("❌ Илтимос, тўғри сон киритинг (масалан: 12200.30)")
        else:
            if yesterday_close == 0:
                st.error("❌ Кечаги курс 0 бўлиши мумкин эмас")
            else:
                change = (today_close - yesterday_close) / yesterday_close * 100

                close_lag7 = df['close'].iloc[-7]
                close_ma7 = df['close'].iloc[-7:].mean()
                change_lag1 = df['change'].iloc[-1]

                my_data = pd.DataFrame([{
                    'close_lag1': yesterday_close,
                    'close_lag7': close_lag7,
                    'close_ma7': close_ma7,
                    'change_lag1': change_lag1,
                    'change': change,
                    'dayofweek': dayofweek,
                    'month': month
                }])

                with st.expander("📌 Модельга юборилган маълумотлар"):
                    st.write(my_data.T)

                prediction = model.predict(my_data)[0]

                if prediction == 1:
                    st.success("🟢 ТАВСИЯ: **BUY (Сотиб олиш)**")
                else:
                    st.error("🔴 ТАВСИЯ: **SELL (Сотиш)**")

                st.info(f"📊 Ҳисобланган ўзгариш (change): **{change:.4f}%**")

# =====================================================
# 🔹 2-РЕЖИМ: Бир неча кунлик прогноз
# =====================================================
else:

    st.subheader("📅 Прогноз параметрлари")

    days = st.number_input(
        "Неча кунга прогноз қилиш керак?",
        min_value=1,
        max_value=30,
        value=7
    )

    def add_new_row(df, direction):
        df = df.copy()
        last_close = df['close'].iloc[-1]
        step = last_close * 0.001  # 0.1%

        predicted_close = (
            last_close + step if direction == 1 else last_close - step
        )

        new_row = {
            'date': df['date'].iloc[-1] + pd.Timedelta(days=1),
            'close': predicted_close,
            'change': (predicted_close - last_close) / last_close * 100
        }

        return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    def predict_days(df, model, days):
        df = df.copy()

        for _ in range(days):
            my_data = pd.DataFrame([{
                'close_lag1': df['close'].iloc[-1],
                'close_lag7': df['close'].iloc[-7],
                'close_ma7': df['close'].iloc[-7:].mean(),
                'change_lag1': df['change'].iloc[-1],
                'change': df['change'].iloc[-1],
                'dayofweek': df['date'].iloc[-1].dayofweek,
                'month': df['date'].iloc[-1].month
            }])

            direction = model.predict(my_data)[0]
            df = add_new_row(df, direction)

        return df

    if st.button("📈 Прогноз қилиш"):
        result_df = predict_days(df, model, days)

        last_real_date = df['date'].iloc[-1]
        history = result_df[result_df['date'] <= last_real_date]
        forecast = result_df[result_df['date'] > last_real_date]

        st.subheader("📉 График")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['date'],
            y=history['close'],
            name="Тарих"
        ))
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['close'],
            name="Прогноз"
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📄 Прогноз жадвали")
        st.dataframe(forecast.reset_index(drop=True))
