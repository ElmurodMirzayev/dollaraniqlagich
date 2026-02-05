import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.title("USD → UZS Forecast (Buy/Sell model)")

# ===== Загружаем модель =====
model = joblib.load("usduzs_buysell.pkl")

# ===== Загружаем данные =====
df = pd.read_csv("USD_UZS_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

st.write("Последние данные")
st.dataframe(df.tail())

# ===== Пользователь вводит =====
days = st.number_input("Сколько дней прогнозировать", 1, 30, 7)

# =========================================
# Функция добавления строки
# =========================================

def add_new_row(df, predicted_direction):

    df = df.copy()

    last_close = df['close'].iloc[-1]

    # маленькое изменение цены
    step = last_close * 0.001   # 0.1%

    if predicted_direction == 1:
        predicted_close = last_close + step
    else:
        predicted_close = last_close - step

    new_date = df['date'].iloc[-1] + pd.Timedelta(days=1)

    new_change = (predicted_close - last_close) / last_close * 100

    new_row = {
        'date': new_date,
        'close': predicted_close,
        'change': new_change
    }

    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


# =========================================
# Функция прогнозирования
# =========================================

def predict_by_day(df, model, days):

    df = df.copy()

    for i in range(days):

        dayofweek = df['date'].iloc[-1].dayofweek
        month = df['date'].iloc[-1].month

        my_data = pd.DataFrame([{
            'close_lag1': df['close'].iloc[-1],
            'close_lag7': df['close'].iloc[-7],
            'close_ma7': df['close'].iloc[-7:].mean(),
            'change_lag1': df['change'].iloc[-1],
            'change': df['change'].iloc[-1],
            'dayofweek': dayofweek,
            'month': month
        }])

        predicted_direction = model.predict(my_data)[0]

        df = add_new_row(df, predicted_direction)

    return df


# =========================================
# Кнопка
# =========================================

if st.button("Сделать прогноз"):

    result_df = predict_by_day(df, model, days)

    last_real_date = df['date'].iloc[-1]

    new_data = result_df[result_df['date'] > last_real_date]
    old_data = result_df[result_df['date'] <= last_real_date]

    st.subheader("График")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=old_data['date'],
        y=old_data['close'],
        mode='lines',
        name='История'
    ))

    fig.add_trace(go.Scatter(
        x=new_data['date'],
        y=new_data['close'],
        mode='lines',
        name='Прогноз'
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Прогноз")
    st.dataframe(new_data)
