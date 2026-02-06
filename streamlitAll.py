import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("USD → UZS Forecast va BUY/SELL")

# ===== Modelarni yuklash (agar mavjud bo'lsa) =====
model_reg = None
model_clf = None
try:
    model_reg = joblib.load("usduzs.pkl")
except Exception as e:
    st.warning("Model 'usduzs.pkl' topilmadi yoki yuklab bo'lmadi (prognoz uchun).")

try:
    model_clf = joblib.load("usduzs_buysell.pkl")
except Exception:
    # ba'zi foydalanuvchilar faqat bitta model nomi bilan ishlaydi — shuning uchun ogohlantirish beramiz
    st.info("Agar BUY/SELL klassifikatori kerak bo'lsa, 'usduzs_buysell.pkl' nomli modelni joylang.")

# ===== CSV yuklash =====
try:
    df = pd.read_csv("USD_UZS_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
except Exception as e:
    st.error("CSV fayl 'USD_UZS_data.csv' topilmadi yoki uni o'qib bo'lmadi. Iltimos fayl mavjudligini tekshiring.")
    st.stop()

st.subheader("So'nggi ma'lumotlar")
st.dataframe(df.tail())

# ===== Mode tanlash =====
mode = st.radio(
    "Rejimni tanlang:",
    ("📌 Bir kunlik BUY / SELL", "🔮 Bir nechta kunlik prognoz")
)

# ===== Helper funktsiyalar =====

def safe_iloc(df, idx, default=None):
    try:
        return df.iloc[idx]
    except Exception:
        return default

# ===== Bir kunlik BUY / SELL rejimi =====
if mode == "📌 Bir kunlik BUY / SELL":

    st.subheader("📥 Kunlik ma'lumotlarni kiriting")

    yesterday_close_str = st.text_input(
        "Kecha USD/UZS kursi", value=f"{df['close'].iloc[-1]:.2f}"
    )

    today_close_str = st.text_input(
        "Bugun USD/UZS kursi", value=f"{df['close'].iloc[-1]:.2f}"
    )

    dayofweek = st.selectbox(
        "Hafta kuni",
        [
            (0, "Dushanba"),
            (1, "Seshanba"),
            (2, "Chorshanba"),
            (3, "Payshanba"),
            (4, "Juma"),
            (5, "Shanba"),
            (6, "Yakshanba")
        ],
        format_func=lambda x: x[1]
    )[0]

    month = st.selectbox(
        "Oy",
        list(range(1, 13)),
        index=df['date'].iloc[-1].month - 1
    )

    if st.button("🔮 BUY / SELL aniqlash"):
        try:
            yesterday_close = float(yesterday_close_str.replace(",", "."))
            today_close = float(today_close_str.replace(",", "."))
        except ValueError:
            st.error("❌ Iltimos, to'g'ri son kiriting (masalan: 12200.30)")
        else:
            if yesterday_close == 0:
                st.error("❌ Kechagi kurs 0 bo'lishi mumkin emas")
            else:
                change = (today_close - yesterday_close) / yesterday_close * 100

                # xavfsiz 7 kun oldingi qiymatlarni olish
                try:
                    close_lag7 = df['close'].iloc[-7]
                except Exception:
                    close_lag7 = df['close'].iloc[-1]

                try:
                    close_ma7 = df['close'].iloc[-7:].mean()
                except Exception:
                    close_ma7 = df['close'].iloc[-1]

                try:
                    change_lag1 = df['change'].iloc[-1]
                except Exception:
                    change_lag1 = 0.0

                my_data = pd.DataFrame([{
                    'close_lag1': yesterday_close,
                    'close_lag7': close_lag7,
                    'close_ma7': close_ma7,
                    'change_lag1': change_lag1,
                    'change': change,
                    'dayofweek': dayofweek,
                    'month': month
                }])

                with st.expander("📌 Modelga yuborilgan ma'lumotlar"):
                    st.write(my_data.T)

                if model_clf is None:
                    st.error("🔴 BUY/SELL klassifikatori yuklanmagan. 'usduzs_buysell.pkl' faylini joylang.")
                else:
                    try:
                        prediction = model_clf.predict(my_data)[0]
                        if int(prediction) == 1:
                            st.success("🟢 TAVSIYA: **BUY (Sotib olish)**")
                        else:
                            st.error("🔴 TAVSIYA: **SELL (Sotish)**")
                    except Exception as e:
                        st.error(f"Model bilan bashorat qilishda xatolik: {e}")

                st.info(f"📊 Hisoblangan oʻzgarish (change): **{change:.4f}%**")

# ===== Bir nechta kunlik prognoz rejimi =====
else:
    st.subheader("📈 Bir nechta kunlik prognoz")

    days = st.number_input("Necha kun prognoz qilinsin", 1, 365, 7)

    def add_new_row(df_local, predicted_close):
        df_local = df_local.copy()

        new_date = df_local['date'].iloc[-1] + pd.Timedelta(days=1)
        last_close = df_local['close'].iloc[-1]
        new_change = predicted_close - last_close

        # xavfsizlagan holda -7 uchun qiymatlar
        try:
            close_lag7_val = df_local['close'].iloc[-7]
        except Exception:
            close_lag7_val = df_local['close'].iloc[-1]

        try:
            close_ma7_val = df_local['close'].iloc[-7:].mean()
        except Exception:
            close_ma7_val = df_local['close'].iloc[-1]

        try:
            change_lag1_val = df_local['change'].iloc[-1]
        except Exception:
            change_lag1_val = 0.0

        new_row = {
            'date': new_date,
            'close': predicted_close,
            'change': new_change,
            'close_lag1': last_close,
            'close_lag7': close_lag7_val,
            'close_ma7': close_ma7_val,
            'change_lag1': change_lag1_val
        }

        return pd.concat([df_local, pd.DataFrame([new_row])], ignore_index=True)

    def predict_by_day(df_input, model, days_count):
        df_work = df_input.copy()
        for i in range(days_count):
            my_date = pd.DataFrame([{
                'close_lag1': df_work['close'].iloc[-1],
                'close_lag7': df_work['close'].iloc[-7] if len(df_work) >= 7 else df_work['close'].iloc[-1],
                'close_ma7': df_work['close'].iloc[-7:].mean() if len(df_work) >= 7 else df_work['close'].iloc[-1],
                'change_lag1': df_work['change'].iloc[-1] if 'change' in df_work.columns else 0.0
            }])

            # model_reg kutilyapti — u regressiya bo'lib, yopiq narxni (predicted_close) beradi
            if model is None:
                raise RuntimeError("Prognoz modeli yuklanmagan. 'usduzs.pkl' faylini joylang.")

            predicted_close = model.predict(my_date)[0]

            # agar model int yoki klass qaytarsa, uni floatga o'tkazishga harakat qilamiz
            try:
                predicted_close = float(predicted_close)
            except Exception:
                # agar qiymat floatga o'tmasa, fallback sifatida oxirgi close + 0
                predicted_close = float(df_work['close'].iloc[-1])

            df_work = add_new_row(df_work, predicted_close)

        return df_work

    if st.button("Sdelat prognoz"):
        try:
            result_df = predict_by_day(df, model_reg, int(days))

            last_real_date = df['date'].iloc[-1]

            new_data = result_df[result_df['date'] >= last_real_date]
            old_data = result_df[(result_df['date'] > last_real_date - pd.Timedelta(days=200)) & (result_df['date'] <= last_real_date)]

            st.subheader("Grafik")

            fig, ax = plt.subplots()

            ax.plot(old_data['date'], old_data['close'], label="Tarixiy ma'lumotlar")
            ax.plot(new_data['date'], new_data['close'], label="Prognoz")

            plt.xticks(rotation=45)
            plt.legend()

            st.pyplot(fig)

            st.subheader("Prognozlar jadvali")
            st.dataframe(result_df.tail(days))

        except Exception as e:
            st.error(f"Prognoz qilishda xatolik: {e}")
