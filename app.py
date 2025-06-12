import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج
model, feature_columns = joblib.load('radiation_model.pkl')

st.title("🔬 AI Radiation Classifier")
st.write("ادخل الخصائص الفيزيائية لتحديد ما إذا كان الإشعاع ضار أم لا")

# إدخال البيانات
radiation_type = st.selectbox("نوع الإشعاع", ['WiFi', '4G', '5G', 'Bluetooth', 'IR'])
frequency = st.number_input("🔸 التردد (GHz)", step=0.01)
power = st.number_input("🔸 شدة الإشارة (dBm)", step=0.1)
duration = st.number_input("🔸 مدة التعرض (بالدقايق)", step=1)
sar = st.number_input("🔸 SAR (W/kg)", step=0.01)

if st.button("🔍 تصنيف"):
    input_data = {
        'frequency_GHz': frequency,
        'power_dBm': power,
        'duration_minutes': duration,
        'SAR_W_per_kg': sar,
    }

    for r_type in ['WiFi', '4G', '5G', 'Bluetooth', 'IR']:
        input_data[f'radiation_type_{r_type}'] = 1 if radiation_type == r_type else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    st.success(f"✅ النتيجة: الإشعاع هو **{prediction}**")
