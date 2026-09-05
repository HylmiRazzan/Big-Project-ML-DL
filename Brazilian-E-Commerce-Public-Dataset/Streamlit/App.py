import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Prediksi NPS", layout="centered")
st.title("Prediksi Sentimen NPS")
st.write("Sistem otomatis membaca ulasan (Bahasa Indonesia didukung).")

@st.cache_resource
def load_model():
    lokasi_folder = os.path.dirname(__file__)
    path_model = os.path.join(lokasi_folder, 'nps_stacking_model.pkl')
    return joblib.load(path_model)

sistem = load_model()
preprocessor = sistem['preprocessor']
model_lgbm = sistem['model_lgbm']
model_xgb = sistem['model_xgb']
komandan_meta = sistem['meta_learner']

with st.form("form_prediksi"):
    st.subheader("1. Ulasan")
    review_title = st.text_input("Judul", "")
    review_message = st.text_area("Pesan", "")
    
    st.subheader("2. Pengiriman & Logistik")
    col1, col2, col3 = st.columns(3)
    delivery_duration_days = col1.number_input("Aktual (Hari)", min_value=0, value=5)
    estimated_duration_days = col2.number_input("Estimasi (Hari)", min_value=0, value=7)
    delivery_delay_days = col3.number_input("Terlambat (Hari)", value=0)
    
    st.subheader("3. Finansial")
    col4, col5, col6 = st.columns(3)
    price = col4.number_input("Harga", min_value=0.0, value=50.0)
    freight_value = col5.number_input("Ongkir", min_value=0.0, value=15.0)
    payment_value = col6.number_input("Total Bayar", min_value=0.0, value=65.0)
    
    col7, col8, col9 = st.columns(3)
    payment_installments = col7.number_input("Cicilan", min_value=1, value=1)
    payment_sequential = col8.number_input("Urutan Bayar", min_value=1, value=1)
    order_item_id = col9.number_input("Jml Barang", min_value=1, value=1)
    
    st.subheader("4. Spesifikasi Produk")
    col10, col11, col12, col13 = st.columns(4)
    product_weight_g = col10.number_input("Berat (g)", min_value=0.0, value=500.0)
    product_length_cm = col11.number_input("Panjang (cm)", min_value=0.0, value=20.0)
    product_height_cm = col12.number_input("Tinggi (cm)", min_value=0.0, value=15.0)
    product_width_cm = col13.number_input("Lebar (cm)", min_value=0.0, value=15.0)
    
    col14, col15 = st.columns(2)
    product_description_lenght = col14.number_input("Karakter Deskripsi", min_value=0, value=200)
    product_photos_qty = col15.number_input("Jml Foto", min_value=0, value=1)
    
    st.subheader("5. Kategori")
    customer_state = st.selectbox("Provinsi", ["SP", "RJ", "MG", "RS", "PR", "Lainnya"])
    payment_type = st.selectbox("Tipe Bayar", ["credit_card", "boleto", "voucher", "debit_card"])
    product_category_name = st.selectbox("Kategori Produk", ["bed_bath_table", "health_beauty", "sports_leisure", "computers_accessories", "Lainnya"])
    
    submit_button = st.form_submit_button("Prediksi")

if submit_button:
    teks_indo = f"{review_title} {review_message}".strip()
    
    if teks_indo != "":
        try:
            teks_portugis = GoogleTranslator(source='id', target='pt').translate(teks_indo)
        except Exception:
            st.warning("⚠️ Server penerjemah Google sedang menolak koneksi. Teks tidak diterjemahkan, hasil prediksi mungkin kurang akurat.")
            teks_portugis = teks_indo
    else:
        teks_portugis = ""

    safe_price = price + 1
    safe_weight = product_weight_g + 1
    safe_estimate = estimated_duration_days + 1
    safe_installments = max(1, payment_installments)
    
    input_data = pd.DataFrame({
        'payment_value': [payment_value],
        'price': [price],
        'freight_value': [freight_value],
        'product_description_lenght': [product_description_lenght],
        'product_weight_g': [product_weight_g],
        'product_length_cm': [product_length_cm],
        'product_height_cm': [product_height_cm],
        'product_width_cm': [product_width_cm],
        'freight_to_price_ratio': [freight_value / safe_price],
        'payment_per_installment': [payment_value / safe_installments],
        'product_volume_cm3': [product_length_cm * product_height_cm * product_width_cm],
        'freight_efficiency': [freight_value / safe_weight],
        'delivery_duration_days': [delivery_duration_days],
        'delivery_delay_days': [delivery_delay_days],
        'delivery_speed_ratio': [delivery_duration_days / safe_estimate],
        'review_length': [len(teks_portugis)],
        'word_count': [len(teks_portugis.split())],
        'payment_sequential': [payment_sequential],
        'payment_installments': [payment_installments],
        'order_item_id': [order_item_id],
        'product_photos_qty': [product_photos_qty],
        'combined_text': [teks_portugis],
        'customer_state': [customer_state],
        'payment_type': [payment_type],
        'product_category_name': [product_category_name]
    })
    
    try:
        x_processed = preprocessor.transform(input_data)
        x_dense = x_processed.toarray() if hasattr(x_processed, "toarray") else x_processed
        
        prob_lgbm = model_lgbm.predict_proba(x_dense)
        prob_xgb = model_xgb.predict_proba(x_dense)
        meta_x = np.hstack((prob_lgbm, prob_xgb))
        
        prediksi_akhir = komandan_meta.predict(meta_x)[0]
        
        kamus_hasil = {
            0: ("Kecewa ", "error"),
            1: ("Netral ", "warning"),
            2: ("Puas ", "success")
        }
        
        teks_hasil, status = kamus_hasil[prediksi_akhir]
        
        if status == "error":
            st.error(f"Hasil: {teks_hasil}")
        elif status == "warning":
            st.warning(f"Hasil: {teks_hasil}")
        else:
            st.success(f"Hasil: {teks_hasil}")
            
    except Exception as e:
        st.error(f"Error: {e}")
