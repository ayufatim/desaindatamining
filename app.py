import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# --- KONFIGURASI DAN PEMUATAN MODEL ---

# Muat model dan scaler yang telah disimpan
try:
    model = joblib.load('stacking_model_beginning_split.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Pastikan file 'stacking_model_beginning_split.pkl' dan 'scaler.pkl' berada di folder yang sama.")
    st.stop()

# Muat data asli untuk mendapatkan informasi kolom dan untuk scaler target
try:
    df_original = pd.read_csv('ExtendedEmployeePerformanceandProductivityData.csv')
except FileNotFoundError:
    st.error("Pastikan file 'ExtendedEmployeePerformanceandProductivityData.csv' ada di folder yang sama.")
    st.stop()

# --- DEFINISI FITUR DAN PEMROSESAN ---

# Tentukan urutan kolom yang benar sesuai dengan saat training
# (setelah one-hot encoding dan sebelum menghapus kolom non-numerik)
# Ini adalah langkah KRUSIAL
numerical_cols = [
    'Employee_ID', 'Age', 'Years_At_Company', 'Monthly_Salary',
    'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours',
    'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 'Training_Hours',
    'Promotions', 'Employee_Satisfaction_Score'
]

categorical_options = {
    'Department': ['IT', 'Finance', 'Customer Support', 'Engineering', 'Marketing', 'HR', 'Operations', 'Sales', 'Legal'],
    'Gender': ['Male', 'Female', 'Other'],
    'Job_Title': ['Specialist', 'Developer', 'Analyst', 'Manager', 'Technician', 'Engineer', 'Consultant'],
    'Education_Level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'Resigned': [True, False]
}

# Membuat scaler untuk variabel target (Performance_Score) agar bisa mengembalikan hasil prediksi ke skala asli
target_scaler = MinMaxScaler()
target_scaler.fit(df_original['Performance_Score'].values.reshape(-1, 1))

# Urutan kolom akhir yang diharapkan oleh model (setelah semua preprocessing di notebook)
# Anda harus mencocokkan ini dengan kolom di X_train dari notebook Anda
final_feature_order = [
    'Employee_ID', 'Age', 'Years_At_Company', 'Education_Level', 'Monthly_Salary',
    'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours', 'Sick_Days',
    'Remote_Work_Frequency', 'Team_Size', 'Training_Hours', 'Promotions',
    'Employee_Satisfaction_Score', 'Department_Engineering', 'Department_Finance',
    'Department_HR', 'Department_IT', 'Department_Legal', 'Department_Marketing',
    'Department_Operations', 'Department_Sales', 'Gender_Male', 'Gender_Other',
    'Job_Title_Consultant', 'Job_Title_Developer', 'Job_Title_Engineer',
    'Job_Title_Manager', 'Job_Title_Specialist', 'Job_Title_Technician',
    'Resigned_True', 'Experience_Salary_Interaction', 'Workload_Intensity',
    'Rolling_Avg_Performance', 'Overtime_Work_Ratio', 'SickDays_WorkDays_Ratio'
]
# Hapus kolom yang tidak ada di model training terakhir Anda dari notebook
# Berdasarkan notebook Anda, kolom-kolom berikut dihapus atau diubah.
# Sesuaikan jika notebook Anda berbeda.
final_feature_order_for_model = [col for col in final_feature_order if col not in ['Hire_Date']]


def preprocess_input(data):
    """
    Fungsi untuk memproses input mentah dari pengguna menjadi format
    yang dapat diterima oleh model.
    """
    # 1. Buat DataFrame dasar dengan semua kolom fitur yang diharapkan model, diisi dengan nol
    input_df = pd.DataFrame(columns=final_feature_order_for_model)
    input_df.loc[0] = 0

    # 2. Isi nilai numerik mentah
    for col in numerical_cols:
        if col in data:
            input_df.loc[0, col] = data[col]

    # 3. Proses fitur kategorikal (One-Hot dan Label Encoding)
    # Department
    if 'Department_' + data['Department'] in input_df.columns:
        input_df.loc[0, 'Department_' + data['Department']] = 1
    # Gender
    if 'Gender_' + data['Gender'] in input_df.columns:
        input_df.loc[0, 'Gender_' + data['Gender']] = 1
    # Job Title
    if 'Job_Title_' + data['Job_Title'] in input_df.columns:
        input_df.loc[0, 'Job_Title_' + data['Job_Title']] = 1
    # Resigned
    if data['Resigned']:
        input_df.loc[0, 'Resigned_True'] = 1

    # Education Level (Label Encoded)
    edu_mapping = {level: i for i, level in enumerate(categorical_options['Education_Level'])}
    input_df.loc[0, 'Education_Level'] = edu_mapping[data['Education_Level']]

    # 4. Lakukan feature engineering yang sama seperti di notebook
    input_df['Experience_Salary_Interaction'] = input_df['Years_At_Company'] * input_df['Monthly_Salary']
    # Hindari pembagian dengan nol
    if input_df.loc[0, 'Projects_Handled'] > 0:
        input_df['Workload_Intensity'] = input_df['Work_Hours_Per_Week'] / input_df['Projects_Handled']
    else:
        input_df['Workload_Intensity'] = 0

    # Untuk Rolling_Avg_Performance, kita tidak bisa menghitungnya dari satu input.
    # Kita bisa gunakan nilai rata-rata dari data training atau biarkan 0/nilai default.
    # Di notebook Anda, kolom ini tidak digunakan sebagai fitur input ke model, jadi kita abaikan.
    # Jika digunakan, Anda perlu strategi untuk menanganinya.
    input_df['Rolling_Avg_Performance'] = 0 # Placeholder

    if input_df.loc[0, 'Work_Hours_Per_Week'] > 0:
        input_df['Overtime_Work_Ratio'] = input_df['Overtime_Hours'] / input_df['Work_Hours_Per_Week']
        input_df['SickDays_WorkDays_Ratio'] = input_df['Sick_Days'] / input_df['Work_Hours_Per_Week']
    else:
        input_df['Overtime_Work_Ratio'] = 0
        input_df['SickDays_WorkDays_Ratio'] = 0


    # 5. Scaling fitur numerik menggunakan scaler yang sudah di-load
    # Pastikan hanya kolom yang ada di scaler yang di-transform
    cols_to_scale = [col for col in numerical_cols if col in input_df.columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # 6. Pastikan urutan kolom sesuai dengan model
    # Kolom 'Performance_Score' tidak ada di X_train, jadi kita hapus dari final_feature_order
    model_input_features = [col for col in final_feature_order_for_model if col != 'Performance_Score']

    # Handle jika ada kolom hasil feature engineering yang tidak ada di training
    for col in list(input_df.columns):
        if col not in model_input_features:
             model_input_features.append(col) # tambahkan jika perlu

    # Reorder dan filter kolom agar sesuai dengan input model
    final_input_df = pd.DataFrame(columns=model.feature_names_in_)
    for col in model.feature_names_in_:
        if col in input_df.columns:
            final_input_df[col] = input_df[col]
        else:
            final_input_df[col] = 0 # Jika ada kolom yang hilang, isi dengan 0
            
    return final_input_df


# --- ANTARMUKA STREAMLIT ---

st.set_page_config(page_title="Prediksi Produktivitas Karyawan", layout="wide")
st.title("📈 Prediksi Skor Kinerja & Produktivitas Karyawan")
st.markdown("Masukkan data karyawan untuk memprediksi skor kinerja mereka.")

with st.sidebar:
    st.header("Masukkan Data Karyawan")

    # Menggunakan nilai min, max, dan rata-rata dari data asli untuk input yang lebih intuitif
    def get_stats(column):
        min_val = int(df_original[column].min())
        max_val = int(df_original[column].max())
        mean_val = int(df_original[column].mean())
        return min_val, max_val, mean_val

    age_min, age_max, age_mean = get_stats('Age')
    years_min, years_max, years_mean = get_stats('Years_At_Company')
    salary_min, salary_max, salary_mean = get_stats('Monthly_Salary')
    hours_min, hours_max, hours_mean = get_stats('Work_Hours_Per_Week')
    projects_min, projects_max, projects_mean = get_stats('Projects_Handled')
    overtime_min, overtime_max, overtime_mean = get_stats('Overtime_Hours')
    sick_min, sick_max, sick_mean = get_stats('Sick_Days')
    remote_min, remote_max, remote_mean = get_stats('Remote_Work_Frequency')
    team_min, team_max, team_mean = get_stats('Team_Size')
    training_min, training_max, training_mean = get_stats('Training_Hours')
    promo_min, promo_max, promo_mean = get_stats('Promotions')
    sat_min, sat_max, sat_mean = 1.0, 5.0, 3.0 # Satisfaction score

    # Input dari pengguna
    age = st.slider("Usia", age_min, age_max, age_mean)
    years_at_company = st.slider("Lama Bekerja (Tahun)", years_min, years_max, years_mean)
    monthly_salary = st.slider("Gaji Bulanan", salary_min, salary_max, salary_mean)
    work_hours = st.slider("Jam Kerja per Minggu", hours_min, hours_max, hours_mean)
    projects_handled = st.slider("Jumlah Proyek", projects_min, projects_max, projects_mean)
    overtime_hours = st.slider("Jam Lembur", overtime_min, overtime_max, overtime_mean)
    sick_days = st.slider("Hari Sakit", sick_min, sick_max, sick_mean)
    remote_freq = st.slider("Frekuensi Kerja Remote (%)", remote_min, remote_max, remote_mean, step=25)
    team_size = st.slider("Ukuran Tim", team_min, team_max, team_mean)
    training_hours = st.slider("Jam Pelatihan", training_min, training_max, training_mean)
    promotions = st.slider("Jumlah Promosi", promo_min, promo_max, promo_mean)
    satisfaction = st.slider("Skor Kepuasan Karyawan", sat_min, sat_max, sat_mean)

    department = st.selectbox("Departemen", categorical_options['Department'])
    gender = st.selectbox("Gender", categorical_options['Gender'])
    job_title = st.selectbox("Jabatan", categorical_options['Job_Title'])
    education = st.selectbox("Tingkat Pendidikan", categorical_options['Education_Level'])
    resigned = st.selectbox("Status Resign", [False, True], format_func=lambda x: "Ya" if x else "Tidak")

# Tombol untuk prediksi
if st.sidebar.button("Prediksi Skor Kinerja", use_container_width=True):
    user_input = {
        'Employee_ID': 0, # Placeholder, karena tidak terlalu berpengaruh setelah scaling
        'Age': age,
        'Years_At_Company': years_at_company,
        'Monthly_Salary': monthly_salary,
        'Work_Hours_Per_Week': work_hours,
        'Projects_Handled': projects_handled,
        'Overtime_Hours': overtime_hours,
        'Sick_Days': sick_days,
        'Remote_Work_Frequency': remote_freq,
        'Team_Size': team_size,
        'Training_Hours': training_hours,
        'Promotions': promotions,
        'Employee_Satisfaction_Score': satisfaction,
        'Department': department,
        'Gender': gender,
        'Job_Title': job_title,
        'Education_Level': education,
        'Resigned': resigned
    }

    # Proses input
    processed_input = preprocess_input(user_input)

    # Lakukan prediksi
    prediction_scaled = model.predict(processed_input)

    # Kembalikan ke skala asli
    # prediction_original = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    prediction_original = prediction_scaled.reshape(-1, 1)

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prediksi Skor Kinerja (1-5)", value=f"{prediction_original[0][0]:.2f}")
        
    with col2:
        score = prediction_original[0][0]
        if score >= 1.0:
            st.success("🎉 **Luar Biasa!** Produktivitas karyawan ini sangat tinggi.")
        elif score >= 0.6:
            st.info("👍 **Baik.** Produktivitas karyawan ini di atas rata-rata.")
        elif score >= 0.3:
            st.warning("😐 **Cukup.** Produktivitas karyawan ini berada di level rata-rata.")
        else:
            st.error("⚠️ **Perlu Perhatian.** Produktivitas karyawan ini di bawah rata-rata.")

   
