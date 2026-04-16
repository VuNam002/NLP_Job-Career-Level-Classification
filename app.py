import streamlit as st
import joblib
import pandas as pd
import re

st.set_page_config(
    page_title="Career Level Predictor",
    layout="centered"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("job_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Không tìm thấy file **job_model.pkl**. Vui lòng chạy `job.py` trước.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()


def filter_location(location):
    result = re.findall(r"\,\s[A-Z]{2}$", str(location))
    return result[0][2:] if result else location


st.markdown('<div class="hero-title">Career Level<br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Nhập thông tin công việc để dự đoán cấp bậc phù hợp (Bằng tiếng Anh) </div>', unsafe_allow_html=True)
st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

with st.form("prediction_form"):
    title = st.text_input("Job Title", placeholder="Data Scientist, Product Manager...")

    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", placeholder="Berlin, BE")
    with col2:
        function = st.text_input("Function", placeholder="IT, Marketing...")

    industry = st.text_input("Industry", placeholder="Technology, Finance...")
    description = st.text_area("Job Description", placeholder="Mô tả ngắn về vị trí công việc...", height=130)

    submitted = st.form_submit_button("Dự đoán →")

if submitted:
    if not all([title, location, description, function, industry]):
        st.warning("Vui lòng điền đầy đủ tất cả các trường.")
    else:
        input_data = pd.DataFrame([{
            "title": title,
            "location": filter_location(location),
            "description": description,
            "function": function,
            "industry": industry
        }])

        try:
            with st.spinner("Đang phân tích..."):
                prediction = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Cấp bậc dự đoán</div>
                <div class="result-value">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {e}")