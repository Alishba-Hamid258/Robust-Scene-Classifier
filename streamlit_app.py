
import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import base64

st.set_page_config(page_title="SceneAI Industrial", page_icon="🛡️", layout="wide")

# PIXEL-PERFECT CSS: No shadows, no rounding, maximum sharpness
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 32px; color: #4CAF50; font-weight: bold; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 12px; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; width: 100%; height: 3.5em; }
    
    img { border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.4); }
    </style>
    """, unsafe_allow_html=True)

API_URL = "http://localhost:8000/predict"

def predict_image(image_file):
    try:
        image_file.seek(0)
        files = {"file": (image_file.name, image_file, image_file.type)}
        response = requests.post(API_URL, files=files)
        return response.json()
    except Exception as e:
        return {"error": "connection", "message": str(e)}

def main():
    with st.sidebar:
        st.title("🛡️ Industrial Guard")
        st.markdown("v7.2 - Pixel-Perfect Display")
        st.divider()
        st.write("### 🛠️ Mode")
        ultra_strict = st.toggle("Ultra-Strict Mode", value=False)
        threshold = st.slider("Min Confidence", 0.0, 1.0, 0.85)
        st.divider()
        st.caption("Rendering: Hardware Pixel-Mapping (No Smoothing)")

    st.title("🖼️ Universal Scene Classifier")
    st.write("---")
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.write("### 📤 Image Upload")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            import base64
            from PIL import Image
            
            # 1. Detect Native Image Size
            image_bytes = uploaded_file.getvalue()
            img_info = Image.open(io.BytesIO(image_bytes))
            native_w, native_h = img_info.size
            
            # 2. Intelligent Sizing Logic
            # If the image is tiny (like the 150px dataset images), don't stretch it to 100%.
            # Instead, display it at a max of 300px and center it beautifully.
            if native_w < 400:
                css_style = "max-width: 300px; width: 100%; display: block; margin: 0 auto; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.4);"
                container_bg = "background-color: #161a24; padding: 20px; border-radius: 15px; text-align: center;"
            else:
                css_style = "width: 100%; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.4);"
                container_bg = ""
                
            # 3. Base64 Encoding for Lossless Browser Display
            encoded = base64.b64encode(image_bytes).decode()
            mime_type = uploaded_file.type
            
            html_code = f'''
                <div style="{container_bg}">
                    <p style="color: #888; font-size: 12px; margin-bottom: 5px; text-align: center;">
                        Native Resolution: {native_w}x{native_h}
                    </p>
                    <img src="data:{mime_type};base64,{encoded}" style="{css_style}">
                </div>
            '''
            st.markdown(html_code, unsafe_allow_html=True)
            
            if st.button("🚀 START INDUSTRIAL ANALYSIS"):
                with st.spinner("Analyzing..."):
                    result = predict_image(uploaded_file)
                    st.session_state['result'] = result

    with col2:
        st.write("### 📊 Classification Result")
        if 'result' in st.session_state and st.session_state['result'] is not None:
            result = st.session_state['result']
            
            if "error" in result:
                st.error(f"Error: {result['message']}")
            else:
                pred_class = result['predicted_class']
                conf = result['confidence']
                is_confused = result.get('is_confused', False)
                suspicion = result.get('suspicion')
                
                # MODERN ADAPTIVE LOGIC
                if pred_class == "other":
                    st.error("🚫 NOT A SCENE: This looks like a non-landscape object.")
                if suspicion:
                    st.warning(f"⚠️ QUALITY WARNING: {suspicion}")
                if is_confused:
                    st.info("🤔 LOW DIFFERENTIATION: The AI is seeing multiple possible scenes.")

                # RESULTS
                c1, c2 = st.columns(2)
                with c1: st.metric("AI Prediction", pred_class.title())
                with c2: st.metric("Confidence", f"{conf*100:.1f}%")
                
                if conf > threshold:
                    st.success(f"✅ Confirmed: **{pred_class.upper()}**")
                else:
                    st.warning(f"🔍 Probable: **{pred_class.upper()}**")
                
                st.write("---")
                st.write("#### Confidence Breakdown")
                probs = result["class_probabilities"]
                df = pd.DataFrame({
                    'Category': [c.title() for c in probs.keys()],
                    'Probability': list(probs.values())
                }).set_index('Category')
                st.bar_chart(df, color="#4CAF50")
        else:
            st.warning("No data available.")

if __name__ == "__main__":
    main()
