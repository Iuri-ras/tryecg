import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Sidebar inputs
st.sidebar.title("🫀 ECG Signal Filtering App")
st.sidebar.markdown("---")
st.sidebar.title("Input Options")

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV file", type="csv")
load_sample = st.sidebar.button("Load Sample Data")

# Sampling frequency input
fs = st.sidebar.number_input(
    label="Sampling Frequency (Hz)",
    min_value=1,
    max_value=5000,
    value=250,
    step=1,
    help="Sampling frequency of your ECG signal in Hertz"
)

st.sidebar.markdown("---")
st.sidebar.header("Datasets")
st.sidebar.markdown("[Kaggle ECG Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
st.sidebar.markdown("[PhysioNet ECG Database](https://physionet.org/about/database/)")

# Main app title and explanation
st.title("🫀 ECG Signal Filtering Application")
st.markdown("""
**What does the filter do?**

- Removes low-frequency baseline drift (<0.5 Hz) and high-frequency noise (>40 Hz).
- Enhances the clarity of the QRS complex, which is key for heart rate analysis.
""")

# Initialize df variable early
df = None

# Load sample data if button pressed
if load_sample:
    time = np.linspace(0, 10, 2500)
    ecg_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.random.randn(2500)
    df = pd.DataFrame({"Time": time, "ECG Signal": ecg_signal})
    st.success("Sample ECG data loaded!")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload an ECG CSV file or load sample data to begin.")

# Display and process data if available
if df is not None:
    with st.expander("Preview Data"):
        st.write(df.head())

    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    # Apply the bandpass filter
    filtered_signal = bandpass_filter(ecg_signal, 0.5, 40, fs=fs)

    # Side-by-side plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original ECG Signal")
        fig, ax = plt.subplots()
        ax.plot(time, ecg_signal, color="blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with col2:
        st.subheader("Filtered ECG Signal")
        fig2, ax2 = plt.subplots()
        ax2.plot(time, filtered_signal, color="green")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        st.pyplot(fig2)

    # Highlight QRS visibility improvement
    st.markdown(
        """
        <div style="
            background-color:#198754;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            margin: 20px 0;
        ">
            QRS Visibility Improved: Filtering removes noise and drift, enhancing the QRS complex for analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Download filtered data    
    filtered_df = pd.DataFrame({"Time": time, "Filtered ECG Signal": filtered_signal})
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered ECG Data as CSV",
        data=csv_data,
        file_name="filtered_ecg.csv",
        mime="text/csv",
        help="Download the filtered ECG signal data",
    )
