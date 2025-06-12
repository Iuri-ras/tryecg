import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Define the bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Add ECG QRS detection
def detect_qrs(ecg_signal, fs):
    # Simple QRS detection using peak finding
    peaks, _ = find_peaks(ecg_signal, distance=fs/2)  # Minimum distance between R-peaks (heart rate)
    return peaks

# Streamlit app title and sidebar inputs
st.title("ECG Signal Filtering and Analysis")

st.sidebar.title("ECG Processing Controls")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", min_value=10.0, max_value=100.0, value=40.0, step=1.0)
fs = st.sidebar.number_input("Sampling Frequency (Hz)", min_value=1, max_value=5000, value=100, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Datasets")
st.sidebar.markdown("[PhysioNet ECG Database](https://physionet.org/about/database/)")

# If file is uploaded
if uploaded_file is not None:
    # Load ECG data from the uploaded file
    ecg_data = pd.read_csv(uploaded_file)

    # Check if the file has the expected format and columns
    if ecg_data.shape[1] < 1:
        st.error("The uploaded file doesn't contain any data.")
    else:
        # Extract the first column as the ECG signal
        ecg_signal = ecg_data.iloc[:, 0]

        # Apply the bandpass filter
        filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

        # Detect QRS peaks
        qrs_peaks = detect_qrs(filtered_ecg, fs)

        # Plot the original ECG signal
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(ecg_signal, color='blue', label="Original ECG")
        ax1.set_title('Original ECG Signal')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc="upper right")
        st.pyplot(fig1)

        # Plot the filtered ECG signal
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(filtered_ecg, color='green', label="Filtered ECG")
        ax2.set_title('Filtered ECG Signal (0.5–40 Hz Bandpass)')
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Amplitude')
        ax2.legend(loc="upper right")
        st.pyplot(fig2)

        # Plot QRS peaks on the filtered signal
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(filtered_ecg, color='green', label="Filtered ECG")
        ax3.plot(qrs_peaks, filtered_ecg[qrs_peaks], 'ro', label="Detected QRS Peaks")
        ax3.set_title("Filtered ECG with Detected QRS Peaks")
        ax3.set_xlabel('Time (samples)')
        ax3.set_ylabel('Amplitude')
        ax3.legend(loc="upper right")
        st.pyplot(fig3)

        # Comment on the QRS Visibility Change
        st.markdown("""
        **QRS Visibility Improvement:**
        After applying the bandpass filter, you will notice that the baseline wander (low-frequency drift) and powerline noise (high-frequency noise) are removed.
        
        The original ECG signal may have low-frequency drifts or high-frequency noise, which can make the QRS complex less visible. After filtering:
        
        - **Baseline Wander Removal**: The filter removes slow fluctuations (baseline wander), which makes the QRS complex clearer and easier to detect.
        - **Noise Reduction**: The high-frequency noise (such as powerline interference) is also removed, allowing for a more accurate representation of the heart's electrical activity.
        
        As a result, the **QRS complex** will be more pronounced, improving visibility and making it easier to analyze for heart rate estimation or other purposes.
        """)

        # Provide an option to download the filtered ECG data
        filtered_df = pd.DataFrame({"Time": ecg_data.iloc[:, 0], "Filtered ECG Signal": filtered_ecg})
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Filtered ECG Data as CSV",
            data=csv_data,
            file_name="filtered_ecg.csv",
            mime="text/csv",
            help="Download the filtered ECG signal data"
        )

# Add help text and instructions
else:
    st.info("Please upload an ECG CSV file to proceed.")
