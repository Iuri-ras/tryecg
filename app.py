import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Define the bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Sampling frequency (assuming 100 Hz, typical for ECG data)
fs = 100.0

# Streamlit app
st.title("ECG Signal Filtering")

# Sidebar options
st.sidebar.title("Input Options")

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
load_sample = st.sidebar.button("Load Sample Data")

# Initialize the dataframe for holding the ECG data
df = None

# Load sample data if the button is pressed
if load_sample:
    # Generate a sample ECG-like signal (sine wave with noise)
    time = np.linspace(0, 10, 1000)
    ecg_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.random.randn(1000)
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

# Process data if available
if df is not None:
    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    # Apply the bandpass filter
    lowcut = 0.5
    highcut = 40.0
    filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

    # Plot the original ECG signal
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time, ecg_signal)
    ax1.set_title('Original ECG Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    st.pyplot(fig1)

    # Plot the filtered ECG signal
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(time, filtered_ecg)
    ax2.set_title('Filtered ECG Signal (0.5–40 Hz Bandpass)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    st.pyplot(fig2)
else:
    st.info("Please upload an ECG CSV file or load sample data to proceed.")
