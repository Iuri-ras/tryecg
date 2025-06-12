import streamlit as st
import pandas as pd
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

# Upload file
uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])

if uploaded_file is not None:
    # Load ECG data from the uploaded file
    ecg_data = pd.read_csv(uploaded_file)

    # Check if the file has the expected format and columns
    if ecg_data.shape[1] < 1:
        st.error("The uploaded file doesn't contain any data.")
    else:
        # Extract the first column as the ECG signal
        ecg_signal = ecg_data.iloc[:, 0]

        # Set the cutoff frequencies for the bandpass filter
        lowcut = 0.5
        highcut = 40.0

        # Apply the bandpass filter
        filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

        # Plot the original ECG signal separately
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(ecg_signal)
        ax1.set_title('Original ECG Signal')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Amplitude')
        
        # Display the original ECG signal plot
        st.pyplot(fig1)

        # Plot the filtered ECG signal separately
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(filtered_ecg)
        ax2.set_title('Filtered ECG Signal (0.5–40 Hz Bandpass)')
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Amplitude')

        # Display the filtered ECG signal plot
        st.pyplot(fig2)
else:
    st.info("Please upload an ECG CSV file to proceed.")
