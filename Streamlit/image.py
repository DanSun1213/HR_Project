import streamlit as st
import pandas as pd
import Functions as fn
import plotly.express as px
import plotly.graph_objects as go
import sys
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
import Functions as fn

def main():

    st.title("Attrition Plot Analysis with OpenAI API")
    st.markdown("<hr style='border: 2px solid #f0f2f6;'>", unsafe_allow_html=True)
    
    # Read the data (you'll need to adjust this path)
    df = fn.read_hr_data('/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv')

    # Generate the plot
    image_buffer = fn.create_attrition_plot1(df)

    # Display the plot
    st.image(image_buffer, caption="Attrition Rate by Job Role", use_column_width=True)

    # Button to trigger OpenAI analysis
    if st.button("Analyze Image with OpenAI", key="analyze_image"):
        api_key = st.secrets["openai_api_key"]
        with st.spinner("Analyzing image..."):
            result = fn.send_image_to_openai1(image_buffer, api_key)
        
        if "error" in result:
            st.error(f"An error occurred: {result['error']}")
        else:
            analysis = result['choices'][0]['message']['content']
            st.subheader("OpenAI Analysis:")
            st.write(analysis)

if __name__ == "__main__":
    main()
    