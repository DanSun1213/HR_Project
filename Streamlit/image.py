import streamlit as st
import pandas as pd
import Functions as fn
import plotly.express as px
import plotly.graph_objects as go
import sys


def main():
    st.title("PyPlot Image Analysis with OpenAI Vision API")

    # Generate the plot
    image_buffer = fn.create_pyplot_image()

    # Display the plot
    st.image(image_buffer, caption="Generated Sine Wave Plot", use_column_width=True)

    # Button to trigger OpenAI analysis
    if st.button("Analyze Image with OpenAI", key="analyze_image"):
        api_key = st.secrets["openai_api_key"]
        with st.spinner("Analyzing image..."):
            result = fn.send_image_to_openai(image_buffer, api_key)
        
        if "error" in result:
            st.error(f"An error occurred: {result['error']}")
        else:
            analysis = result['choices'][0]['message']['content']
            st.subheader("OpenAI Analysis:")
            st.write(analysis)

if __name__ == "__main__":
    main()