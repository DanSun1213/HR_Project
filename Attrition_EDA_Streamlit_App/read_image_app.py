import streamlit as st
import pandas as pd
from PIL import Image
import Functions as fn
import io
import matplotlib.pyplot as plt

# Constants
DEFAULT_DATA_PATH = '/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv'

# Page Configuration
st.set_page_config(page_title="HR Analytics Dashboard", page_icon="📊", layout="wide")

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv(DEFAULT_DATA_PATH)

def analyze_image(image_data, api_key, prompt):
    with st.spinner("Analyzing image..."):
        result = fn.send_image_to_openai_1(image_data, api_key, prompt)
    
    if "error" in result:
        st.error(f"An error occurred: {result['error']}")
    else:
        analysis = result['choices'][0]['message']['content']
        st.success("Analysis complete!")
        st.subheader("OpenAI Analysis:")
        st.write(analysis)
        
        st.download_button(
            label="Download Image Analysis",
            data=analysis,
            file_name="image_analysis.txt",
            mime="text/plain"
        )

def main():
    st.title("HR Analytics Dashboard")

    # Load data
    df = load_data()

# Sidebar
    st.sidebar.header("Navigator")
    
    st.sidebar.markdown("""
    ### Explore Key HR Insights
    
    📊 **Main Features:**
    - Attrition Analysis
    - Demographic Insights
    - Job Satisfaction Trends
    - Salary Distribution
    - Education Impact
    
    🔍 **Special Feature:**  
    AI-powered image analysis for deeper insights!
    
    ---
    
    📌 **How to Use:**  
    Navigate through tabs for different analyses.
    
    🗃️ **Data Source:**  
    IBM HR Analytics Employee Attrition & Performance
    """)
    
    st.sidebar.caption("Created by Dan Sun, HR Data Analyst")
    
    # Social Links
    st.sidebar.markdown("### Connect with Me")
    col1, col2 = st.sidebar.columns(2)
    col1.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/DanSun1213)")
    col2.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dan-sun-9b3315186/)")

    # Main content of your dashboard continues here...

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Exploratory Data Analysis (EDA)", "Dashboard Image Analysis", "Upload Image Analysis", "Data Overview"])

    # Tab 1: HR Analysis (Your existing code remains the same)
    with tab1:

        # st.subheader("Demographic Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig_age = fn.hist_with_hue(df, 'Age', 'Attrition')
            st.pyplot(fig_age)
        
        with col2:
            st.subheader("Education Attrition")
            fig = fn.count_percent_plot(df, 'EducationField', 'Attrition')
            st.pyplot(fig)


        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Marital Status Attrition")
            gender = fn.pie_bar_plot(df, 'MaritalStatus', 'Attrition')
            st.pyplot(gender)           


        with col4:
            st.subheader("Job Satisfaction Attrition")
            fig_satisfaction = fn.pie_bar_plot(df, 'JobSatisfaction', 'Attrition')
            st.pyplot(fig_satisfaction)

        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Job Role Attirtion")
        # Attrition Rate by Job Role
        # st.subheader("Attrition Rate by Job Role")
            image_buffer = fn.create_attrition_plot1(df)
            st.image(image_buffer, use_column_width=True)

        with col6:
            st.subheader("Monthly Income Attirtion")
            salary = fn.plot_salary_analysis(df)
            st.pyplot(salary)
        # Your existing code for HR Analysis...

    # Tab 2: Dashboard Image Analysis
    with tab2:
        st.header("Dashboard Image Analysis")
        image_selection = st.selectbox("Select an image to analyze:", 
                                       ["Age Distribution", 
                                        "Education Attrition",
                                        "Marital Status Attrition",
                                        "Job Satisfaction Attrition",
                                        "Job Role Attrition",
                                        "Monthly Income Attrition"])
        
        if image_selection == "Age Distribution":
            st.pyplot(fig_age)
            image_to_analyze = fn.fig_to_img_buffer(fig_age)
        elif image_selection == "Education Attrition":
            st.pyplot(fig)
            image_to_analyze = fn.fig_to_img_buffer(fig)
        elif image_selection == "Marital Status Attrition":
            st.pyplot(gender)
            image_to_analyze = fn.fig_to_img_buffer(gender)
        elif image_selection == "Job Satisfaction Attrition":
            st.pyplot(fig_satisfaction)
            image_to_analyze = fn.fig_to_img_buffer(fig_satisfaction)
        elif image_selection == "Job Role Attrition":
            st.image(image_buffer, caption="Attrition Rate by Job Role", use_column_width=True)
            image_to_analyze = image_buffer
        else:  # Monthly Income Attrition
            st.pyplot(salary)
            image_to_analyze = fn.fig_to_img_buffer(salary)

        if st.button("Analyze Dashboard Image"):
            api_key = st.secrets["openai_api_key"]
            prompt = fn.get_data_analysis_prompt()
            analyze_image(image_to_analyze, api_key, prompt)

    # Tab 3: Upload Image Analysis
    with tab3:
        st.header("Upload and Analyze Your Own Image")
        uploaded_image = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            if st.button("Analyze Uploaded Image"):
                api_key = st.secrets["openai_api_key"]
                prompt = fn.get_data_analysis_prompt()
                analyze_image(img_byte_arr, api_key, prompt)
        else:
            st.warning("Please upload an image to analyze.")

# Tab 4: Data Overview
    with tab4:
        st.header("Data Overview")
        
        # Dataset Information
        st.subheader("About the Dataset")
        st.info("""
        This is a fictional dataset created by IBM data scientists to explore factors that lead to employee attrition. 
        
        With this data, you can explore important questions such as:
        - Show me a breakdown of distance from home by job role and attrition
        - Compare average monthly income by education and attrition
        
        Source: [IBM HR Analytics Employee Attrition & Performance on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)
        """)
        
        st.write("Dataset Preview:")
        st.write(df.head())
        st.write(f"Total records: {len(df)}")

        # Column Selection
        st.subheader("Select columns to view:")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Choose columns", all_columns, default=all_columns[:5])

        # Display selected columns
        if selected_columns:
            st.write(df[selected_columns].head())
        else:
            st.warning("Please select at least one column to view the data.")

        # Download Options
        st.subheader("Download Data")
        download_option = st.radio("Choose columns to download:", ("All Columns", "Selected Columns Only"))
        
        if download_option == "All Columns":
            csv = df.to_csv(index=False).encode('utf-8')
            file_name = "ibm_hr_analytics_attrition.csv"
        else:
            if selected_columns:
                csv = df[selected_columns].to_csv(index=False).encode('utf-8')
                file_name = "ibm_hr_analytics_attrition_selected.csv"
            else:
                st.warning("Please select at least one column to download the data.")
                st.stop()

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=file_name,
            mime="text/csv",
        )

        # Dataset Statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of records: {len(df)}")
            st.write(f"Number of columns: {len(df.columns)}")
        with col2:
            st.write(f"Number of numerical columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
            st.write(f"Number of categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
        
        st.write("Column names:", ", ".join(df.columns))
        
        # Basic Statistics
        st.subheader("Basic Statistics for Numerical Columns")
        st.write(df.describe())

        # Additional Dataset Exploration Ideas
        st.subheader("Explore the Dataset")
        st.write("Here are some ideas to explore the dataset:")
        exploration_ideas = [
            "Analyze the relationship between overtime and attrition",
            "Investigate how job satisfaction correlates with years at the company",
            "Examine the impact of work-life balance on attrition rates",
            "Compare attrition rates across different departments",
            "Analyze how performance ratings relate to monthly income"
        ]
        for idea in exploration_ideas:
            st.write(f"- {idea}")

if __name__ == "__main__":
    main()
