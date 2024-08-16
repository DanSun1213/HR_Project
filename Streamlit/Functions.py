import pandas as pd

def read_hr_data(file_path):
    return pd.read_csv(file_path)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_education(df: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(14, 8))
    
    education_order = ['Below College', 'College', 'Bachelor', 'Master', 'Doctor']
    education_counts = df['Education_Category'].value_counts()
    education_counts = education_counts.reindex(education_order)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(education_counts)))
    
    ax = sns.barplot(x=education_counts.index, y=education_counts.values, palette=colors, order=education_order)
    ax.set_title('Distribution of Education Levels', fontsize=20, pad=20)
    ax.set_xlabel('Education Level', fontsize=14)
    ax.set_ylabel('Number of Employees', fontsize=14)
    
    for i, v in enumerate(education_counts.values):
        ax.text(i, v + 5, f'{v}\n({v/len(df):.1%})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    total = len(df)
    most_common = education_counts.idxmax()
    least_common = education_counts.idxmin()
    stats_text = f'Total Employees: {total}\nMost Common: {most_common}\nLeast Common: {least_common}'
    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    x = range(len(education_order))
    y = education_counts.values
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", linewidth=2)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()

def visualize_job_involvement(df: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(14, 8))
    
    involvement_order = ['Low', 'Medium', 'High', 'Very High']
    involvement_counts = df['JobInvolvement_Category'].value_counts()
    involvement_counts = involvement_counts.reindex(involvement_order)
    
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(involvement_counts)))
    
    ax = sns.barplot(x=involvement_counts.index, y=involvement_counts.values, palette=colors, order=involvement_order)
    ax.set_title('Distribution of Job Involvement', fontsize=20, pad=20)
    ax.set_xlabel('Involvement Level', fontsize=14)
    ax.set_ylabel('Number of Employees', fontsize=14)
    
    for i, v in enumerate(involvement_counts.values):
        ax.text(i, v + 5, f'{v}\n({v/len(df):.1%})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    most_common = involvement_counts.idxmax()
    least_common = involvement_counts.idxmin()
    stats_text = f'Total Employees: {len(df)}\nMost Common: {most_common}\nLeast Common: {least_common}'
    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    x = range(len(involvement_order))
    y = involvement_counts.values
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", linewidth=2)
    
    plt.tight_layout()
    
    return plt.gcf()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition_rate(df):
    plt.figure(figsize=(10, 8))
    attrition_rate = df['Attrition'].value_counts(normalize=True)
    colors = ['#ff9999', '#66b3ff']
    explode = (0, 0.1)
    
    plt.pie(attrition_rate.values, explode=explode, labels=attrition_rate.index, colors=colors,
            autopct='%1.1f%%', startangle=90, shadow=True, textprops={'fontsize': 14})
    
    plt.title('Overall Attrition Rate', fontsize=18)
    plt.axis('equal')
    plt.legend(title="Attrition", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
    return plt.gcf()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Correlation Matrix of Numeric Features', fontsize=16)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.tight_layout()
    return plt.gcf()

def plot_demographic_breakdown(df):
    demographic_cols = ['Age', 'Gender', 'MaritalStatus', 'Education', 'EducationField']
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    axes = axes.flatten()
    
    for i, col in enumerate(demographic_cols):
        attrition_by_demo = df.groupby(col)['Attrition'].value_counts(normalize=True).unstack()
        attrition_by_demo.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'Attrition Rate by {col}', fontsize=14)
        axes[i].set_ylabel('Percentage', fontsize=12)
        axes[i].legend(title='Attrition', loc='upper right')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_salary_analysis(df):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Attrition', y='MonthlyIncome', data=df)
    plt.title('Monthly Income Distribution by Attrition', fontsize=16)
    plt.ylabel('Monthly Income', fontsize=12)
    plt.xlabel('Attrition', fontsize=12)
    return plt.gcf()

def plot_job_satisfaction(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
    plt.title('Job Satisfaction vs Attrition', fontsize=16)
    plt.xlabel('Job Satisfaction', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition')
    return plt.gcf()

def plot_years_at_company(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Attrition', y='YearsAtCompany', data=df)
    plt.title('Years at Company vs Attrition', fontsize=16)
    plt.xlabel('Attrition', fontsize=12)
    plt.ylabel('Years at Company', fontsize=12)
    return plt.gcf()

def initial_hr_analysis(df):
    plt.style.use('default')
    
    plots = [
        plot_attrition_rate(df),
        plot_correlation_matrix(df),
        plot_demographic_breakdown(df),
        plot_salary_analysis(df),
        plot_job_satisfaction(df),
        plot_years_at_company(df)
    ]
    
    return plots

def apply_filters(df):
    import streamlit as st
    st.sidebar.header("Filters")

    # Department filter
    department = st.sidebar.multiselect("Department", df['Department'].unique())

    # Gender filter
    gender = st.sidebar.multiselect("Gender", df['Gender'].unique())

    # Education filter
    education = st.sidebar.multiselect("Education", df['Education'].unique())

    # Apply filters
    if department:
        df = df[df['Department'].isin(department)]
    
    if gender:
        df = df[df['Gender'].isin(gender)]
    
    if education:
        df = df[df['Education_Category'].isin(education)]

    return df

def plot_employee_attrition(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    
    # Visualization to show Employee Attrition in Counts
    attrition_rate = df["Attrition"].value_counts()
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='Set2', ax=ax1)
    ax1.set_title("Employee Attrition Counts", fontweight="black", size=14, pad=15)
    for i, v in enumerate(attrition_rate.values):
        ax1.text(i, v, v, ha="center", fontsize=14)

    # Visualization to show Employee Attrition in Percentage
    colors = sns.color_palette('Set2', len(attrition_rate))
    ax2.pie(attrition_rate, labels=["No", "Yes"], autopct="%.2f%%", textprops={"size":14},
            colors=colors, explode=[0, 0.1], startangle=90)
    center_circle = plt.Circle((0, 0), 0.3, fc='white')
    ax2.add_artist(center_circle)
    ax2.set_title("Employee Attrition Rate", fontweight="black", size=14, pad=15)

    plt.tight_layout()
    return fig


def pie_bar_plot(df, col, hue):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract value counts for the specified column
    value_counts = df[col].value_counts().sort_index()
    
    # First subplot: Pie chart
    ax1.set_title(f"Distribution by {col}", fontweight="black", size=14, pad=15)
    colors = sns.color_palette('Set2', len(value_counts))
    wedges, texts, autotexts = ax1.pie(value_counts.values, labels=value_counts.index, 
                                       autopct="%.1f%%", pctdistance=0.75, startangle=90,
                                       colors=colors, textprops={"size":14})
    center_circle = plt.Circle((0, 0), 0.4, fc='white')
    ax1.add_artist(center_circle)
    
    # Second subplot: Bar plot
    new_df = df[df[hue] == 'Yes']
    value_1 = value_counts
    value_2 = new_df[col].value_counts().sort_index()  # Sort the values in the same order
    attrition_percentages = np.floor((value_2 / value_1) * 100).values
    
    sns.barplot(x=value_2.index, y=value_2.values, palette='Set2', ax=ax2)
    ax2.set_title(f"Attrition Rate by {col}", fontweight="black", size=14, pad=15)
    
    for index, value in enumerate(value_2):
        ax2.text(index, value, f"{value} ({int(attrition_percentages[index])}%)", 
                 ha="center", va="bottom", size=10)
    
    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def hist_with_hue(df, col, hue):


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6))
    
    # Histogram
    sns.histplot(x=col, hue=hue, data=df, kde=True, palette='Set2', ax=ax1)
    
    # Configure the x-axis to display integer values and center-align the labels
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xticklabels(ax1.get_xticks(), rotation=90, ha='center')
    ax1.set_title(f"Distribution by {col}", fontweight="black", size=14, pad=10)
    
    # Box plot
    sns.boxplot(x=hue, y=col, data=df, palette='Set2', ax=ax2)
    ax2.set_title(f"Distribution by {col} & {hue}", fontweight="black", size=14, pad=10)
    
    plt.tight_layout()
    return fig

def count_percent_plot(df, col, hue):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 8))

    # First subplot: Employees by column
    value_1 = df[col].value_counts()
    sns.barplot(x=value_1.index, y=value_1.values, order=value_1.index, palette='Set2', ax=ax1)
    ax1.set_title(f"Employees by {col}", fontweight="black", size=14, pad=15)
    for index, value in enumerate(value_1.values):
        count_percentage = "{:.1f}%".format((value / len(df)) * 100)
        ax1.text(index, value, f"{value} ({count_percentage})", ha="center", va="bottom", size=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    # Sort the values for the second subplot to match the order of the first subplot
    value_2 = df[df[hue] == 'Yes'][col].value_counts().reindex(value_1.index)

    # Second subplot: Employee Attrition by column
    attrition_rate = (value_2 / value_1 * 100).values
    sns.barplot(x=value_2.index, y=value_2.values, order=value_1.index, palette='Set2', ax=ax2)
    ax2.set_title(f"Employee Attrition by {col}", fontweight="black", size=14, pad=15)
    for index, value in enumerate(value_2.values):
        attrition_percentage = "{:.1f}%".format(np.round(attrition_rate[index], 1))
        ax2.text(index, value, f"{value} ({attrition_percentage})", ha="center", va="bottom", size=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    plt.tight_layout()
    return fig
def plot_years_at_company_vs_attrition(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, ax=ax)
    
    ax.set_title('Years at Company vs Attrition', fontsize=16)
    ax.set_xlabel('Attrition', fontsize=12)
    ax.set_ylabel('Years at Company', fontsize=12)
    
    # Add median values on top of each box
    for i, artist in enumerate(ax.artists):
        # The box extends from the first quartile to the third quartile
        # The median is represented by a line inside the box
        median = df[df['Attrition'] == (['No', 'Yes'][i])]['YearsAtCompany'].median()
        ax.text(i, median, f'Median: {median:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# Function to create the visualization
def create_attrition_plot(df):
    plt.figure(figsize=(10, 20))
    attrition_by_role = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack()
    attrition_by_role = attrition_by_role.sort_values('Yes', ascending=False)
    ax = attrition_by_role.plot(kind='bar', stacked=True, width=0.8)
    
    plt.title('Attrition Rate by Job Role', fontsize=20, pad=20)
    plt.xlabel('Job Role', fontsize=14, labelpad=10)
    plt.ylabel('Percentage', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Attrition', fontsize=8, title_fontsize=10, loc='upper left', bbox_to_anchor=(1, 0.1))
    
    for i, role in enumerate(attrition_by_role.index):
        yes_rate = attrition_by_role.loc[role, 'Yes'] * 100
        plt.text(i, 1.02, f'{yes_rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    summary_text = (f"Total Employees: {len(df)}\n"
                    f"Overall Attrition Rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%\n"
                    f"Highest Attrition: {attrition_by_role['Yes'].idxmax()} ({attrition_by_role['Yes'].max()*100:.1f}%)\n"
                    f"Lowest Attrition: {attrition_by_role['Yes'].idxmin()} ({attrition_by_role['Yes'].min()*100:.1f}%)")
    
    plt.text(1.1, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return plt

# Function to get AI insights
def get_ai_insights(df):
    import requests
    import socks
    import socket

# Configure SOCKS5 proxy
    socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    socket.socket = socks.socksocket

    data_json = df.to_json(orient='records')
    
    api_key = st.secrets["openai_api_key"]  # Store your API key in Streamlit secrets
    url = 'https://api.openai.com/v1/chat/completions'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-4',
        'messages': [
            {
                "role": "system",
                "content": """You are an experienced data scientist specializing in HR analytics. 
                You've been asked to analyze employee attrition data for a company. 
                In the data, 'Attrition' column values mean:
                - 'Yes': The employee has left the company
                - 'No': The employee is still active with the company
                Your task is to provide insights on employee attrition rates, identify patterns or factors 
                contributing to attrition, and suggest actionable recommendations to improve employee 
                retention and overall company performance."""
            },
            {
                "role": "user",
                "content": f"""Here's the employee attrition data for analysis:
                {data_json}
                
                Based on this data, please provide:
                1. An overview of the attrition rates across different job roles.
                2. Insights into potential factors contributing to higher attrition in certain roles.
                3. Identification of any patterns or trends in the data related to attrition.
                4. Actionable recommendations to reduce attrition and improve employee retention.
                5. Suggestions on how these changes could positively impact company performance.
                
                Please structure your response with clear headings for each of these points."""
            }
        ],
        'max_tokens': 500,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        insights = result['choices'][0]['message']['content']
        return insights
    except Exception as e:
        return f"An error occurred: {e}"

def create_attrition_plot1(df):
    plt.figure(figsize=(10, 20))
    
    # Calculate attrition rates
    attrition_by_role = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack()
    attrition_by_role = attrition_by_role.sort_values('Yes', ascending=False)
    
    # Create the stacked bar plot
    ax = attrition_by_role.plot(kind='bar', stacked=True, width=0.8)
    
    plt.title('Attrition Rate by Job Role', fontsize=20, pad=20)
    plt.xlabel('Job Role', fontsize=14, labelpad=10)
    plt.ylabel('Percentage', fontsize=14, labelpad=10)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust legend
    plt.legend(title='Attrition', fontsize=8, title_fontsize=10, loc='upper left', bbox_to_anchor=(1, 0.1))
    
    # Add percentage labels
    for i, role in enumerate(attrition_by_role.index):
        yes_rate = attrition_by_role.loc[role, 'Yes'] * 100
        plt.text(i, 1.02, f'{yes_rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Add summary text
    summary_text = (f"Total Employees: {len(df)}\n"
                    f"Overall Attrition Rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%\n"
                    f"Highest Attrition: {attrition_by_role['Yes'].idxmax()} ({attrition_by_role['Yes'].max()*100:.1f}%)\n"
                    f"Lowest Attrition: {attrition_by_role['Yes'].idxmin()} ({attrition_by_role['Yes'].min()*100:.1f}%)")
    
    plt.text(1.1, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return plt

import streamlit as st
import requests
import base64
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import socks
import socket

# Configure SOCKS5 proxy
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
socket.socket = socks.socksocket

def create_pyplot_image():
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Sine Wave")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def encode_image(image_buffer):
    return base64.b64encode(image_buffer.getvalue()).decode('utf-8')

def send_image_to_openai(image_buffer, api_key):
    socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    socket.socket = socks.socksocket
    api_key = st.secrets["openai_api_key"]
    base_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base64_image = encode_image(image_buffer)
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this pyplot image. Describe what you see, including the type of plot, axes labels, and any notable features."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}




