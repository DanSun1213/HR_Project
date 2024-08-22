
def hist_with_hue(df, col, hue):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker


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

def plot_years_at_company_vs_attrition(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
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
