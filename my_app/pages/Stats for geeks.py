import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load your data
ipc_year_df = pd.read_csv(r"C:\Users\Shriya Deshpande\Downloads\ipc_crimes_by_year.csv")
category_df = pd.read_csv(r"C:\Users\Shriya Deshpande\Downloads\cleaned_file.csv")
statewise_df = pd.read_csv(r"C:\Users\Shriya Deshpande\Downloads\statewise_ipc_crimes_2020_2022.csv")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Year-wise IPC Crimes", "Crimes by Category","Distribution of Total Categories" ,"Statewise Crimes"])

with tab1:
    st.header("Year-wise IPC Crimes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=ipc_year_df, x="Year", y="IPC_Crimes", marker="o", ax=ax)
    ax.set_title("IPC Crimes Over the Years")
    ax.set_ylabel("Number of Crimes")
    st.pyplot(fig)

with tab2:
    st.header("Crime by Category (2020-2022)")

    # Reshape the DataFrame: melt only the "Cases" columns
    melted_df = pd.melt(
        category_df,
        id_vars=["Crime Head"],
        value_vars=["2020 Cases", "2021 Cases", "2022 Cases"],
        var_name="Year",
        value_name="Cases"
    )

    # Extract only the year part (e.g., "2020" from "2020 Cases")
    melted_df["Year"] = melted_df["Year"].str.extract(r"(\d{4})")

    # Instead of matplotlib pie chart, use Plotly
    selected_year = st.selectbox("Select Year", sorted(melted_df["Year"].unique()))
    filtered_df = melted_df[melted_df["Year"] == selected_year]
    
    st.subheader(f"Crime Statistics for {selected_year}")
    st.dataframe(filtered_df)
    
    # Create interactive pie chart with Plotly
    st.subheader("Crime Cases by Type")
    fig = px.pie(
        filtered_df,
        values="Cases",
        names="Crime Head",
        title=f"Crime Cases Distribution in {selected_year}",
        hover_data=["Cases"]  # Show cases in hover data
    )
    
    # Improve layout
    fig.update_traces(textposition='inside', textinfo='percent')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# 1. Create pie chart for categories containing "(Total)"
with tab3:
    st.header("Distribution of Total Categories")
    
    # Filter to only include categories with "(Total)" in the name
    total_categories_df = category_df[category_df["Crime Head"].str.contains("(Total)", case=False)]
    
    # Reshape the DataFrame for visualization
    melted_totals = pd.melt(
        total_categories_df,
        id_vars=["Crime Head"],
        value_vars=["2020 Cases", "2021 Cases", "2022 Cases"],
        var_name="Year",
        value_name="Cases"
    )
    melted_totals["Year"] = melted_totals["Year"].str.extract(r"(\d{4})")
    
    # Select year for pie chart
    selected_year_totals = st.selectbox("Select Year for Total Categories", 
                                        sorted(melted_totals["Year"].unique()),
                                        key="total_categories_year")
    
    # Filter data for selected year
    filtered_totals = melted_totals[melted_totals["Year"] == selected_year_totals]
    
    # Create pie chart using Plotly for better interactivity
    fig_totals = px.pie(
        filtered_totals,
        values="Cases",
        names="Crime Head",
        title=f"Distribution of Total Categories ({selected_year_totals})",
        hover_data=["Cases"]
    )
    fig_totals.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_totals, use_container_width=True)


with tab4:
    st.header("Statewise IPC Crimes")

    # Exclude total rows from the original DataFrame
    statewise_df_cleaned = statewise_df[~statewise_df["State/UT"].isin(["TOTAL ALL INDIA", "TOTAL STATE(S)", "TOTAL UT(S)", "UNION TERRITORIES:"])]
    
    # Melt the cleaned dataframe to reshape year columns
    melted = pd.melt(
        statewise_df_cleaned,  # Use the cleaned DataFrame here
        id_vars=["State/UT"],
        value_vars=["2020", "2021", "2022"],
        var_name="Year",
        value_name="Total Crimes"
    )

    # Merge static columns (like 2022 crime rate)
    static_columns = statewise_df_cleaned[["State/UT", "Rate of Cognizable Crimes (IPC) (2022)"]].drop_duplicates()
    melted = pd.merge(melted, static_columns, on="State/UT", how="left")

    # Now use the melted data
    selected_year = st.selectbox("Select Year", melted["Year"].unique(), key="state_tab")
    filtered = melted[melted["Year"] == selected_year]

    # Plotting the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=filtered.sort_values("Total Crimes", ascending=False),
        x="State/UT",
        y="Rate of Cognizable Crimes (IPC) (2022)",
        ax=ax
    )
    ax.set_title(f"Statewise IPC Crime Rate in {selected_year}")
    ax.set_xlabel("State/UT")
    ax.set_ylabel("Rate of Cognizable Crimes (IPC) (2022)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)
