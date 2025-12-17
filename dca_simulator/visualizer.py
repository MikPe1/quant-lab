
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_simulation_results(df: pd.DataFrame):
    """
    Generates interactive Plotly charts for Portfolio Value vs. Total Invested and Profit/Loss over time.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results with columns like 'Date', 'Portfolio Value',
                          'Total Invested', and 'Profit/Loss'.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the two subplots.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display for charts.",
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=20))
        fig.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
        return fig

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Portfolio Value vs. Total Invested", "Profit/Loss Over Time"))

    # Chart 1: Portfolio Value vs. Total Invested
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Portfolio Value'], mode='lines',
                             name='Portfolio Value', line=dict(color='green')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Total Invested'], mode='lines',
                             name='Total Invested', line=dict(color='blue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Buy&Hold Value'], mode='lines',
                             name='Buy&Hold Value', line=dict(color='red')),
                  row=1, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)

    # Chart 2: Profit/Loss Over Time
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Profit/Loss'], mode='lines',
                             name='Profit/Loss', line=dict(color='purple')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Buy&Hold Profit/Loss'], mode='lines',
                             name='Buy&Hold Profit/Loss', line=dict(color='orange')),
                  row=2, col=1)
    fig.update_yaxes(title_text="Profit/Loss ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(height=700, title_text="DCA Simulation Results", hovermode="x unified")
    return fig

def display_results_table(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=[
            'Date', 'Price', 'Shares Bought', 'Total Shares',
            'Total Invested', 'Portfolio Value', 'Profit/Loss',
            'Buy&Hold Value', 'Buy&Hold Profit/Loss', 'Dip Triggered'
        ])

    df_display = df.copy()

    currency_cols = [
        'Price', 'Shares Bought',
        'Total Invested', 'Portfolio Value', 'Profit/Loss',
        'Buy&Hold Value', 'Buy&Hold Profit/Loss'
    ]

    for col in currency_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )

    if 'Total Shares' in df_display.columns:
        df_display['Total Shares'] = df_display['Total Shares'].apply(
            lambda x: f"{x:,.4f}" if pd.notna(x) else ""
        )

    if 'Dip Triggered' in df_display.columns:
        df_display['Dip Triggered'] = df_display['Dip Triggered'].apply(
            lambda x: 'Yes' if x else 'No'
        )

    return df_display

