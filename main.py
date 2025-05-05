
# main.py

import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')     # headless backend for server
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

app = FastAPI()
CSV_PATH = 'mc.csv'


def load_and_prepare():
    # 1) Load & parse dates
    df = pd.read_csv(CSV_PATH, parse_dates=['CloseDate'], low_memory=False)

    # 2) Add PropertyID if missing
    if 'PropertyID' not in df.columns:
        df.insert(0, 'PropertyID', [f"A{i}" for i in range(1, len(df) + 1)])

    # 3) Clean ClosePrice
    df['ClosePrice'] = pd.to_numeric(df['ClosePrice'], errors='coerce')
    df = df.dropna(subset=['ClosePrice'])

    # 4) Prepare for grouping/plotting
    df['YearMonth'] = df['CloseDate'].dt.to_period('M').astype(str)
    df['MonthStart'] = pd.to_datetime(df['YearMonth'] + '-01')

    return df

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/plot")
def plot(ids: str = Query(
    ..., 
    description="Comma-separated list of exactly three PropertyID values, e.g. A1,A25,A100"
)):
    # Parse & validate user IDs
    ids_list = [pid.strip() for pid in ids.split(',')]
    if len(ids_list) != 3:
        raise HTTPException(400, "Please provide exactly three PropertyIDs (comma-separated).")

    # Load & prep data
    df = load_and_prepare()

    # Compute monthly averages
    monthly_avg = (
        df
        .groupby('MonthStart', as_index=False)['ClosePrice']
        .mean()
        .sort_values('MonthStart')
    )
    last_x = monthly_avg['MonthStart'].iloc[-1]
    last_y = monthly_avg['ClosePrice'].iloc[-1]

    # Extract the sold‑points for the given IDs
    sold = df[df['PropertyID'].isin(ids_list)].copy()
    if sold.empty:
        raise HTTPException(404, "None of the provided PropertyIDs were found.")
    sold['MonthStart'] = sold['CloseDate'].dt.to_period('M').dt.to_timestamp()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        monthly_avg['MonthStart'],
        monthly_avg['ClosePrice'],
        marker='o',
        label='Monthly Avg'
    )
    plt.scatter(
        sold['MonthStart'],
        sold['ClosePrice'],
        color='red',
        zorder=5,
        label='Selected Sales'
    )

    # Draw dotted lines + annotate with flipped sign logic
    for _, row in sold.iterrows():
        # Connector line
        plt.plot(
            [row['MonthStart'], last_x],
            [row['ClosePrice'], last_y],
            linestyle=':',
            color='gray',
            linewidth=1
        )
        # Compute raw percentage diff = (current_avg – sold_price) / sold_price * 100
        raw_diff = (last_y - row['ClosePrice']) / row['ClosePrice'] * 100
        # If current > sold_price → raw_diff > 0 → show a “–” for negative
        # If current < sold_price → raw_diff < 0 → show a “+” for positive
        sign = '–' if raw_diff > 0 else '+'
        pct = abs(raw_diff)
        plt.annotate(
            f"{row['PropertyID']} ({sign}{pct:.1f}%)",
            xy=(row['MonthStart'], row['ClosePrice']),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center'
        )

    # Styling
    plt.title('Avg Close Price by Month with Selected Sales → Latest Avg')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Add horizontal margin so it’s not overly zoomed
    ax = plt.gca()
    ax.margins(x=0.05)
    ax.figure.autofmt_xdate()

    # Save to buffer & return as PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

