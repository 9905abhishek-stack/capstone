"""
╔═══════════════════════════════════════════════════════════╗
║   Interactive Dashboard — Time Series Capstone 2026       ║
║   Built with Plotly Dash + Bootstrap dark theme           ║
║                                                           ║
║   Run: python dashboard/app.py                            ║
║   Open: http://localhost:8050                             ║
╚═══════════════════════════════════════════════════════════╝
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ──────────────────────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'dashboard_data.pkl')

with open(DATA_PATH, 'rb') as f:
    D = pickle.load(f)

close_df = D['close_df']
returns_df = D['returns_df']
test_data = D['test_data']
forecast_details = D['forecast_details']
allocation = D['allocation']
garch_diag = D['garch_diagnostics']
trend_assessments = D['trend_assessments']
all_metrics = D['all_metrics']
sector_map = D['sector_map']
ticker_sectors = D['ticker_sectors']

TICKERS = list(close_df.columns)
MODELS = list(all_metrics.keys())

# Color palette
PALETTE = {
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'accent1': '#4ECDC4',
    'accent2': '#FFE66D',
    'accent3': '#AA96DA',
    'bg_dark': '#0d1117',
    'bg_card': '#161b22',
    'bg_card2': '#1c2333',
    'border': '#30363d',
    'text': '#c9d1d9',
    'text_dim': '#8b949e',
}

STOCK_COLORS = [
    '#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
    '#A8E6CF', '#FF8B94', '#F38181', '#AA96DA', '#FCBAD3',
    '#6C5B7B', '#F67280',
]

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=PALETTE['bg_card'],
        plot_bgcolor=PALETTE['bg_card'],
        font=dict(color=PALETTE['text'], family='Inter, sans-serif'),
        xaxis=dict(gridcolor=PALETTE['border'], zerolinecolor=PALETTE['border']),
        yaxis=dict(gridcolor=PALETTE['border'], zerolinecolor=PALETTE['border']),
        margin=dict(l=50, r=30, t=50, b=40),
    )
)

# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

def make_kpi_card(title, value, subtitle="", color=PALETTE['primary']):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={'fontSize': '0.75rem', 'color': PALETTE['text_dim'],
                                 'marginBottom': '4px', 'textTransform': 'uppercase',
                                 'letterSpacing': '1px'}),
            html.H3(value, style={'color': color, 'fontWeight': '700', 'marginBottom': '2px'}),
            html.P(subtitle, style={'fontSize': '0.75rem', 'color': PALETTE['text_dim'], 'margin': 0}),
        ]),
        style={
            'backgroundColor': PALETTE['bg_card2'],
            'border': f'1px solid {PALETTE["border"]}',
            'borderRadius': '12px',
            'textAlign': 'center',
        },
    )

def short(ticker):
    return ticker.replace('.NS', '')


# ──────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    ],
    title='TSA 2026 — Capstone Dashboard',
    suppress_callback_exceptions=True,
)

# ──────────────────────────────────────────────────────────────
# Navigation
# ──────────────────────────────────────────────────────────────
PAGES = ['Overview', 'Forecasts', 'Volatility', 'Portfolio', 'Models']

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([
            html.Span("📈 ", style={'fontSize': '1.3rem'}),
            html.Span("TSA 2026 ", style={'fontWeight': '700', 'color': PALETTE['primary']}),
            html.Span("Capstone Dashboard", style={'fontWeight': '300'}),
        ], style={'fontSize': '1.1rem'}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink(
                p, id=f'nav-{p.lower()}', href='#', active=p == 'Overview',
                style={'color': PALETTE['text'], 'fontWeight': '500'},
                class_name='px-3',
            )) for p in PAGES
        ], navbar=True),
    ], fluid=True),
    color=PALETTE['bg_card'],
    dark=True,
    style={'borderBottom': f'1px solid {PALETTE["border"]}', 'padding': '8px 0'},
)


# ──────────────────────────────────────────────────────────────
# PAGE: Overview
# ──────────────────────────────────────────────────────────────

def build_overview():
    total_invested = allocation['Amount (₹)'].sum()
    n_stocks = len(TICKERS)
    n_models = len(MODELS)
    n_sectors = len(sector_map)

    # Best model
    avg_mapes = {}
    for model, stocks in all_metrics.items():
        mapes = [m.get('MAPE (%)', np.nan) for m in stocks.values()]
        avg_mapes[model] = np.nanmean(mapes)
    best_model = min(avg_mapes, key=avg_mapes.get)

    # Price chart
    fig_prices = go.Figure()
    for i, t in enumerate(TICKERS):
        fig_prices.add_trace(go.Scatter(
            x=close_df.index, y=close_df[t],
            name=short(t), line=dict(color=STOCK_COLORS[i % len(STOCK_COLORS)], width=1.5),
            hovertemplate=f'{short(t)}: ₹%{{y:.2f}}<extra></extra>',
        ))
    fig_prices.update_layout(
        title='Stock Universe — Daily Close Prices (2021–2025)',
        xaxis_title='Date', yaxis_title='Price (₹)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
        height=420, **PLOTLY_TEMPLATE['layout'],
    )

    return html.Div([
        dbc.Row([
            dbc.Col(make_kpi_card('Stocks Tracked', str(n_stocks), f'{n_sectors} sectors'), md=3),
            dbc.Col(make_kpi_card('Capital', f'₹{total_invested:,.0f}', 'Virtual allocation', PALETTE['accent2']), md=3),
            dbc.Col(make_kpi_card('Models Built', str(n_models), ', '.join(MODELS), PALETTE['accent1']), md=3),
            dbc.Col(make_kpi_card('Best Model', best_model, f'Avg MAPE: {avg_mapes[best_model]:.2f}%', PALETTE['accent3']), md=3),
        ], className='g-3 mb-4'),
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_prices, id='overview-prices')),
                 style={'backgroundColor': PALETTE['bg_card'], 'border': f'1px solid {PALETTE["border"]}',
                        'borderRadius': '12px'}),
    ])


# ──────────────────────────────────────────────────────────────
# PAGE: Forecasts
# ──────────────────────────────────────────────────────────────

def build_forecasts():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Select Stock', style={'fontWeight': '600', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='forecast-stock',
                    options=[{'label': short(t), 'value': t} for t in TICKERS],
                    value=TICKERS[0],
                    style={'backgroundColor': PALETTE['bg_card2'], 'color': '#000'},
                ),
            ], md=4),
            dbc.Col([
                html.Label('Select Model', style={'fontWeight': '600', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='forecast-model',
                    options=[{'label': m, 'value': m} for m in MODELS],
                    value=MODELS[0],
                    style={'backgroundColor': PALETTE['bg_card2'], 'color': '#000'},
                ),
            ], md=4),
        ], className='mb-4'),
        dbc.Card(dbc.CardBody(dcc.Graph(id='forecast-chart')),
                 style={'backgroundColor': PALETTE['bg_card'], 'border': f'1px solid {PALETTE["border"]}',
                        'borderRadius': '12px'}),
        html.Div(id='forecast-metrics', className='mt-3'),
    ])


# ──────────────────────────────────────────────────────────────
# PAGE: Volatility
# ──────────────────────────────────────────────────────────────

def build_volatility():
    # Rolling volatility chart
    fig_vol = go.Figure()
    for i, t in enumerate(TICKERS):
        log_ret = returns_df[t] if t in returns_df else pd.Series()
        roll_vol = log_ret.rolling(21).std().dropna()
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol, name=short(t),
            line=dict(color=STOCK_COLORS[i % len(STOCK_COLORS)], width=1.2),
        ))
    fig_vol.update_layout(
        title='21-Day Rolling Volatility (Std Dev of Log Returns)',
        xaxis_title='Date', yaxis_title='Volatility',
        height=400, **PLOTLY_TEMPLATE['layout'],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
    )

    # Trend assessment table
    trend_rows = []
    for t in TICKERS:
        ta = trend_assessments.get(t, {})
        gd = garch_diag.get(t, {})
        trend_rows.append({
            'Stock': short(t),
            'Sector': ticker_sectors.get(t, '?'),
            'Trend': ta.get('direction', '?'),
            'Slope (%)': ta.get('trend_slope_pct', '?'),
            'GARCH Persistence': round(gd.get('persistence', 0), 3),
            'Avg Daily Vol': f"{gd.get('avg_daily_vol', 0):.4f}",
        })

    return html.Div([
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_vol)),
                 style={'backgroundColor': PALETTE['bg_card'], 'border': f'1px solid {PALETTE["border"]}',
                        'borderRadius': '12px', 'marginBottom': '20px'}),
        dbc.Card(dbc.CardBody([
            html.H5('Trend & Volatility Assessment', style={'color': PALETTE['primary'], 'marginBottom': '12px'}),
            dash_table.DataTable(
                data=trend_rows,
                columns=[{'name': c, 'id': c} for c in trend_rows[0].keys()],
                style_header={'backgroundColor': PALETTE['bg_card2'], 'color': PALETTE['primary'],
                              'fontWeight': '600', 'border': f'1px solid {PALETTE["border"]}'},
                style_cell={'backgroundColor': PALETTE['bg_card'], 'color': PALETTE['text'],
                            'border': f'1px solid {PALETTE["border"]}', 'textAlign': 'center',
                            'padding': '8px', 'fontSize': '0.85rem'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Trend} = UPWARD'}, 'color': '#00D4AA', 'fontWeight': '600'},
                    {'if': {'filter_query': '{Trend} = DOWNWARD'}, 'color': '#FF6B6B', 'fontWeight': '600'},
                ],
            ),
        ]), style={'backgroundColor': PALETTE['bg_card'], 'border': f'1px solid {PALETTE["border"]}',
                   'borderRadius': '12px'}),
    ])


# ──────────────────────────────────────────────────────────────
# PAGE: Portfolio
# ──────────────────────────────────────────────────────────────

def build_portfolio():
    labels = [short(s) for s in allocation['Stock']]
    colors_pie = STOCK_COLORS[:len(labels)]

    fig_pie = go.Figure(go.Pie(
        labels=labels, values=allocation['Weight (%)'],
        hole=0.55, marker=dict(colors=colors_pie, line=dict(color=PALETTE['bg_dark'], width=2)),
        textinfo='label+percent', textfont=dict(size=11),
        hovertemplate='%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>',
    ))
    fig_pie.update_layout(
        title='Portfolio Allocation',
        height=420, **PLOTLY_TEMPLATE['layout'],
        annotations=[dict(text=f'₹10L', x=0.5, y=0.5, font_size=18, showarrow=False,
                          font_color=PALETTE['primary'])],
    )

    # Sector allocation
    sector_alloc = {}
    for _, row in allocation.iterrows():
        sector = ticker_sectors.get(row['Stock'], 'Unknown')
        sector_alloc[sector] = sector_alloc.get(sector, 0) + row['Amount (₹)']

    fig_sector = go.Figure(go.Bar(
        x=list(sector_alloc.keys()),
        y=list(sector_alloc.values()),
        marker_color=[STOCK_COLORS[i] for i in range(len(sector_alloc))],
        text=[f'₹{v:,.0f}' for v in sector_alloc.values()],
        textposition='outside',
    ))
    fig_sector.update_layout(
        title='Capital by Sector',
        yaxis_title='Amount (₹)',
        height=400, **PLOTLY_TEMPLATE['layout'],
    )

    # Correlation heatmap
    corr = returns_df.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=[short(c) for c in corr.columns], y=[short(c) for c in corr.index],
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate='%{text}', textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>',
    ))
    fig_corr.update_layout(
        title='Return Correlation Matrix',
        height=500, **PLOTLY_TEMPLATE['layout'],
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_pie)),
                             style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                                    'border': f'1px solid {PALETTE["border"]}'}), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_sector)),
                             style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                                    'border': f'1px solid {PALETTE["border"]}'}), md=6),
        ], className='g-3 mb-4'),
        dbc.Card(dbc.CardBody([
            html.H5('Allocation Detail', style={'color': PALETTE['primary'], 'marginBottom': '12px'}),
            dash_table.DataTable(
                data=allocation.to_dict('records'),
                columns=[
                    {'name': 'Stock', 'id': 'Stock'},
                    {'name': 'Weight (%)', 'id': 'Weight (%)'},
                    {'name': 'Amount (₹)', 'id': 'Amount (₹)', 'type': 'numeric',
                     'format': dash_table.FormatTemplate.money(0)},
                ],
                style_header={'backgroundColor': PALETTE['bg_card2'], 'color': PALETTE['primary'],
                              'fontWeight': '600', 'border': f'1px solid {PALETTE["border"]}'},
                style_cell={'backgroundColor': PALETTE['bg_card'], 'color': PALETTE['text'],
                            'border': f'1px solid {PALETTE["border"]}', 'textAlign': 'center',
                            'padding': '8px'},
            ),
        ]), style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                   'border': f'1px solid {PALETTE["border"]}', 'marginBottom': '20px'}),
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_corr)),
                 style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                        'border': f'1px solid {PALETTE["border"]}'}),
    ])


# ──────────────────────────────────────────────────────────────
# PAGE: Models
# ──────────────────────────────────────────────────────────────

def build_models():
    # Comparison table
    rows = []
    for model, stocks in all_metrics.items():
        for ticker, metrics in stocks.items():
            row = {'Model': model, 'Stock': short(ticker)}
            row.update({k: round(v, 2) if isinstance(v, (int, float)) and not np.isnan(v) else '—' for k, v in metrics.items()})
            rows.append(row)

    # Average by model
    avg_rows = []
    for model, stocks in all_metrics.items():
        mapes = [m.get('MAPE (%)', np.nan) for m in stocks.values()]
        rmses = [m.get('RMSE', np.nan) for m in stocks.values()]
        das = [m.get('Directional Accuracy (%)', np.nan) for m in stocks.values()]
        avg_rows.append({
            'Model': model,
            'Avg MAPE (%)': round(np.nanmean(mapes), 2),
            'Avg RMSE': round(np.nanmean(rmses), 2),
            'Avg Dir. Accuracy (%)': round(np.nanmean(das), 2),
        })

    fig_comp = go.Figure()
    for i, row in enumerate(avg_rows):
        fig_comp.add_trace(go.Bar(
            name=row['Model'],
            x=['MAPE (%)', 'Dir. Accuracy (%)'],
            y=[row['Avg MAPE (%)'], row['Avg Dir. Accuracy (%)']],
            marker_color=STOCK_COLORS[i % len(STOCK_COLORS)],
            text=[f"{row['Avg MAPE (%)']:.1f}", f"{row['Avg Dir. Accuracy (%)']:.1f}"],
            textposition='outside',
        ))
    fig_comp.update_layout(
        title='Model Performance Comparison',
        barmode='group', height=400,
        **PLOTLY_TEMPLATE['layout'],
    )

    return html.Div([
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_comp)),
                 style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                        'border': f'1px solid {PALETTE["border"]}', 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5('Average Metrics by Model', style={'color': PALETTE['primary'], 'marginBottom': '12px'}),
                dash_table.DataTable(
                    data=avg_rows,
                    columns=[{'name': c, 'id': c} for c in avg_rows[0].keys()],
                    style_header={'backgroundColor': PALETTE['bg_card2'], 'color': PALETTE['primary'],
                                  'fontWeight': '600', 'border': f'1px solid {PALETTE["border"]}'},
                    style_cell={'backgroundColor': PALETTE['bg_card'], 'color': PALETTE['text'],
                                'border': f'1px solid {PALETTE["border"]}', 'textAlign': 'center', 'padding': '8px'},
                ),
            ]), style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                       'border': f'1px solid {PALETTE["border"]}'}), md=12),
        ], className='mb-4'),
        dbc.Card(dbc.CardBody([
            html.H5('Detailed Metrics', style={'color': PALETTE['primary'], 'marginBottom': '12px'}),
            dash_table.DataTable(
                data=rows,
                columns=[{'name': c, 'id': c} for c in rows[0].keys()] if rows else [],
                page_size=15, sort_action='native', filter_action='native',
                style_header={'backgroundColor': PALETTE['bg_card2'], 'color': PALETTE['primary'],
                              'fontWeight': '600', 'border': f'1px solid {PALETTE["border"]}'},
                style_cell={'backgroundColor': PALETTE['bg_card'], 'color': PALETTE['text'],
                            'border': f'1px solid {PALETTE["border"]}', 'textAlign': 'center',
                            'padding': '8px', 'fontSize': '0.85rem'},
            ),
        ]), style={'backgroundColor': PALETTE['bg_card'], 'borderRadius': '12px',
                   'border': f'1px solid {PALETTE["border"]}'}),
    ])


# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────
app.layout = html.Div([
    navbar,
    dbc.Container(
        html.Div(id='page-content', className='mt-4 mb-5'),
        fluid=True, style={'maxWidth': '1400px'},
    ),
    dcc.Store(id='current-page', data='Overview'),
], style={
    'backgroundColor': PALETTE['bg_dark'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif',
})


# ──────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────

@app.callback(
    Output('current-page', 'data'),
    [Input(f'nav-{p.lower()}', 'n_clicks') for p in PAGES],
    prevent_initial_call=True,
)
def nav_click(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'Overview'
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return btn_id.replace('nav-', '').capitalize()


@app.callback(
    Output('page-content', 'children'),
    Input('current-page', 'data'),
)
def render_page(page):
    if page == 'Overview' or page == 'overview':
        return build_overview()
    elif page == 'Forecasts' or page == 'forecasts':
        return build_forecasts()
    elif page == 'Volatility' or page == 'volatility':
        return build_volatility()
    elif page == 'Portfolio' or page == 'portfolio':
        return build_portfolio()
    elif page == 'Models' or page == 'models':
        return build_models()
    return build_overview()


@app.callback(
    [Output('forecast-chart', 'figure'), Output('forecast-metrics', 'children')],
    [Input('forecast-stock', 'value'), Input('forecast-model', 'value')],
)
def update_forecast(ticker, model):
    fig = go.Figure()

    if ticker and ticker in test_data:
        test = test_data[ticker]
        fig.add_trace(go.Scatter(
            x=test.index, y=test.values,
            name='Actual (Test)', line=dict(color=PALETTE['primary'], width=2),
        ))

        if ticker in forecast_details and model in forecast_details[ticker]:
            fd = forecast_details[ticker][model]
            preds = fd.get('test_predictions', [])
            if preds:
                fig.add_trace(go.Scatter(
                    x=test.index[:len(preds)], y=preds,
                    name=f'{model} Predicted', line=dict(color=PALETTE['secondary'], width=2, dash='dash'),
                ))

            # Future forecast
            fc = fd.get('forecast', [])
            if fc:
                last_date = test.index[-1]
                fc_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=len(fc))
                fig.add_trace(go.Scatter(
                    x=fc_dates, y=fc,
                    name='2-Day Forecast', mode='lines+markers',
                    line=dict(color=PALETTE['accent2'], width=3),
                    marker=dict(size=10, symbol='diamond'),
                ))
                lower = fd.get('lower', [])
                upper = fd.get('upper', [])
                if lower and upper:
                    fig.add_trace(go.Scatter(
                        x=list(fc_dates) + list(fc_dates[::-1]),
                        y=list(upper) + list(lower[::-1]),
                        fill='toself', fillcolor='rgba(255,230,109,0.15)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='95% CI', showlegend=True,
                    ))

    fig.update_layout(
        title=f'{short(ticker or "")} — {model or ""} Forecast',
        xaxis_title='Date', yaxis_title='Price (₹)',
        height=450, **PLOTLY_TEMPLATE['layout'],
        hovermode='x unified',
    )

    # Metrics card
    metrics_card = html.Div()
    if ticker and model and model in all_metrics and ticker in all_metrics[model]:
        m = all_metrics[model][ticker]
        metrics_card = dbc.Row([
            dbc.Col(make_kpi_card('MAPE', f"{m.get('MAPE (%)', '—')}%", '', PALETTE['primary']), md=3),
            dbc.Col(make_kpi_card('RMSE', f"{m.get('RMSE', '—')}", '', PALETTE['accent1']), md=3),
            dbc.Col(make_kpi_card('MAE', f"{m.get('MAE', '—')}", '', PALETTE['accent2']), md=3),
            dbc.Col(make_kpi_card('Dir. Accuracy', f"{m.get('Directional Accuracy (%)', '—')}%", '', PALETTE['accent3']), md=3),
        ], className='g-3')

    return fig, metrics_card


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  ╔═══════════════════════════════════════════════╗")
    print("  ║   TSA 2026 Dashboard — http://localhost:8050  ║")
    print("  ╚═══════════════════════════════════════════════╝\n")
    app.run(debug=True, port=8050)
