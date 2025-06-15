"""
Advanced visualization tools for reports
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Create interactive visualizations"""
    
    def __init__(self):
        self.default_layout = {
            'template': 'plotly_dark',
            'font': {'family': 'Arial', 'size': 12},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    
    def create_equity_curve(
        self,
        equity_data: pd.DataFrame,
        title: str = "Portfolio Equity Curve"
    ) -> go.Figure:
        """Create interactive equity curve"""
        
        fig = go.Figure()
        
        # Main equity line
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=equity_data['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00ff41', width=2)
        ))
        
        # Add cash line if available
        if 'cash' in equity_data.columns:
            fig.add_trace(go.Scatter(
                x=equity_data.index,
                y=equity_data['cash'],
                mode='lines',
                name='Cash',
                line=dict(color='#ffaa00', width=1, dash='dash')
            ))
        
        # Calculate and show drawdown
        peak = equity_data['equity'].expanding().max()
        drawdown = (equity_data['equity'] - peak) / peak * 100
        
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=drawdown,
            mode='lines',
            name='Drawdown %',
            line=dict(color='#ff0066', width=1),
            yaxis='y2',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 102, 0.1)'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            yaxis2=dict(
                title="Drawdown (%)",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            **self.default_layout
        )
        
        return fig
    
    def create_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """Create returns distribution histogram"""
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns * 100,  # Convert to percentage
            nbinsx=50,
            name='Returns',
            marker_color='#00ff41',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mean = returns.mean() * 100
        std = returns.std() * 100
        x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist * len(returns) * (returns.max() - returns.min()) * 100 / 50,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#ffaa00', width=2)
        ))
        
        # Add vertical lines for mean and median
        fig.add_vline(x=mean, line_dash="dash", line_color="white", annotation_text="Mean")
        fig.add_vline(x=returns.median() * 100, line_dash="dash", line_color="gray", annotation_text="Median")
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns (%)",
            yaxis_title="Frequency",
            showlegend=True,
            **self.default_layout
        )
        
        return fig
    
    def create_performance_dashboard(
        self,
        performance_data: Dict,
        trades_df: pd.DataFrame
    ) -> go.Figure:
        """Create comprehensive performance dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Monthly Returns', 'Win Rate Trend',
                'Trade P&L Distribution', 'Cumulative Trades',
                'Risk Metrics', 'Trade Duration'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'indicator'}, {'type': 'histogram'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Monthly Returns
        if 'monthly' in performance_data:
            monthly_df = pd.DataFrame(performance_data['monthly'])
            colors = ['#00ff41' if r > 0 else '#ff0066' for r in monthly_df['return']]
            
            fig.add_trace(
                go.Bar(
                    x=monthly_df['period'],
                    y=monthly_df['return'],
                    marker_color=colors,
                    name='Monthly Returns'
                ),
                row=1, col=1
            )
        
        # 2. Win Rate Trend
        if not trades_df.empty:
            trades_df['cumulative_win_rate'] = (
                trades_df['pnl'] > 0
            ).expanding().mean() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df.index,
                    y=trades_df['cumulative_win_rate'],
                    mode='lines',
                    line=dict(color='#00ff41', width=2),
                    name='Win Rate'
               ),
               row=1, col=2
           )
       
       # 3. Trade P&L Distribution
       if not trades_df.empty and 'pnl' in trades_df.columns:
           fig.add_trace(
               go.Histogram(
                   x=trades_df['pnl'],
                   nbinsx=30,
                   marker_color='#00ff41',
                   opacity=0.7,
                   name='P&L'
               ),
               row=2, col=1
           )
       
       # 4. Cumulative Trades
       if not trades_df.empty:
           trades_df['trade_count'] = range(1, len(trades_df) + 1)
           trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
           
           fig.add_trace(
               go.Scatter(
                   x=trades_df['trade_count'],
                   y=trades_df['cumulative_pnl'],
                   mode='lines',
                   line=dict(color='#ffaa00', width=2),
                   name='Cumulative P&L'
               ),
               row=2, col=2
           )
       
       # 5. Risk Metrics (Indicators)
       risk_metrics = performance_data.get('risk_metrics', {})
       
       fig.add_trace(
           go.Indicator(
               mode="number+gauge+delta",
               value=risk_metrics.get('sharpe_ratio', 0),
               domain={'x': [0, 1], 'y': [0, 1]},
               title={'text': "Sharpe Ratio"},
               gauge={
                   'axis': {'range': [-1, 3]},
                   'bar': {'color': "#00ff41"},
                   'steps': [
                       {'range': [-1, 0], 'color': "#ff0066"},
                       {'range': [0, 1], 'color': "#ffaa00"},
                       {'range': [1, 3], 'color': "#00ff41"}
                   ]
               }
           ),
           row=3, col=1
       )
       
       # 6. Trade Duration
       if not trades_df.empty and 'duration' in trades_df.columns:
           fig.add_trace(
               go.Histogram(
                   x=trades_df['duration'],
                   nbinsx=20,
                   marker_color='#ffaa00',
                   opacity=0.7,
                   name='Duration'
               ),
               row=3, col=2
           )
       
       # Update layout
       fig.update_layout(
           height=1200,
           showlegend=False,
           **self.default_layout
       )
       
       # Update axes
       fig.update_xaxes(title_text="Month", row=1, col=1)
       fig.update_yaxes(title_text="Return (%)", row=1, col=1)
       fig.update_xaxes(title_text="Trade #", row=1, col=2)
       fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
       fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
       fig.update_yaxes(title_text="Frequency", row=2, col=1)
       fig.update_xaxes(title_text="Trade Count", row=2, col=2)
       fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=2)
       fig.update_xaxes(title_text="Duration (hours)", row=3, col=2)
       fig.update_yaxes(title_text="Frequency", row=3, col=2)
       
       return fig
   
   def create_correlation_heatmap(
       self,
       returns_df: pd.DataFrame,
       title: str = "Symbol Correlation Matrix"
   ) -> go.Figure:
       """Create correlation heatmap"""
       
       # Calculate correlation
       corr_matrix = returns_df.corr()
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=corr_matrix.values,
           x=corr_matrix.columns,
           y=corr_matrix.columns,
           colorscale='RdBu',
           zmid=0,
           text=corr_matrix.values.round(2),
           texttemplate='%{text}',
           textfont={"size": 10},
           colorbar=dict(title="Correlation")
       ))
       
       fig.update_layout(
           title=title,
           xaxis={'side': 'bottom'},
           yaxis={'side': 'left'},
           **self.default_layout
       )
       
       return fig
   
   def create_position_analysis(
       self,
       positions_history: pd.DataFrame,
       title: str = "Position Analysis"
   ) -> go.Figure:
       """Create position analysis charts"""
       
       fig = make_subplots(
           rows=2, cols=2,
           subplot_titles=(
               'Position Values Over Time',
               'Position Concentration',
               'Entry vs Exit Analysis',
               'Hold Time vs Return'
           ),
           specs=[
               [{'type': 'scatter'}, {'type': 'pie'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]
           ]
       )
       
       # 1. Position values over time (stacked area)
       if 'symbol' in positions_history.columns:
           for symbol in positions_history['symbol'].unique():
               symbol_data = positions_history[positions_history['symbol'] == symbol]
               
               fig.add_trace(
                   go.Scatter(
                       x=symbol_data.index,
                       y=symbol_data['value'],
                       mode='lines',
                       name=symbol,
                       stackgroup='one'
                   ),
                   row=1, col=1
               )
       
       # 2. Current position concentration (pie chart)
       if 'current_positions' in positions_history.columns:
           current = positions_history.iloc[-1]['current_positions']
           if current:
               symbols = list(current.keys())
               values = [pos['value'] for pos in current.values()]
               
               fig.add_trace(
                   go.Pie(
                       labels=symbols,
                       values=values,
                       hole=0.3
                   ),
                   row=1, col=2
               )
       
       # 3. Entry vs Exit prices
       if 'entry_price' in positions_history.columns and 'exit_price' in positions_history.columns:
           fig.add_trace(
               go.Scatter(
                   x=positions_history['entry_price'],
                   y=positions_history['exit_price'],
                   mode='markers',
                   marker=dict(
                       size=8,
                       color=positions_history['pnl'],
                       colorscale='RdYlGn',
                       showscale=True
                   ),
                   text=positions_history['symbol'],
                   name='Entry vs Exit'
               ),
               row=2, col=1
           )
           
           # Add diagonal line (breakeven)
           price_range = [
               positions_history['entry_price'].min(),
               positions_history['entry_price'].max()
           ]
           fig.add_trace(
               go.Scatter(
                   x=price_range,
                   y=price_range,
                   mode='lines',
                   line=dict(color='white', dash='dash'),
                   showlegend=False
               ),
               row=2, col=1
           )
       
       # 4. Hold time vs Return
       if 'hold_time' in positions_history.columns and 'return_pct' in positions_history.columns:
           fig.add_trace(
               go.Scatter(
                   x=positions_history['hold_time'],
                   y=positions_history['return_pct'],
                   mode='markers',
                   marker=dict(
                       size=8,
                       color=positions_history['return_pct'],
                       colorscale='RdYlGn',
                       showscale=True
                   ),
                   text=positions_history['symbol'],
                   name='Hold Time vs Return'
               ),
               row=2, col=2
           )
       
       fig.update_layout(
           height=800,
           showlegend=True,
           **self.default_layout
       )
       
       return fig
   
   def create_strategy_comparison(
       self,
       strategy_results: Dict[str, pd.DataFrame],
       metric: str = 'equity'
   ) -> go.Figure:
       """Compare multiple strategies"""
       
       fig = go.Figure()
       
       colors = px.colors.qualitative.Set1
       
       for i, (strategy_name, data) in enumerate(strategy_results.items()):
           if metric in data.columns:
               fig.add_trace(go.Scatter(
                   x=data.index,
                   y=data[metric],
                   mode='lines',
                   name=strategy_name,
                   line=dict(color=colors[i % len(colors)], width=2)
               ))
       
       fig.update_layout(
           title=f"Strategy Comparison: {metric.title()}",
           xaxis_title="Date",
           yaxis_title=metric.title(),
           hovermode='x unified',
           **self.default_layout
       )
       
       return fig
   
   def create_risk_dashboard(
       self,
       risk_metrics: Dict,
       var_history: pd.DataFrame = None
   ) -> go.Figure:
       """Create risk analysis dashboard"""
       
       fig = make_subplots(
           rows=2, cols=2,
           subplot_titles=(
               'Risk Metrics Overview',
               'Value at Risk History',
               'Risk Contribution by Position',
               'Risk-Return Scatter'
           ),
           specs=[
               [{'type': 'indicator'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'scatter'}]
           ]
       )
       
       # 1. Risk metrics gauges
       fig.add_trace(
           go.Indicator(
               mode="gauge+number",
               value=risk_metrics.get('current_risk', 0) * 100,
               domain={'x': [0, 0.5], 'y': [0.5, 1]},
               title={'text': "Portfolio Risk (%)"},
               gauge={
                   'axis': {'range': [0, 100]},
                   'bar': {'color': "#ffaa00"},
                   'steps': [
                       {'range': [0, 30], 'color': "#00ff41"},
                       {'range': [30, 70], 'color': "#ffaa00"},
                       {'range': [70, 100], 'color': "#ff0066"}
                   ],
                   'threshold': {
                       'line': {'color': "red", 'width': 4},
                       'thickness': 0.75,
                       'value': 90
                   }
               }
           ),
           row=1, col=1
       )
       
       # 2. VaR History
       if var_history is not None and not var_history.empty:
           fig.add_trace(
               go.Scatter(
                   x=var_history.index,
                   y=var_history['var_95'],
                   mode='lines',
                   name='95% VaR',
                   line=dict(color='#ffaa00', width=2)
               ),
               row=1, col=2
           )
           
           fig.add_trace(
               go.Scatter(
                   x=var_history.index,
                   y=var_history['var_99'],
                   mode='lines',
                   name='99% VaR',
                   line=dict(color='#ff0066', width=2)
               ),
               row=1, col=2
           )
       
       # 3. Risk contribution by position
       if 'position_risks' in risk_metrics:
           positions = list(risk_metrics['position_risks'].keys())
           risks = list(risk_metrics['position_risks'].values())
           
           fig.add_trace(
               go.Bar(
                   x=positions,
                   y=risks,
                   marker_color='#00ff41',
                   name='Risk Contribution'
               ),
               row=2, col=1
           )
       
       # 4. Risk-Return scatter
       if 'position_metrics' in risk_metrics:
           metrics_df = pd.DataFrame(risk_metrics['position_metrics'])
           
           fig.add_trace(
               go.Scatter(
                   x=metrics_df['risk'],
                   y=metrics_df['return'],
                   mode='markers+text',
                   text=metrics_df['symbol'],
                   textposition="top center",
                   marker=dict(
                       size=metrics_df['size'] / metrics_df['size'].max() * 50,
                       color=metrics_df['sharpe'],
                       colorscale='RdYlGn',
                       showscale=True
                   ),
                   name='Positions'
               ),
               row=2, col=2
           )
       
       fig.update_layout(
           height=800,
           showlegend=True,
           **self.default_layout
       )
       
       return fig
   
   def save_static_charts(
       self,
       figures: Dict[str, go.Figure],
       output_dir: str
   ):
       """Save charts as static images"""
       
       from pathlib import Path
       
       output_path = Path(output_dir)
       output_path.mkdir(parents=True, exist_ok=True)
       
       for name, fig in figures.items():
           # Save as HTML (interactive)
           html_path = output_path / f"{name}.html"
           fig.write_html(str(html_path))
           
           # Save as PNG (static)
           try:
               png_path = output_path / f"{name}.png"
               fig.write_image(str(png_path))
           except Exception as e:
               logger.warning(f"Could not save PNG for {name}: {e}")
       
       logger.info(f"Charts saved to {output_path}")