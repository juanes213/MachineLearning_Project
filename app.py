from jupyter_dash import JupyterDash
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
from typing import List, Any
import base64
import warnings

warnings.filterwarnings('ignore')

# Inicializar la aplicación Dash
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# Descargar datos
simbolo = "EURUSD=X"
futuro = yf.Ticker(simbolo)
datos_historicos = futuro.history(start="2022-01-01", end="2023-06-30")
datos_historicos.index = pd.to_datetime(datos_historicos.index)
datos_historicos['Retornos'] = datos_historicos['Close'].pct_change()

dh_weekly = datos_historicos.resample('W').mean()

# Q-learning settings
thresholds = [0.07, 0.08, 0.09, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25] 
n_thresholds = len(thresholds)
q_values = np.zeros(n_thresholds)
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.2  # Exploration rate for ε-greedy strategy
choices = []
data = np.cumsum(np.random.randn(100, n_thresholds), axis=0) 
strategy_colors = ['#FF5733', '#33C1FF', '#DAF7A6', '#C70039', '#FFC300', '#900C3F', '#581845', '#C70039', '#900C3F', '#581845', '#C70039']  # Colors for the strategies

def calcular_indicadores(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # EMA
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

    # SMA
    df['SMA'] = df['Close'].rolling(window=50).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACDSignal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDHist'] = df['MACD'] - df['MACDSignal']

    return df

app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="Introducción", tab_id="tab-intro"),
        dbc.Tab(label="Live Data", tab_id="tab-live-data"),
        dbc.Tab(label="Análisis Exploratorio", tab_id="tab-eda"),
        dbc.Tab(label="Análisis Backtesting", tab_id="tab-backtesting"),
        dbc.Tab(label="Modelo", tab_id="tab-modelo"),
    ], id="tabs", active_tab="tab-intro"),
    html.Div(id="content"),
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
])

@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-eda":
        candlestick_fig = go.Figure(data=[go.Candlestick(x=datos_historicos.index,
                                                         open=datos_historicos['Open'], 
                                                         high=datos_historicos['High'],
                                                         low=datos_historicos['Low'], 
                                                         close=datos_historicos['Close'],
                                                         increasing_line_color= 'lightblue', decreasing_line_color= 'darkblue')])
        candlestick_fig.update_layout(title="Candlestick Chart - Provides a visual representation of price movement.", template="plotly_dark")

        fig_weekly = px.line(dh_weekly, x=dh_weekly.index, y='Close',
                     title='Weekly Time Series (BTC-USD)',
                     labels={'x':'Fecha', 'Close':'Precio de Cierre'})
        fig_weekly.update_layout(template = 'plotly_dark')

        daily_returns_fig = go.Figure(data=[go.Scatter(x=datos_historicos.index, y=datos_historicos['Retornos'], mode='lines', line=dict(color='darkblue'))])
        daily_returns_fig.update_layout(title="Daily Returns Chart - Illustrates the day-to-day percentage change in price.", template="plotly_dark")

        distribution_fig = px.histogram(datos_historicos, x='Retornos', nbins=50, marginal='violin', color_discrete_sequence=['blue'])
        distribution_fig.update_layout(title="Distribution of Daily Returns - Analyzes the frequency and distribution of return variations.", template="plotly_dark")

        return html.Div([
            dcc.Graph(figure=candlestick_fig, style={'height': '60vh'}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_weekly), width=6),
                dbc.Col(dcc.Graph(figure=daily_returns_fig), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=distribution_fig), width=12),
            ]),
            html.P("Cada gráfico proporciona información sobre diferentes aspectos del mercado de futuros, ayudando a los traders a tomar decisiones informadas.")
        ])
    elif at == "tab-intro":
        return html.Div([
            html.H1("Exploración Avanzada del Trading Algorítmico"),

            dcc.Markdown('''
            Este proyecto se adentra en el apasionante mundo del **trading algorítmico**, aprovechando el poder de Python, el aprendizaje por refuerzo y técnicas de vanguardia para desarrollar e implementar estrategias de trading automatizadas.

            **Objetivos Clave:**

            * **Investigación de Estrategias:** Exploramos una variedad de estrategias de trading, desde reglas simples hasta algoritmos complejos, para identificar enfoques prometedores.
            * **Gestión de Riesgos:** Implementamos mecanismos de control de riesgos, como órdenes de stop-loss y take-profit, para proteger el capital y maximizar la rentabilidad.
            * **Aprendizaje por Refuerzo:** Aplicamos un modelo híbrido que combina Q-learning y redes neuronales para optimizar las decisiones de trading en tiempo real.
            * **Evaluación Rigorosa:** Utilizamos backtesting exhaustivo para evaluar el rendimiento de las estrategias en datos históricos y simular escenarios de mercado.

            **Enfoque Innovador:**

            Nuestro enfoque se distingue por la integración de **Q-learning**, un algoritmo de aprendizaje por refuerzo, con **redes neuronales**. Esta combinación permite al modelo aprender y adaptarse a las dinámicas cambiantes del mercado, mejorando continuamente su capacidad para tomar decisiones de trading óptimas.

            **Proceso Iterativo:**

            1. **Backtesting:** Evaluamos diversas estrategias de trading utilizando datos históricos para identificar umbrales de recompensa óptimos.
            2. **Selección de Estrategia:** El modelo de Q-learning, guiado por los resultados del backtesting, selecciona la estrategia más prometedora.
            3. **Implementación en Tiempo Real:** La estrategia seleccionada se ejecuta en tiempo real, y el modelo monitorea su rendimiento.
            4. **Aprendizaje Continuo:** El modelo actualiza sus parámetros y ajusta sus decisiones en función de los resultados obtenidos, buscando maximizar la rentabilidad.

            **Visualización Dinámica:**

            A continuación, se presentan dos gráficos interactivos que ilustran la simulación de estrategias y la elección de umbrales en tiempo real.

            '''),

            dcc.Graph(id='strategy-graph'),
            dcc.Graph(id='choice-graph')
        ])
    elif at == "tab-live-data":
        # Obtener datos en tiempo real
        futuro = yf.Ticker(simbolo)
        datos_actuales = futuro.history(period="1d")
        precio_actual = datos_actuales['Close'][0]  
        cambio_porcentaje = datos_actuales['Close'].pct_change()[0] * 100  

        # Calcular indicadores técnicos
        datos_con_indicadores = calcular_indicadores(datos_historicos.copy())

        # Gráfica de precio actual
        live_fig = go.Figure(data=[go.Indicator(
            mode="number+delta",
            value=precio_actual,
            delta={'reference': datos_con_indicadores['Close'][-2], 'relative': True, 'valueformat': '.2%'},
            title={"text": "Live Price"},
            number={'valueformat': '.2f'},
            domain={'x': [0, 1], 'y': [0, 1]}
        )])
        live_fig.update_layout(
            height=200,  
            paper_bgcolor="#1e1e1e",
            font=dict(color="white", size=36),  
            margin=dict(t=50, b=50, l=50, r=50),  
        )

        # Establecer el color del indicador según el cambio porcentual
        if cambio_porcentaje > 0:
            live_fig.update_traces(delta_increasing_color="green")
        else:
            live_fig.update_traces(delta_decreasing_color="red")

        # Gráfica de candlestick
        candlestick_fig = go.Figure(data=[go.Candlestick(x=datos_con_indicadores.index,
                                                        open=datos_con_indicadores['Open'],
                                                        high=datos_con_indicadores['High'],
                                                        low=datos_con_indicadores['Low'],
                                                        close=datos_con_indicadores['Close'],
                                                        increasing_line_color='lightblue', decreasing_line_color='darkblue')])
        candlestick_fig.update_layout(title="Candlestick", template="plotly_dark")

        # Gráficas de indicadores individuales
        rsi_fig = go.Figure(data=[go.Scatter(x=datos_con_indicadores.index, y=datos_con_indicadores['RSI'], mode='lines', line=dict(color='blue'))])
        rsi_fig.update_layout(title="RSI", template="plotly_dark")

        ema_fig = go.Figure(data=[go.Scatter(x=datos_con_indicadores.index, y=datos_con_indicadores['EMA'], mode='lines', line=dict(color='orange'))])
        ema_fig.update_layout(title="EMA", template="plotly_dark")

        sma_fig = go.Figure(data=[go.Scatter(x=datos_con_indicadores.index, y=datos_con_indicadores['SMA'], mode='lines', line=dict(color='purple'))])
        sma_fig.update_layout(title="SMA", template="plotly_dark")

        macd_fig = make_subplots(rows=1, cols=1)
        macd_fig.add_trace(go.Bar(x=datos_con_indicadores.index, y=datos_con_indicadores['MACDHist'], name='MACD Histograma', marker_color='lightblue'))
        macd_fig.add_trace(go.Scatter(x=datos_con_indicadores.index, y=datos_con_indicadores['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        macd_fig.add_trace(go.Scatter(x=datos_con_indicadores.index, y=datos_con_indicadores['MACDSignal'], mode='lines', name='MACD Signal', line=dict(color='orange')))
        macd_fig.update_layout(title="MACD", template="plotly_dark")

        return html.Div([
            dcc.Graph(figure=live_fig),
            dcc.Graph(figure=candlestick_fig, style={'height': '40vh'}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=rsi_fig), width=6),
                dbc.Col(dcc.Graph(figure=ema_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=sma_fig), width=6),
                dbc.Col(dcc.Graph(figure=macd_fig), width=6),
            ]),
        ])
    elif at == "tab-backtesting":
        image_paths = [
            'img/drawdown_overtime.png',
            'img/balance_overtime.png',
            'img/drawdown_balance_perfiles.png',
            'img/balance_overtime_perfil.png',
            'img/drawdown_overtime_perfil.png'
        ]

        # Análisis Markdown
        markdown_text = """
        ## Análisis de Backtesting

        ### Drawdown vs. Balance vs. Perfil

        **Distribución General:**

        *   Los perfiles se distribuyen a lo largo del eje del balance desde aproximadamente -15k hasta 10k y en el eje de drawdown desde 0 hasta más de 80k.
        *   Esta dispersión indica que el algoritmo experimenta una amplia gama de resultados en términos de balance y drawdown, dependiendo del timeframe utilizado.

        **Timeframes Cortos (Perfiles Morados/Azules):**

        *   Los perfiles con colores más morados/azules (timeframes cortos) tienden a agruparse en regiones con balances negativos y drawdowns elevados.
        *   Los timeframes cortos parecen estar asociados con mayores drawdowns y balances negativos, sugiriendo que la estrategia del algoritmo es menos efectiva en periodos de tiempo más cortos, posiblemente debido a la mayor volatilidad o a la ineficacia de las señales de trading en estos intervalos.

        **Timeframes Largos (Perfiles Amarillos/Naranjas):**

        *   Los perfiles con colores más amarillos/naranjas (timeframes largos) se agrupan más cerca del eje de balance positivo y presentan drawdowns más controlados.
        *   Los timeframes más largos parecen ser más beneficiosos para el algoritmo, mostrando menores drawdowns y balances más positivos. Esto indica una mayor estabilidad y efectividad de la estrategia en periodos de tiempo más largos, donde las tendencias de mercado pueden ser más fáciles de captar y aprovechar.

        **Centroide (Punto Destacado):**

        *   El punto destacado, que representa el centroide de todos los perfiles, se sitúa cerca del eje de balance cero y en una región de drawdown relativamente bajo.
        *   El centroide indica que, en promedio, el algoritmo tiende a operar con un balance cercano a cero y un drawdown moderado. Esto sugiere que, considerando todos los timeframes, la estrategia del algoritmo tiende a ser neutral en términos de ganancia/pérdida neta, con una exposición al riesgo moderada.

        **Conclusiones:**

        *   **Estrategia de Timeframes Cortos:** Los timeframes cortos tienden a estar asociados con resultados negativos y mayores drawdowns, lo que sugiere que las decisiones de trading basadas en intervalos cortos de tiempo podrían estar expuestas a una mayor volatilidad y a señales menos fiables.
        *   **Estrategia de Timeframes Largos:** Los timeframes largos muestran una tendencia hacia balances positivos y drawdowns más controlados, indicando que la estrategia del algoritmo es más efectiva en estos intervalos, probablemente debido a una mejor capacidad para identificar y seguir las tendencias de mercado más estables.
        """

        graphs = []
        for path in image_paths:
            with open(path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                fig = go.Figure(go.Image(source=f'data:image/png;base64,{encoded_image}'))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    width=1200,  
                    height=1000  
                )
                graphs.append(dcc.Graph(figure=fig))

        # Combina el Markdown con las imágenes
        layout = html.Div([
            dcc.Markdown(markdown_text),
            *graphs  # Desempaqueta la lista de gráficos
        ], style={'margin': '0'})
        return layout
    elif at == "tab-modelo":
        return html.Div([
            html.H2("Modelo DQN: Deep Q-Network"),

            dcc.Markdown('''
            El DQN (Deep Q-Network) es un algoritmo avanzado de aprendizaje por refuerzo que combina redes neuronales profundas con el algoritmo Q-learning. Su objetivo es aprender una política óptima para tomar decisiones en entornos complejos.

            ### Fundamentos Matemáticos y Estadísticos del DQN

            El DQN se basa en la **ecuación de Bellman**, la cual establece que el valor Q óptimo de un par estado-acción (s, a) es igual a la recompensa inmediata esperada más el valor Q descontado del siguiente estado óptimo. La red neuronal en el DQN **aproxima la función Q**, cuyos parámetros se actualizan iterativamente para minimizar la diferencia entre el valor Q objetivo (calculado con la ecuación de Bellman) y el valor Q predicho por la red. Esta minimización se logra a través del **descenso de gradiente** con el **error cuadrático medio (MSE)** como función de pérdida.

            El DQN también incorpora consideraciones estadísticas clave como:

            * **Muestreo de experiencias:** Se seleccionan aleatoriamente muestras de la memoria para romper correlaciones y mejorar la estabilidad del entrenamiento.
            * **Objetivos fijos:** Se utiliza una red objetivo con parámetros fijos para calcular los valores Q objetivo y evitar divergencias en el entrenamiento.
            * **Exploración vs. explotación:** Se balancea la exploración de nuevas acciones con la explotación de las mejores acciones conocidas a través de la variable épsilon.
            '''),

            html.H3("Resultados y Análisis"),
            html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(open('img/modelo_perfiles.png', 'rb').read()).decode()), style={'width': '100%'}),  # Asegúrate de tener la imagen en la carpeta correcta

            dcc.Markdown('''
            **Interpretación de los Resultados**

            * **Distribución General:** Los perfiles se distribuyen a lo largo del eje del balance (aproximadamente 0 a 10k) y del eje del drawdown (0 a más de 10k), indicando que el algoritmo experimenta diversos resultados en términos de balance positivo y drawdown, dependiendo del perfil (threshold) utilizado.
            * **Centroide:** El punto resaltado, que representa el centroide de todos los perfiles, se encuentra cerca del eje de balance cero y en una región de drawdown relativamente bajo. Esto sugiere que, en promedio, el algoritmo tiende a operar con un balance cercano a cero y un riesgo moderado.
            * **Conclusiones:** La mayoría de los umbrales (perfiles) tienden a acumularse cerca de 0, tanto para el drawdown como para el balance. Esto indica una gestión equilibrada del riesgo y el beneficio, donde el algoritmo busca mantener un balance estable y minimizar las pérdidas.

            ### Implicaciones para el Trading Algorítmico

            Los resultados sugieren que el modelo DQN, al ser entrenado y optimizado con diferentes umbrales, puede aprender estrategias de trading efectivas. La capacidad del modelo para adaptarse a diversas condiciones de mercado y encontrar un equilibrio entre riesgo y beneficio lo convierte en una herramienta prometedora para el trading algorítmico.

            Sin embargo, es crucial recordar que estos resultados se basan en datos históricos y simulaciones. Es importante realizar pruebas exhaustivas y validación en entornos reales antes de implementar cualquier estrategia basada en este modelo.
            '''),
        ])

@app.callback(
    [Output('strategy-graph', 'figure'), Output('choice-graph', 'figure')],
    Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    global q_values, choices
    
    if n < 100:
        chosen_threshold = np.random.randint(n_thresholds) if np.random.rand() < epsilon else np.argmax(q_values)
        choices.append(chosen_threshold)
        rewards = data[n, :] if n > 0 else np.zeros(n_thresholds)
        q_values *= gamma
        q_values[chosen_threshold] += alpha * (rewards[chosen_threshold] + gamma * np.max(q_values) - q_values[chosen_threshold])
        
        strategy_fig = go.Figure()
        for i in range(n_thresholds):
            strategy_fig.add_trace(go.Scatter(x=list(range(n+1)), y=data[:n+1, i], mode='lines', name=f'Threshold {thresholds[i]}', line=dict(color=strategy_colors[i % len(strategy_colors)])))
        strategy_fig.update_layout(title="Strategy Simulation", template="plotly_dark")

        choice_fig = go.Figure()
        choice_fig.add_trace(go.Scatter(x=list(range(n+1)), y=[thresholds[i] for i in choices], mode='markers', marker=dict(size=10, color=[strategy_colors[i % len(strategy_colors)] for i in choices])))
        choice_fig.update_layout(title="Chosen Threshold", showlegend=False, template="plotly_dark")
        
        return strategy_fig, choice_fig
    else:
        raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
