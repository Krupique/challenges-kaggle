# Visualização dos dados
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

# O objetivo desta biblioteca é agregar todas as funções responsáveis para exibição de gráficos.

# Cria gráfico de acordo com o parâmetro type, portanto, temos: Barplot, Line, Boxplot, Histogram.
def exibirGrafico(list_df, x, y, type, title = 'Title Default', width = 1000, height = 400, xlabel = 'xlabel', ylabel = 'ylabel'):
    fig = go.Figure()

    if type == 'bar':
        fig.add_traces(go.Bar(x=list_df[x], y=list_df[y]))
    elif type == 'line':
        fig.add_traces(go.Line(x=list_df[x], y=list_df[y]))
    elif type == 'box':
        fig.add_traces(go.Box(x=list_df[x], y=list_df[y]))
    elif type == 'histogram':
        fig.add_traces(go.Histogram(x=list_df[y]))


    fig.update_layout(
        title=f'<span>{title}</span>', 
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(title=f'<span>{xlabel}</span>'),
        yaxis=dict(title=f'<span>{ylabel}</span>'),
        margin=dict(l=10, r=10, t=35, b=0),
    )

    return fig