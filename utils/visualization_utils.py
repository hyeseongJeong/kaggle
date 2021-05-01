import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def show_facet_grid(data: pd.DataFrame, col: str, sub_col: str, hue: str, aplha: float=.5, bins: int=20):
    g = sns.FacetGrid(data, col=col, hue=hue)
    g.map(sns.histplot, sub_col, alpha=aplha, bins=bins)
    g.add_legend()
    plt.show()


def show_facet_grid2(data: pd.DataFrame, col: str, *map_values):
    g = sns.FacetGrid(data, col=col, size=4.5, aspect=1.6)
    g.map(sns.pointplot, *map_values, order=None, hue_order=None)
    g.add_legend()
    plt.show()


def show_factor_flot(data: pd.DataFrame, x: str, y: str):
    sns.factorplot(x, y, data=data, aspect=2.5)
    plt.show()


def show_scatter_grid(data: pd.DataFrame, x: str, y: str, color: str, log_x=True, size_max=20, template='plotly', title='-'):
    fig = px.scatter(data,
                     x=x,
                     y=y,
                     color=color,
                     log_x=log_x,
                     size_max=size_max,
                     template=template,
                     title=title)
    fig.show()


def show_bar_plot(data: pd.DataFrame, x: str, y: str, hue: str = None):
    # data[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_index().plot(kind='bar')
    sns.barplot(x=x, y=y, hue=hue, data=data)
    plt.show()
