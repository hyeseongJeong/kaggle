import pandas as pd
import pandas_profiling as pp


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_x_rate_by_group(data: pd.DataFrame, x: str, group: str):

    return data[[group, x]].groupby([group], as_index=False).mean().sort_values(by=x, ascending=False)


def replace_values(data: pd.DataFrame, raw: str, values: list, name: str):
    try:
        data[raw] = data[raw].replace(values, name)
        return True
    except Exception as e:
        return False


def profile_report(data: pd.DataFrame, title='report'):
    html = {'style': {'full_width': True}}
    report = pp.ProfileReport(data, title=title, html=html)
    report.to_file(output_file=f'./{title}.html')
