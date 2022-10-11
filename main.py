import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
from random import randint
import numpy as np

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports_new.csv", delimiter=',')
app = Flask(__name__)
about = "Полный текст и геокодированные отчеты о наблюдениях НЛО от Национального центра исследований НЛО (NUFORC). " \
        "Национальный центр исследований НЛО (NUFORC) собирает и обслуживает более 100 000 сообщений о наблюдениях " \
        "НЛО. Этот набор данных содержит само содержимое отчета, включая время, длительность местоположения и другие " \
        "атрибуты, как в необработанном виде, как оно записано на сайте NUFORC, так и в уточненной " \
        "стандартизированной форме, которая также содержит координаты широты. "
link = "http://127.0.0.1:5000"
about_filtration = "Фильтрация по "


@app.route("/")
def index():
    return render_template("index.html", link=link)


@app.route("/get_data", methods=['GET'])
def get_data():
    data = request.args
    a = int(data['columns'].split(',')[0]) - 1
    b = int(data['columns'].split(',')[1]) + 1
    c = int(data['rows'].split(',')[0]) - 1
    d = int(data['rows'].split(',')[1]) + 1
    new_df = df.iloc[c: d, a: b]
    return render_template("datatable.html", link=link, column_names=new_df.columns.values,
                           row_data=list(new_df.values.tolist()), column_types=new_df.dtypes,
                           null_values=new_df.isnull().sum(axis=0).array, about=about, zip=zip)


@app.route("/filters", methods=['GET'])
def filters():
    data = request.args
    str_start = int(data['rows'].split(',')[0]) - 1
    str_end = int(data['rows'].split(',')[1]) + 1
    new_df = df.loc[str_start: str_end]
    return filtration(data['filter'], new_df)
    # return render_template("filter_table.html", df_data=list(new_df.values.tolist()), df_names=new_df.columns.values,
    #                        title_info=about_filtration + data['filter'], zip=zip)


def filtration(filter_arg, new_df):
    duration = new_df.groupby(filter_arg).min()[['duration']]
    duration.rename(columns={'duration': 'min'}, inplace=True)
    max_duration = new_df.groupby(filter_arg).max()[['duration']]
    mean_duration = new_df.groupby(filter_arg).mean()[['duration']]
    duration['max'] = max_duration['duration']
    duration['average'] = mean_duration['duration']
    return duration.to_html()


if __name__ == "__main__":
    app.run(debug=True)
