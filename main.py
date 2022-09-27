import pandas as pd
from flask import Flask, redirect, url_for, request, render_template

bf = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
app = Flask(__name__)
about = "Полный текст и геокодированные отчеты о наблюдениях НЛО от Национального центра исследований НЛО (NUFORC). Национальный центр исследований НЛО (NUFORC) собирает и обслуживает более 100 000 сообщений о наблюдениях НЛО. Этот набор данных содержит само содержимое отчета, включая время, длительность местоположения и другие атрибуты, как в необработанном виде, как оно записано на сайте NUFORC, так и в уточненной стандартизированной форме, которая также содержит координаты широты."

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/table")
def table():
    return bf.to_html(header="true", table_id="table")


@app.route("/resultdata", methods=['GET'])
def resultdata():
    data = request.args
    a = int(data['columns'].split(',')[0])
    b = int(data['columns'].split(',')[1]) + 1
    c = int(data['rows'].split(',')[0])
    d = int(data['rows'].split(',')[1]) + 1
    newdf = bf.iloc[c: d, a: b]
    datatypes = newdf.dtypes
    ne = newdf.isnull().sum(axis=0).array
    return render_template("datatable.html", column_names=newdf.columns.values,
                           row_data=list(newdf.values.tolist()), info=datatypes, ne=ne, about=about, zip=zip)


def filt(group_column, df):
    time = df.groupby(group_column).min()[['продолжительность']]
    time.rename(columns={'продолжительность': 'min'}, inplace=True)
    max_time = bf.groupby(group_column).max()[['продолжительность']]
    mean_time = bf.groupby(group_column).mean()[['продолжительность']]
    time['max'] = max_time['продолжительность']
    time['average'] = mean_time['продолжительность']
    return time.to_html()


@app.route("/filters", methods=['GET'])
def filters():
    return "<h3>Минимальная, максимальная, средняя продолжительность для города</h3><br>" \
           + filt('город', bf) + "<br>" \
           + "<br><h3>Минимальная, максимальная, средняя цена</h3><br>" \
           + filt('штат', bf) + "<br>" \
           + "<h3>Минимальная, максимальная, средняя цена у домов, сгруппированных по кол-ву этажей</h3><br>" \
           + filt('время', bf) + "<br>" \
           + "<h3>Минимальная, максимальная, средняя цена у домов на набережных</h3><br>" \
           + filt('опубликовано', bf) + "<br>" \
           # + render_template("filter_table.html", column_names=newdf.columns.values,
           #                   row_data=list(newdf.values.tolist()), zip=zip)


if __name__ == "__main__":
    app.run(debug=True)
