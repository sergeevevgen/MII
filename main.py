import pandas as pd
from flask import Flask, redirect, url_for, request, render_template

bf = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
app = Flask(__name__)
about = "Полный текст и геокодированные отчеты о наблюдениях НЛО от Национального центра исследований НЛО (NUFORC). Национальный центр исследований НЛО (NUFORC) собирает и обслуживает более 100 000 сообщений о наблюдениях НЛО. Этот набор данных содержит само содержимое отчета, включая время, длительность местоположения и другие атрибуты, как в необработанном виде, как оно записано на сайте NUFORC, так и в уточненной стандартизированной форме, которая также содержит координаты широты."


@app.route("/")
def home():
    return "<html><form action='http://127.0.0.1:5000/resultdata' method=get><h2>Столбцы</h2><input type=text size=20 " \
           "name=columns><h2>Строки</h2><input type=text size=20 name=rows>" \
           "<br><input type=submit value='Enter'></form></html>"


@app.route("/table")
def num_text():
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


if __name__ == "__main__":
    app.run(debug=True)
