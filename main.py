import math
from sklearn.linear_model import LinearRegression
import pandas as pd
from bitarray import bitarray
from flask import Flask, redirect, url_for, request, render_template
import json
import plotly
import plotly.express as px

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
df2 = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports_new.csv", delimiter=',')
app = Flask(__name__)
about = "Полный текст и геокодированные отчеты о наблюдениях НЛО от Национального центра исследований НЛО (NUFORC). " \
        "Национальный центр исследований НЛО (NUFORC) собирает и обслуживает более 100 000 сообщений о наблюдениях " \
        "НЛО. Этот набор данных содержит само содержимое отчета, включая время, длительность местоположения и другие " \
        "атрибуты, как в необработанном виде, как оно записано на сайте NUFORC, так и в уточненной " \
        "стандартизированной форме, которая также содержит координаты широты. "
link = "http://127.0.0.1:5000"
about_filtration = "Фильтрация по "


@app.route("/")
def bloom():
    return render_template("bloom.html", title="Фильтр Блума", link=link)


@app.route("/index")
def index():
    return render_template("index.html", link=link)


@app.route("/data", methods=['GET'])
def get_data():
    data = request.args
    a = int(data['columns'].split(',')[0]) - 1
    b = int(data['columns'].split(',')[1])
    c = int(data['rows'].split(',')[0]) - 1
    d = int(data['rows'].split(',')[1])
    new_df = df.iloc[c: d, a: b]
    return render_template("datatable.html", title="НЛО", link=link, column_names=new_df.columns.values,
                           row_data=list(new_df.values.tolist()), column_types=new_df.dtypes,
                           null_values=new_df.isnull().sum(axis=0).array, about=about, description='Количество '
                                                                                                   'нулевых '
                                                                                                   'значений', zip=zip)


@app.route("/filters", methods=['GET'])
def filters():
    data = request.args
    str_start = int(data['rows'].split(',')[0]) - 1
    str_end = int(data['rows'].split(',')[1]) - 1
    new_df = df.loc[str_start: str_end]
    new_df = filtration(data['filter'], new_df)
    return render_template("filter_table.html", df_data=list(new_df.values.tolist()), df_names=new_df.columns.values,
                           title_info=about_filtration + data['filter'], zip=zip)


@app.route("/visualization", methods=['GET'])
def visualization():
    data = request.args
    c = int(data['rows'].split(',')[0]) - 1
    d = int(data['rows'].split(',')[1]) - 1
    new_df = df.loc[c: d]
    if data['filter'] == 'shape':
        gr = graphics(new_df, 'shape', 'duration', "Graphic represent shape/duration correlation", type_gr='bar')
        return render_template("visualize.html", graphJSON=gr)
    if data['filter'] == 'state':
        gr = graphics(new_df, 'state', 'duration', "Graphic represent state/duration correlation", type_gr='scatter')
        return render_template("visualize.html", graphJSON=gr)
    if data['filter'] == 'duration':
        d = filtration(data['filter'], new_df)
        gr = graphics(d, 'duration', ["min", "max", "mean"], "Graphic represent duration", type_gr='scatter')
        return render_template("visualize.html", graphJSON=gr)
    if data['filter'] == 'difference':
        d = filtration(data['filter'], new_df)
        gr = graphics(d, 'difference', ["min", "max", "mean"], "Graphic represent difference", type_gr='scatter')
        return render_template("visualize.html", graphJSON=gr)
    return render_template("visualize.html")


@app.route("/linear_regression", methods=['GET'])
def linear_reg():
    data = []
    new_df = df
    new_df['date_time'] = pd.to_datetime(new_df['date_time'], format='%Y/%m/%dT%H:%M:%S').copy()
    new_df['date_time'] = new_df['date_time'].dt.strftime('%m')
    for i in new_df['date_time']:
        if i == "1" or "2" or "12":
            data.append("winter")
        if i == "3" or "4" or "5":
            data.append("spring")
        if i == "6" or "7" or "8":
            data.append("summer")
        if i == "9" or "10" or "11":
            data.append("autumn")

    k = round(len(new_df.axes[0]) * 0.99)
    # x = data[:k]
    # y = new_df['duration'][:k]
    new_df2 = pd.DataFrame()
    new_df2['season'] = data[:k]
    new_df2['duration'] = new_df['duration'][:k]
    model = LinearRegression()
    model.fit(new_df2['season'], new_df2['duration'])
    gr = graphics(new_df2, "season", "duration", "regression", 'scatter')
    return render_template("visualize.html", graphJSON=gr)


@app.route("/bloom_filter", methods=['GET'])
def bloom_filter():
    data = request.args
    f_bloom = BloomFilter(200, 100)
    key_word_array = ["shape", "city", "state", "sights", "ufo", "indians", "diabetes",
                      "house", "sale"]
    str_ = "Full text and geocoded UFO sightings reports from the National UFO Research Center (NUFORCE). There are " \
           "its shape, city, state and duration, and other kinds of sights"
    dict_links = {'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database':
                      'Predict the onset of diabetes based on diagnostic measures of indians.',
                  'https://www.kaggle.com/datasets/harlfoxem/housesalesprediction': 'This dataset contains house sale '
                                                                                    'prices for King County'}

    for i in range(len(key_word_array)):
        f_bloom.add_to_filter(key_word_array[i])

    if not f_bloom.check_is_not_in_filter(data['key']):
        if data['key'].lower() in str_.lower():
            return redirect(link + "/index")
        for i in dict_links.keys():
            if data['key'] in dict_links[i].lower():
                return redirect(i)
    return render_template("bloom.html", about="Данные по ключевому слову " + data['key'] + " не найдены",
                           title="Фильтр блума")


# Function for graphics
def graphics(new_df, x, y, title, type_gr):
    if type_gr == 'bar':
        fig = px.bar(new_df, x=x, y=y,
                     barmode='group', title=title)
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graph_json
    if type_gr == 'scatter':
        fig = px.scatter(new_df, x=x, y=y, title=title)
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graph_json
    return


# Function for grouping data
def filtration(filter_arg, new_df):
    if filter_arg == 'duration':
        min_duration = new_df[filter_arg].min()
        max_duration = new_df[filter_arg].max()
        mean_duration = new_df[filter_arg].mean()
        data = {'': ['min', 'max', 'mean'], filter_arg: [min_duration, max_duration, mean_duration]}
        return pd.DataFrame(data)

    if filter_arg == 'difference':
        new_df['posted'] = pd.to_datetime(new_df['posted'], format='%Y/%m/%dT%H:%M:%S').copy()
        new_df['date_time'] = pd.to_datetime(new_df['date_time'], format='%Y/%m/%dT%H:%M:%S').copy()
        new_df[filter_arg] = new_df['posted'] - new_df['date_time']

        min_duration = new_df[filter_arg].min()
        max_duration = new_df[filter_arg].max()
        mean_duration = new_df[filter_arg].mean()
        data = {'': ['min', 'max', 'mean'], filter_arg: [min_duration, max_duration, mean_duration]}
        return pd.DataFrame(data)

    duration = new_df.groupby(filter_arg)[['duration']].min()
    duration.rename(columns={'duration': 'min'}, inplace=True)
    max_duration = new_df.groupby(filter_arg)[['duration']].max()
    mean_duration = new_df.groupby(filter_arg)[['duration']].mean()
    duration['max'] = max_duration['duration']
    duration['average'] = mean_duration['duration']
    duration[filter_arg] = duration.axes[0].values
    return duration


# Bloom filter
class BloomFilter(object):

    def __init__(self, size, number_expected_elements=100):
        self.size = size
        self.number_expected_elements = number_expected_elements
        self.bloom_filter = bitarray(self.size)
        self.bloom_filter.setall(0)
        self.number_hash_functions = round((self.size / self.number_expected_elements) * math.log(2))

    def _hash_djb2(self, s):
        hash_ = 5381
        for x in s:
            hash_ = ((hash_ << 5) + hash_) + ord(x)
        return hash_ % self.size

    def _hash(self, item, K):
        return self._hash_djb2(str(K) + item)

    def add_to_filter(self, item):
        for i in range(self.number_hash_functions):
            self.bloom_filter[self._hash(item, i)] = 1

    def check_is_not_in_filter(self, item):
        for i in range(self.number_hash_functions):
            if self.bloom_filter[self._hash(item, i)] == 0:
                return True
        return False


if __name__ == "__main__":
    app.run(debug=True)
