# ОБНАРУЖЕНИЕ И ИДЕНТИФИКАЦИЯ ИЗДЕЛИЙ НА
# ИЗОБРАЖЕНИЯХ ВИДЕОРЯДА КРЕПЁЖНЫХ
# ДЕТАЛЕЙ с. 31
import math

import nltk
from nltk import BigramCollocationFinder, BigramAssocMeasures, FreqDist, ConditionalFreqDist, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer

# Открытие и закрытие файла
f = open('text.txt', 'r', encoding="utf-8")
text = f.read()
tokens = word_tokenize(text.lower(), language="russian")
f.close()
filtered_tokens = list()
stop_words = stopwords.words("russian")
stop_words.extend(['т.д.', '.', ',', '"', '""', ':', ';', '(', ')', '[', ']', '{', '}'])

# Удаляем из текста Стоп-слова
for token in tokens:
    if token not in stop_words:
        filtered_tokens.append(token)

# print(filtered_tokens)

# Записываем данные в новый файл (данные приведены к нижнему регистру и разделены по словам)
# for i in range(len(filtered_tokens)):
#     filtered_tokens[i] += '\n'
#
# filtered_tokens[len(filtered_tokens) - 1] = filtered_tokens[len(filtered_tokens) - 1][-2:]

# f = open("text_wo_sw.txt", "w", encoding="utf-8")
# f.writelines(filtered_tokens)
# f.close()

# Морфологический разбор
morph = pymorphy2.MorphAnalyzer()
# Количество глаголов в прошедшем времени
verbs_past = 0
lemmatized_word = set()
for word in filtered_tokens:
    p = morph.parse(word)[0]
    print(morph.normal_forms(word)[0])
    lemmatized_word.add(p.normal_form)
    if 'VERB' in p.tag and 'past' in p.tag:
        verbs_past += 1

print(f"Количество глаголов в прошедшем времени в тексте - {verbs_past} шт.")

# Статистический анализ
# Лемматизируем текст
# lemmatize = nltk.WordNetLemmatizer()
# filtered_tokens = [lemmatize.lemmatize(word) for word in filtered_tokens]

# # Импортируем модуль TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer()
#
# # Преобразуем текст
# values = tfidf_vectorizer.fit_transform(lemmatize_word)
# print(tfidf_vectorizer.get_feature_names_out())
# sorted_tf_idf = sorted(values.items(), key=lambda x: x[1], reverse=True)
# print(sorted_tf_idf[:10])

# 2ой способ
# Стемминг (приведение к начальной форме)
stemmer = SnowballStemmer('russian')
filtered_tokens = [stemmer.stem(token) for token in filtered_tokens]
# Создание списка двусловий (биграмм)
finder = BigramCollocationFinder.from_words(filtered_tokens)

# Ограничение частоты двусловий (биграммов) до 100
# finder.apply_freq_filter(2)

# Расчет TF×IDF для каждого двусловия (биграммы)
bigram_measures = BigramAssocMeasures()
n = len(filtered_tokens)
word_fd = FreqDist(filtered_tokens)
bigram_fd = ConditionalFreqDist(nltk.bigrams(filtered_tokens))
tf_idf = {}
for bigram in finder.nbest(bigram_measures.raw_freq, len(filtered_tokens)):
    tf = bigram_fd[bigram[0]][bigram[1]] / n
    idf = math.log(n / (word_fd[bigram[0]] * word_fd[bigram[1]]))
    tf_idf[bigram] = tf * idf

# сортировка биграмм по значению TF×IDF
sorted_tf_idf = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)

# вывод 10 наиболее статистически значимых биграмм
print(sorted_tf_idf[:10])
