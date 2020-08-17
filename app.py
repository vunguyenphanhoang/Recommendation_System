
import sqlite3
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from bs4 import BeautifulSoup as soup
from requests import get
import requests
import re

import streamlit as st
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
import glob
from collections import Counter
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@st.cache
def get_data():
    '''
    Hàm này để connect vào DB, lấy data rồi thao tác ở dạng dataframe bằng thư viện pandas
    '''
    conn = sqlite3.connect('./Tiki_web_DB.sqlite')
    df = pd.read_sql('select * from product_all_df', conn)
    return df


def main():
    df = get_data()
    data = pd.read_csv('product_fixed.csv').iloc[:,1:15]

    page = st.sidebar.selectbox("Choose a page", ["Visualize", "Recommendation System"])

    if page == "Visualize":

        st.dataframe(data.head(10))
        # st.dataframe(data)

        st.title("Trực quan hoá dữ liệu trên Tiki")

        sns.set()

        st.markdown("Biểu đồ tương quan giữa số lượng Review _ Rating")

        fig2 = plt.figure()
        sns.scatterplot(x='rating_average', y='review_count', data=data)
        plt.xscale('linear')
        plt.title('Tương quan giữa số lượng review và rating')
        st.pyplot(fig2)

        st.title("5 Sản Phẩm được review nhiều nhất")
        df_hottest_products = data[data["review_count"] > 0][["product_id",
                                                  "category",
                                                  "name",
                                                  "price",
                                                  "list_price",
                                                  "rating_average",
                                                  "review_count",
                                                  "badges"]]
        df_review = df_hottest_products.sort_values('review_count', ascending=False)
        fig = px.pie(data, values=df_review['review_count'][:5], names=df_review['name'][:5])
        st.plotly_chart(fig)



        st.markdown("Biểu đồ phân bố ngành hàng.")
        fig4 = plt.figure(figsize=(10,8))
        data['category'].value_counts().plot.bar()
        plt.title('Sản Phẩm')
        st.pyplot(fig4)

    elif page == "Recommendation System":
        conn = sqlite3.connect('./Tiki_web_DB.sqlite')
        df.to_sql('product_all_df', conn, if_exists='replace', index=False)

        st.title("Dữ liệu Tiki đã crawl:")
        st.dataframe(df.head(10))

        stop_word = []
        with open("vietnamese-stopwords.txt",encoding="utf-8") as f :
            text = f.read()
            for word in text.split() :
                stop_word.append(word)
            f.close()

        punc = list(punctuation)
        stop_word = stop_word + punc

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=stop_word)
        tfidf_matrix = tf.fit_transform(df['description'].values.astype('U'))

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        results = {}

        for idx, row in df.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices]

            results[row['id']] = similar_items[1:]

        def item(id):
            return df.loc[df['id'] == id]['name'].tolist()[0].split(' - ')[0]

        def recommend(item_id, num):
            st.write("Đề xuất " + str(num) + " sản phẩm tương tự " + item(item_id) + "...")
            st.write("-------")
            recs = results[item_id][:num]
            for rec in recs:
                st.write(item(rec[1]))

        code = st.selectbox('Chọn 1 sản phẩm mà bạn muốn xem',df['id'])
        recommend(item_id=code, num=10)

if __name__ == "__main__":
    main()
