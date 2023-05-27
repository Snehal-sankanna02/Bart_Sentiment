#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
nltk.download('punkt')
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from rouge import Rouge
from nltk.sentiment import SentimentIntensityAnalyzer


# In[2]:


st.set_page_config(page_title='InNews: A Summarised News📰 Portal', page_icon="newspaper.ico")


# In[3]:


def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


# In[4]:


def fetch_top_news():
    site = 'https://news.google.com/news/rss'
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


# In[5]:


def fetch_category_news(topic):
    site = 'https://news.google.com/news/rss/headlines/section/topic/{}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


# In[6]:


def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_column_width=True)
    except:
        image = Image.open("no_image.jpg")
        st.image(image, use_column_width=True)


# In[7]:


def get_sentiment_label(sentiment_score):
    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def display_news(list_of_news, news_quantity):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    rouge = Rouge()
    sentiment_analyzer = SentimentIntensityAnalyzer()  # Sentiment Analysis model

    c = 0
    for news in list_of_news:
        c += 1
        st.write('**({}) {}**'.format(c, news.title.text))
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        fetch_news_poster(news_data.top_image)
        with st.expander(news.title.text):
            article_text = news_data.text
            inputs = tokenizer.batch_encode_plus([article_text], max_length=1024, return_tensors='pt', truncation=True)
            inputs.to(device)
            summary_ids = model.generate(inputs['input_ids'], num_beams=8, max_length=100, early_stopping=True)
            summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

            st.markdown(
                '''<h6 style='text-align: justify;'>{}"</h6>'''.format(summary),
                unsafe_allow_html=True)
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
            
            # Calculate ROUGE scores
            reference_summary = news_data.summary
            scores = rouge.get_scores(summary, reference_summary)

            st.write("ROUGE-1: {:.4f}".format(scores[0]['rouge-1']['f']))
            st.write("ROUGE-2: {:.4f}".format(scores[0]['rouge-2']['f']))
            st.write("ROUGE-L: {:.4f}".format(scores[0]['rouge-l']['f']))

            # Perform sentiment analysis
            sentiment_scores = sentiment_analyzer.polarity_scores(summary)
            sentiment_score = sentiment_scores['compound']
            sentiment_label = get_sentiment_label(sentiment_score)

            st.write("Sentiment Score:", sentiment_score)
            st.write("Sentiment Label:", sentiment_label)

        st.success("Published Date: " + news.pubDate.text)
        if c >= news_quantity:
            break


# In[8]:


def run():
    st.title("InNews: A Summarised News📰")
    image = Image.open("newspaper.png") 

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")
    category = ['--Select--', 'Trending🔥 News', 'Favourite💙 Topics', 'Search🔍 Topic']
    cat_op = st.selectbox('Select your Category', category)
    if cat_op == category[0]:
        st.warning('Please select Type!!')
    elif cat_op == category[1]:
        st.subheader("✅ Here is the Trending🔥 news for you")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        news_list = fetch_top_news()
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        av_topics = ['Choose Topic', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE',
                     'HEALTH']
        st.subheader("Choose your favourite Topic")
        chosen_topic = st.selectbox("Choose your favourite Topic", av_topics)
        if chosen_topic == av_topics[0]:
            st.warning("Please Choose the Topic")
        else:
            no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
            news_list = fetch_category_news(chosen_topic)
            if news_list:
                st.subheader("✅ Here are the some {} News for you".format(chosen_topic))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(chosen_topic))

    elif cat_op == category[3]:
        user_topic = st.text_input("Enter your Topic🔍")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=15, step=1)

        if st.button("Search") and user_topic != '':
            user_topic_pr = user_topic.replace(' ', '')
            news_list = fetch_news_search_topic(topic=user_topic_pr)
            if news_list:
                st.subheader("✅ Here are the some {} News for you".format(user_topic.capitalize()))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(user_topic))
        else:
            st.warning("Please write Topic Name to Search🔍")


run()


# In[ ]:




