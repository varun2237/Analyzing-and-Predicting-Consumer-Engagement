# We will be using Internet News and Consumer Engagement dataset from to analyze consumer data, predict top article and popularity score.

# Statistical packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Text
import re
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from wordcloud import WordCloud, STOPWORDS

# ML model
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error,f1_score,accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score

import joblib

# Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Over sampling
import imblearn 
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import NearMiss


plt.rc('figure',figsize=(8,7.5))

A = np.random.seed(2021)

    plt.plot(A)


    plt.show()


## Creating sentimental polarity 
analyzer = SentimentIntensityAnalyzer()
def compound_score(txt):
    return analyzer.polarity_scores(txt)["compound"]

## Sentiments
def sentiment(score):
    emotion = ""
    if score >= 0.5:
        emotion = "Positive"
    elif score <= -0.5:
        emotion = "Negative"
    else:
        emotion = "Neutral"
    return emotion

## Importing CSV file
df = pd.read_csv("tripadvisor_hotel_reviews.csv")

## Applying Compound score
polarity_scores = df["Review"].astype("str").apply(compound_score)
df["Sentiment_Score"] = polarity_scores

## Applying Sentiment
df["Sentiment"] = df["Sentiment_Score"].apply(sentiment)


## Saving preprocessed file 
df.to_csv("Trip-Advisor-rating-sentiments.csv",index=False)
df.Sentiment.value_counts()

# Loading Preprocessed Dataset
# Importing the Trip-Advisor-Hotel-Review Dataset
data=pd.read_csv('Trip-Advisor-rating-sentiments.csv')

# Having a look at the data
data.head()

# Checking Missing Values
data.isna().sum()

# Countplot of Sentiments

sns.countplot(data=data,x="Sentiment",palette="pastel");

# Visualization
# Preparing data for visualization  
Viz_1 = data[['Rating','Sentiment']].value_counts().rename_axis(['Rating','Sentiment']).reset_index(name='counts')
import plotly.express as px
# Plotting the Bar Graph 
fig = px.bar(x=Viz_1.Rating, y=Viz_1.counts, color=Viz_1.Sentiment,color_discrete_sequence=px.colors.qualitative.Pastel,title="Sentiment & Ratings",labels={'x':'Ratings','y':'Total Number'})
fig.show()

#Viz2 Data preparation 
Viz_2 = data['Rating'].value_counts().rename_axis(['Rating']).reset_index(name='counts')
    
# Plotting  pie chart for ratings
fig_pie = px.pie(values=Viz_2.counts, names=Viz_2.Rating, title='Rating Distribution of the data', color_discrete_sequence = px.colors.qualitative.Pastel)
fig_pie.show()

# Jointplot on the basis of Rating and Sentiment Score of the data
jp = sns.jointplot(data=data,x='Rating',y='Sentiment_Score',kind="reg",color='#ff7373')

# jp.fig.suptitle('Jointplot on the basis of Rating and Sentiment Score of the data',fontsize=20);
import plotly.graph_objects as go
fig = go.Figure()

Ratings = [1,2,3,4,5]

for rating in Ratings:
    fig.add_trace(go.Violin(x=data['Rating'][data['Rating'] == rating],
                            y=data['Sentiment_Score'][data['Rating'] == rating],
                            name=rating,
                            box_visible=True,
                            meanline_visible=True))
fig.update_layout(
    title="Violin plot of Rating and Sentiment Score with box plot",
    xaxis_title="Rating",
    yaxis_title="Sentiment Score",
    font=dict(
        family="Courier New, monospace",
        size=12,
        
    )
)
fig.show()

## Wordcloud of Different Sentiments
from mpl_toolkits.axisartist.axislines import Subplot
import matplotlib.pyplot as plt 
import matplotlib.lines as lines 


text1 =''
for i in data[data['Sentiment']==str('Sentiment')]['Review'].values:
    text1+=i + ' '
    
wc = WordCloud(width = 800, height = 800,background_color="white",min_font_size = 10,\
    repeat=True,)
wc.generate(text1)
plt.figure( figsize = (8, 8), facecolor = None) 
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.title('Sentiment'+' Reviews',fontsize=32);


# Getting all the reviews termed positive in a single string and forming a word cloud of the string
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[14, 14], facecolor = None)

text1 =''
for i in data[data['Sentiment']=='Positive']['Review'].values:
    text1+=i + ' '

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc1 = WordCloud(width = 800, height = 800,background_color="white",min_font_size = 10,\
    repeat=True, mask=mask)
wc1.generate(text1)

ax1.axis("off")
ax1.imshow(wc1, interpolation="bilinear")
ax1.set_title('Positive Reviews',fontsize=20);

text2 =''
for i in data[data['Sentiment']=='Negative']['Review'].values:
    text2+=i + ' '

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)


wc2 = WordCloud(width = 800, height = 800,background_color="white",min_font_size = 10,\
    repeat=True, mask=mask)
wc2.generate(text2)

ax2.axis("off")
ax2.imshow(wc2, interpolation="bilinear")
ax2.set_title('Neutral Reviews',fontsize=20);

text3 =''
for i in data[data['Sentiment']=='Neutral']['Review'].values:
    text3+=i + ' '

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc3 = WordCloud(width = 800, height = 800,background_color="white",min_font_size = 10,\
    repeat=True, mask=mask)
wc3.generate(text3)

ax3.axis("off")
ax3.imshow(wc3, interpolation="bilinear")
ax3.set_title('Negetive Reviews',fontsize=20);

plt.show()


## Testing Gensim Keywords
import gensim
from gensim.summarization import keywords

input_1 =  'AI Platform Pipelines has two major parts: (1) the infrastructure for deploying and running structured AI workflows that are integrated with Google Cloud Platform services and (2) the pipeline tools for building, debugging, and sharing pipelines and components. The service runs on a Google Kubernetes cluster that’s automatically created as a part of the installation process, and it’s accessible via the Cloud AI Platform dashboard. With AI Platform Pipelines, developers specify a pipeline using the Kubeflow Pipelines software development kit (SDK), or by customizing the TensorFlow Extended (TFX) Pipeline template with the TFX SDK. This SDK compiles the pipeline and submits it to the Pipelines REST API server, which stores and schedules the pipeline for execution.' 

keywords(input_1).split("\n")

data["keywords"] = data["Review"].apply(keywords)
data["keywords"] = data["keywords"].astype("str").str.replace('\n',',',) 

words = []
for x in data.keywords.values:
    x=x.split(",")
    for i in x:
        words.append(i)
        
from collections import Counter
word_counter = Counter(words)
word_df = pd.DataFrame(np.array(list(word_counter.items())),columns=["keyword","count"])
word_df["count"] = word_df["count"].astype(int)
word_df = word_df.sort_values(['count'], ascending=False)
top_20 = word_df[0:19]
word_df.head(10)


# Barplot of Top 20 Keywords

sns.set(rc={'figure.figsize':(15,8)})
fig, ax = plt.subplots()

ax = sns.barplot(data=top_20,x="keyword",y="count",palette="pastel")
ax.patch.set_visible(False)
ax.tick_params(axis='x', labelrotation = 45)
ax.set_title("Top 20 Keywords",fontsize=20);

## Review Text Processing using NLTK
import nltk
from nltk.tokenize import word_tokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

top5 = ["hotel","room","rooms","hotels"]
for x in top5:
    data["Review"] = data["Review"].astype(str).str.replace(x,"")

data.head(2)

data2=data.copy()

def removing_stop_words(txt):
    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenizer(txt) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    return filtered_sentence
    
data2["Review"] = data2["Review"].apply(removing_stop_words)


# Making a function to lemmatize 
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import keras 

# Tokenize the reviews
tokenizer = tf.keras.preprocessing.text.Tokenizer()
lemmatizer = WordNetLemmatizer() 


def lemmatize(data):
    lema_data=[]
    for j in data:
        x=j.lower()
        x=lemmatizer.lemmatize(j,pos='n')
        x=lemmatizer.lemmatize(j,pos='v')
        x=lemmatizer.lemmatize(j,pos='a')
        x=lemmatizer.lemmatize(j,pos='r')
        x=lemmatizer.lemmatize(x)
        lema_data.append(x)
    return lema_data


data2["Review"] = data2["Review"].apply(lemmatize)

data2["Review"] = data2["Review"].apply(lambda x:" ".join(token for token in x))

data2.head(2)

X = data2["Review"].values

tokenizer = word_tokenizer()
tokenizer.fit_on_texts(X)

fig, ax = plt.subplots()
sns.set(rc={'figure.figsize':(12,9)})
length_dist = [len(x.split(" ")) for x in X]
sns.histplot(length_dist,palette="pastel")
ax.patch.set_visible(False)
ax.set_xlim(0,600)
ax.set_ylim(0,1200)
ax.set_title("Sentence length distribution",fontsize=20);
plt.show()


X = tokenizer.texts_to_sequences(X)

max_length = max([len(x) for x in X])
vocab_size = len(tokenizer.word_index)+1

print("Vocabulary size: {}".format(vocab_size))
print("max length of sentence: {}".format(max_length))


# Padding the reviews [Pads sequences to the same length.]
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=600)


# Padding the reviews [Pads sequences to the same length.]
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=600)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)

## 


## Long Short Term Memory(LSTM) :
from keras import layers as L
import tensorflow as tf

embedding_dim = 100
units = 128

model = tf.keras.Sequential([
    L.Embedding(vocab_size, int(embedding_dim), input_length=X.shape[1]),
    L.Bidirectional(L.LSTM(int(units),return_sequences=True)),
    L.Conv1D(64,3),
    L.MaxPool1D(),
    L.Flatten(),
    L.Dropout(0.2),
    L.Dense(128, activation="relu"),
    L.Dropout(0.2),
    L.Dense(64, activation="relu"),
    L.Dropout(0.2),
    L.Dense(5, activation="softmax")
])


tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


model.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam',metrics=['accuracy'] )

EPOCHS = 10
BATCH_SIZE = 32
val_split = 0.2


history = model.fit(X_train, y_train, epochs=int(EPOCHS), validation_split=float(val_split), batch_size=int(BATCH_SIZE), verbose=2)


pred = model.predict(X_test)
pred_final = np.argmax(pred,axis=-1)
pred_final

from sklearn.metrics import accuracy_score
print('Accuracy: {}%'.format(round(accuracy_score(pred_final, y_test)*100),2))


from sklearn.metrics import mean_squared_error
print("Root mean square error: {}".format(round(np.sqrt(mean_squared_error(pred_final,y_test)),3)))


model.save("Tripadvisor_BiLSTM.h5")


new_model = tf.keras.models.load_model('Tripadvisor_BiLSTM.h5')

# Check its architecture
new_model.summary()
