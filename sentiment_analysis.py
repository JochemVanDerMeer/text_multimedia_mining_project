import spacy
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import defaultdict

#df = pd.read_excel('capstone_airline_reviews_untouched.xlsx')
df = pd.read_excel('small_airline_reviews.xlsx')
nlp = spacy.load("en_core_web_sm")
query_words_seatcomfort = ['seatcomfort', 'seat comfort', 'comfortable', 'comfy', 'seat', 'chair', 'legroom', 'leg room']
query_words_cabinservice = ['cabin service', 'inflight service', 'in-flight service', 'cabin-service', 'service in the cabin', 'flight service', 'onboard service', 'on-board service', 'flight crew', 'cabin crew']
query_words_foodbev = ['food', 'meal', 'drink', 'beverage', 'foodservice', 'mealservice', 'cafe', 'food service']
query_words_entertainment = ['movie', 'screen', 'entertainment', 'film']
query_words_groundservice = ['groundservice', 'ground service', 'service on the ground', 'check-in', 'check in']
query_words_valueformoney = ['value for money', 'good value', 'cheap', 'expensive', 'price', 'fare', 'cost']
low_cost_airlines = ['Air Arabia', 'AirAsia', 'easyJet', 'Eurowings', 'flydubai', 'Frontier Airlines', 'Germanwings', 'IndiGo', 'Jetblue Airways', 'Norwegian', 'Pegasus Airlines', 'Ryanair', 'Southwest Airlines', 'Spirit Airlines', 'Sunwing Airilnes', 'Virgin America', 'Vueling Airlines', 'Wizz Air', 'WOW air']

punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
airlines_list = []

airlines_x_train = []
airlines_x_test = []
reviews_x_train = []
reviews_x_test = []

le = preprocessing.LabelEncoder()

def clean_df(df):
    new_df = df.dropna(subset=['customer_review'])
    return new_df

def specify_label(label):
    #instead of having 1,2,3,4 or 5 stars, a review is labeled as positive, negative or neutral.
    if int(label) == 1 or int(label) == 2 or int(label) == 3:
        return -1
    if int(label) == 4 or int(label) == 5:
        return 1
    else:
        return 0

def get_sentences(service_aspect, query_words):
    res = []
    new_df = clean_df(df)
    for idx, row in new_df.iterrows():
        if str(row['customer_review']) != 'nan' and str(row[service_aspect]) != 'nan':
            for queryword in query_words:
                for idx, val in enumerate(sent_tokenize(row['customer_review'])):
                    if queryword in val:
                        label = specify_label(row[service_aspect])
                        res.append([idx, sent_tokenize(row['customer_review'])[idx], label, row['airline']])
                        #res.append([idx, sent_tokenize(row['customer_review'])[idx], row['cabin_service']])
                        break
    return res

def preprocess_sentences(ls):
    # stop words are removed, converted to lowercase and lemmatization is applied
    for i in ls:
        i[1] = nlp(i[1])
        tokens = [t for t in i[1]]
        tokens = [t for t in tokens if not t.is_stop]
        tokens = [(t.lemma_).lower() for t in tokens]
        i[1] = tokens
    return ls

def show_query_results(service_aspect, query_words):
    # shows the result per sentence that contains a matched query
    for i in preprocess_sentences(get_sentences(service_aspect, query_words)):
        for j in i:
            print(j)
        print('\n\n')       

def flatten(l):
    #used to flatten a list
    return [item for sublist in l for item in sublist]

def create_x_and_y(ls):
    #creating x (sentences) and y (sentiment labels)
    x_set = []
    y_set = []
    for i in ls:
        x_set.append([i[1], i[3]])
        y_set.append(i[2])
    return x_set, y_set

def split_train_test(service_aspect, query_words):
    #splitting the data in train set and test set
    x_set, y_set = create_x_and_y(preprocess_sentences(get_sentences(service_aspect, query_words)))
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set)
    for i in x_train:
        reviews_x_train.append((i[0]))
        airlines_x_train.append(flatten(i[1]))
    for j in x_test:
        reviews_x_test.append((j[0]))
        airlines_x_test.append(flatten(j[1]))
    return reviews_x_train, reviews_x_test, y_train, y_test

def padding_or_truncate_set(set):
    #normalizing the reviews, as each review gets the same length through either padding or truncating
    N = 20
    for i in set:  
        i += [''] * (N - len(i))

    for i in range(len(set)):
        set[i] = le.fit_transform(set[i])

    for i in range(len(set)):
        if len(set[i]) >= N:
            set[i] = set[i][0:N]
    return set

def get_results_per_airline(ls):
    #this returns a dictionary with the number of sentiments found and the cumulative sentiment score
    res = defaultdict(int)
    for i in ls:
        res[i[0]] += i[1]
        sizestring = i[0] + '_size'
        res[sizestring] += 1
    return dict(res)

def compute_average_sentiment(res_dict):
    #this returns the average sentiment per airline
    res = {}
    for i in low_cost_airlines:
        if i in res_dict:
            size_string = i + '_size'
            res[i] = res_dict[i] / res_dict[size_string]
    return res

def sentiment_analysis(service_aspect, query_words):
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    x_train, x_test, y_train, y_test = split_train_test(service_aspect, query_words)
    x_train = padding_or_truncate_set(x_train)
    x_test = padding_or_truncate_set(x_test)
    ada.fit(x_train, y_train)
    prediction = ada.predict(x_test)
    print(classification_report(y_test, prediction))
    airlines_x_test1 = ["".join(x) for x in airlines_x_test]
    predictions_airlines = list(zip(airlines_x_test1, prediction))
    res = [prediction for prediction in predictions_airlines if prediction[0] in low_cost_airlines]
    #print(get_results_per_airline(res))
    print(compute_average_sentiment(get_results_per_airline(res)))




sentiment_analysis('seat_comfort', query_words_seatcomfort)
#sentiment_analysis('cabin_service', query_words_cabinservice)
#sentiment_analysis('food_bev', query_words_foodbev)
#sentiment_analysis('entertainment', query_words_entertainment)
#sentiment_analysis('ground_service', query_words_groundservice)
#sentiment_analysis('value_for_money', query_words_valueformoney)
