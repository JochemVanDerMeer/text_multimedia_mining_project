import spacy
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel('small_airline_reviews.xlsx')
nlp = spacy.load("en_core_web_sm")
query_words_cabinservice = ['cabin service', 'inflight service', 'in-flight service', 'cabin-service', 'service in the cabin', 'flight service', 'onboard service', 'on-board service', 'flight crew', 'cabin crew']
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

le = preprocessing.LabelEncoder()

def clean_df(df):
    new_df = df.dropna(subset=['customer_review'])
    return new_df

def specify_label(label):
    #instead of having 1,2,3,4 or 5 stars, a review is labeled as positive, negative or neutral.
    if int(label) == 1 or int(label) == 2:
        return -1
    if int(label) == 3:
        return 0
    if int(label) == 4 or int(label) == 5:
        return 1
    else:
        return 0

def get_sentences():
    res = []
    new_df = clean_df(df)
    for idx, row in new_df.iterrows():
        if str(row['customer_review']) != 'nan' and str(row['cabin_service']) != 'nan':
            for queryword in query_words_cabinservice:
                for idx, val in enumerate(sent_tokenize(row['customer_review'])):
                    if queryword in val:
                        label = specify_label(row['cabin_service'])
                        res.append([idx, sent_tokenize(row['customer_review'])[idx], label])
                        break
    return res

#todo - extend also for the other categories that were rated

def preprocess_sentences(ls):
    # stop words are removed, converted to lowercase and lemmatization is applied
    for i in ls:
        i[1] = nlp(i[1])
        tokens = [t for t in i[1]]
        tokens = [t for t in tokens if not t.is_stop]
        tokens = [(t.lemma_).lower() for t in tokens]
        i[1] = tokens
    return ls

def show_results():
    # shows the result per sentence that contains a matched query
    for i in preprocess_sentences(get_sentences()):
        for j in i:
            print(j)
        print('\n\n')       

def create_x_and_y(ls):
    #creating x (sentences) and y (sentiment labels)
    x_set = []
    y_set = []
    for i in ls:
        x_set.append(i[1])
        y_set.append(i[2])
    return x_set, y_set

def split_train_test():
    #splitting the data in train set and test set
    x_set, y_set = create_x_and_y(preprocess_sentences(get_sentences()))
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set)
    return x_train, x_test, y_train, y_test

def padding_or_truncate_set(set):
    N = 20
    for i in set:  
        i += [''] * (N - len(i))

    for i in range(len(set)):
        set[i] = le.fit_transform(set[i])

    for i in range(len(set)):
        if len(set[i]) >= N:
            set[i] = set[i][0:N]
    return set

def sentiment_analysis():
    logisticRegr = LogisticRegression()
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    x_train, x_test, y_train, y_test = split_train_test()
    x_train = padding_or_truncate_set(x_train)
    x_test = padding_or_truncate_set(x_test)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    print(classification_report(y_test, prediction))

sentiment_analysis()

