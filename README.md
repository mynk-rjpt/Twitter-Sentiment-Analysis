# Twitter-Sentiment-Analysis
## Twitter Sentiment Analysis using Machine Learning
### Dataset Information
The dataset used in this project is 'training.1600000.processed.noemoticon.csv' from Kaggle.
This is the sentiment140 dataset.
It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment .
It contains the following 6 fields:

1. target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
2. ids: The id of the tweet ( 2087)
3. date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
4. flag: The query (lyx). If there is no query, then this value is NO_QUERY.
5. user: the user that tweeted (robotickilldozr)
6. text: the text of the tweet (Lyx is cool)

### Requirements

There are some general library requirements for the project and some which are specific to
individual methods. The general requirements are as follows.   
● Numpy, Pandas    
● scikit-learn     
● scipy      
● nltk     

The library requirements specific to some methods are:  
● Multinomial Naive Bayes  
● SVM  
● Logistic Regression  
● xgboost for XGBoost  
● Linear SVC  
● Random Forest  

### Structure of Code

#### Importing Required Libraries
1. pandas,numpy,nltk,re,future,matplotlib.pyplot
2. train_test_split,GridSearchCV
3. CountVectorizer, TfidfVectorizer
4. TfidfTransformer
5. BernoulliNB, MultinomialNB
6. metrics,roc_auc_score
7. accuracy_score,label_binarize,LogisticRegression
8. Pipeline,svm,LinearSVC,SVR,
9. RandomForestClassifier,DecisionTreeClassifier
10. BeautifulSoup,stopwords,SnowballStemmer

#### Reading CSV file
11. Mounting from google drive  
12. Using pd.read_csv and encoding latin
13. Showing df.head()

#### Preprocessing
14. Lowercasing the letter
15. Removing Usernames
16. Removing URLs
17. Removing all digits
18. Removing Quotations
19. Replacing Emojis with their corresponding sentiment part eg : positive emoji or negative emoji
20. Replacing contractions
21. Removing punctuations
22. Replacing double spaces with single spaces
23. Plotts
24. Word clouds

#### Using Classifiers
25. Used Count Vectorizer
26. Used Multinomial Naive Bayes
27. Used Linear SVC
28. Used Logistic Regression
29. Used SVM
30. Using Decision Trees
31. Using Xgb

### Process of Execution

One has to simply open the colab file and keep on running all the cells. Give path for
reading the csv file. The first 3 classifiers are showing the best results. In order to compare performance, the user can run other classifiers too.






