"""
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# set up


# getting data
mkdir reference
cd reference
git clone https://github.com/scikit-learn/scikit-learn.git
cd -

mkdir data
cd data
python reference/scikit-learn/doc/tutorial/text_analytics/data/languages/fetch_data.py
cd -
"""

from sklearn.datasets import fetch_20newsgroups

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
twenty_train = fetch_20newsgroups(
    data_home="data",
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=1,
)

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


from sklearn.pipeline import Pipeline

text_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()),
    ]
)
text_clf.fit(twenty_train.data, twenty_train.target)
# How to use it
# text_clf.predict_proba(["God is love"]) --> [3]
# twenty_train.target_names[3] --> 'soc.religion.christian'



# Evaluation of models
import numpy as np
twenty_test = fetch_20newsgroups(
    data_home="data", categories=categories, shuffle=True, random_state=1
)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)



# using SVM 
from sklearn.linear_model import SGDClassifier
svm_text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
svm_text_clf.fit(twenty_train.data, twenty_train.target)
svm_predicted = svm_text_clf.predict(docs_test)
np.mean(svm_predicted == twenty_test.target)


from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))

metrics.confusion_matrix(twenty_test.target, predicted)


## Grid Search parameters
from sklearn.model_selection import GridSearchCV

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
# twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
# gs_clf.best_score_
# gs_clf.best_params_
# {'clf__alpha': 0.01, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}
# gs_clf.cv_results_ --> pandas result
pass
