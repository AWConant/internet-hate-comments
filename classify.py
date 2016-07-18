#!/usr/bin/env python

##############################################
# Preprocess the file:
#  - convert from unicode to python string
#  - vectorize the strings(feature extraction)
# Classify the input:
#  - Random Forest Classifier
#  - Naive Bayes Classifier
#  - Support Vector Machine
#  - Perceptron for Ensemble learning
#############################################

from __future__ import division
import warnings
import sys

# For preprocessing
import csv
import re
from unidecode import unidecode

# For vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# Clustering
from sklearn.cluster import KMeans


def ireplace(haystack, old, new):
    """
    Replace old with new in a string insensitive to case.
    Written by aland on stackoverflow.
    - Input: string to be searched, string to replace, string to replace with
    - Return: replaced string
    """
    comp_old = re.compile(re.escape(old), re.IGNORECASE)
    return comp_old.sub(new, haystack)

def read_input(filename):
    """
    Read data files and store texts and labels
    Parse and Cache the result
    - Input: filename, true/false write output to files?
    - Return: data(list of texts), labels(list of labels) 
    """
    try:
        infile = open(filename, 'r')
    except IOError:
        print 'Error:', filename, 'could not be opened.'
        sys.exit()

    read = csv.reader(infile)

    format_line = read.next() # This line contains the format of the file
    l_num = format_line.index('Insult')
    t_num = format_line.index('Comment')

    data = []
    labels = []

    for row in read:
        # Decode comment body unicode
        body = row[t_num]
        body = body.decode('unicode_escape')
        body = unidecode(body)

        # Remove surrounding quotes
        body = body[1:-1]

        # Add a space to make substitutions easier
        body = ' ' + body

        # Remove backslash-escaped things
        body = ireplace(body, '\r', ' ')
        body = ireplace(body, '\n', ' ')
        body = ireplace(body, '\'', '')

        # Cut down 3+ consecutive same characters to 2
        # Credit to 'Howard' from stackoverflow for this sorcery.
        body = re.sub(r'(.)\1+', r'\1\1', body)

        # Remove angle brackets and contained text
        # Credit to Paulo Scardine on stackoverflow
        #body = re.sub(r'<.+?>s*', '', body)

        # Remove punctuation
        body = ireplace(body, '!', ' ')
        body = ireplace(body, '.', ' ')
        body = ireplace(body, '_', ' ')
        body = ireplace(body, '-', ' ')
        body = ireplace(body, '&', 'and')
        body = ireplace(body, '%', 'percent')

        # Remove contractions
        body = ireplace(body, ' isnt ', ' is not ')
        body = ireplace(body, ' hes ', ' he is ')
        body = ireplace(body, ' youd ', ' you would ')
        body = ireplace(body, ' shes', ' she is ')
        body = ireplace(body, ' gonna ', ' going to ')
        body = ireplace(body, ' didnt ', ' did not ')
        body = ireplace(body, ' couldve ', ' could have ')
        body = ireplace(body, ' wouldnt ', ' would not ')
        body = ireplace(body, ' cant ', ' can not ')
        body = ireplace(body, ' arent ', ' are not ')
        body = ireplace(body, ' aint ', ' am not ')
        body = ireplace(body, ' dont ', ' do not ')
        body = ireplace(body, ' wont ', ' will not ')
        body = ireplace(body, ' theyd ', ' they would ')
        body = ireplace(body, ' theyve ', ' they have ')
        body = ireplace(body, ' youre ', ' you are ')
        body = ireplace(body, ' wanna ', ' want to ')
        body = ireplace(body, ' ive ', ' I have ')
        body = ireplace(body, ' its ', ' it is ')
        body = ireplace(body, ' im ', ' i am ')
        body = ireplace(body, ' thatll ', ' that will ')

        # Common misspellings
        body = ireplace(body, ' i ', ' I ')
        body = ireplace(body, ' u ', ' you ')
        body = ireplace(body, ' ur ', ' your ')
        body = ireplace(body, ' wtf ', ' what the fuck ')
        body = ireplace(body, ' tbh ', ' to be honest ')
        body = ireplace(body, ' imo ', ' in my opinion ')

        # Emoticons
        body = ireplace(body, ':)', 'smiley')
        body = ireplace(body, ': )', 'smiley')
        body = ireplace(body, ':(', 'frownie')
        body = ireplace(body, ': (', 'frownie')

        data.append(body)
        labels.append(int(row[l_num]))

    infile.close()
    return data, labels

def write_parsed_data(filename, data, labels):
    """
    Writes parsed comment bodies and labels to an outfile
    - Input: filename, list of comment bodies, list of integer labels
    - Return: None
    """
    outfile = filename[:-4] + '_data'

    with open(outfile, 'w') as datafile:
        print 'Writing comment bodies to file', outfile + "..."
        for label, line in zip(labels, data):
            datafile.write(str(label) + ', ' + str(line) + '\n')

def vectorize(data, element='word'):
    """
    To perform classification, first need to convert text into vectors
    We use a CountVectorizer in sklearn, which counts the instances of
    each word
    - Input: data(list of texts)
    - Return: (vectorized data, sklearn vectorizer) tuple
    """

    # Ignore obnoxious numpy deprecation warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Use vectorizers to vectorize texts
        if element == 'word':
            vect = CountVectorizer(ngram_range=(1,2)) 
        else:
            vect = CountVectorizer(ngram_range=(3,7), analyzer='char_wb') 
        X = vect.fit_transform(data)

        return X, vect

def validate(clfs, clf_names, test_data, test_label):
    """
    Run a classifier over test set and count its accuracy
    """
    count = [0] * len(clfs)

    for text, label in zip(test_data, test_label):
        for i, clf in enumerate(clfs):
            result = clf.predict(text)[0]
            if result == label:
                count[i] += 1
    count = [c / len(test_label) for c in count]

    for name, c in zip(clf_names, count):
        print "The accuracy of %s classifier is %.1f%%" % (name, c * 100)

    return count

def trainClfs(clfs, clf_names, data, label):
    """Train a set of classifiers with given data and label"""
    for clf, clf_name in zip(clfs, clf_names):
        print "Training %s Classifier..." % (clf_name),
        sys.stdout.flush()
        try:
            clf = clf.fit(data, label)
        except TypeError:
            clf = clf.fit(data.toarray(), label)
        print "Trained!"

    return clfs

def trainEnsemble(word_clfs, char_clfs, word_data, char_data, train_label, layers):
    """
    Train a perceptron to combine the results of multiple classifiers
    Example of ensemble learning
    """
    nn = MLPClassifier(algorithm='l-bfgs', \
                       alpha=1e-5, \
                       hidden_layer_sizes=layers, \
                       random_state=1)
    nn_input = []
    for word_text, char_text in zip(word_data, char_data):
        word = [clf.predict(word_text.toarray())[0] for clf in word_clfs]
        char = [clf.predict(char_text.toarray())[0] for clf in char_clfs]
        nn_input.append(word + char)
    nn.fit(nn_input, train_label)

    return nn 

def validateEnsemble(ens, word_clfs, char_clfs, word_data, char_data, test_label):
    """Calculate the accuracy of ensemble on the test data set"""
    count = 0

    for word_text, char_text, label in zip(word_data, char_data, test_label):
        word = [clf.predict(word_text)[0] for clf in word_clfs]
        char = [clf.predict(char_text)[0] for clf in char_clfs]
        if ens.predict([word + char])[0] == label:
            count += 1

    print "The accuracy of ensemble is %.1f%%" % (count/len(word_data) * 100)

def main():
    # Check for formedness of input
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print ('Usage: python classify.py <training dataset>'
               '<test dataset> <opt: "out">')
        sys.exit()

    # Read command line arguments
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    try:
        write_output = sys.argv[3]
    except IndexError:
        write_output = ''

    # Read input files
    print '-' * 40
    print "Reading inputs..."
    print "Training set:\t %s" % (train_filename)
    train_data, train_label = read_input(train_filename)
    print "Test set:\t %s" % (test_filename)
    test_data, test_label = read_input(test_filename)

    # Optionally write parsed data to outfiles
    if write_output == 'out':
        write_parsed_data(train_filename, train_data, train_label)
        write_parsed_data(test_filename, test_data, test_label)

    # Vectorize texts
    print '-' * 40
    print "Vectorizing inputs...",
    # Word n-grams
    X_word, vect_word = vectorize(train_data, 'word')
    test_X_word = [vect_word.transform([text]).toarray() for text in test_data]
    # char n-grams
    X_char, vect_char = vectorize(train_data, 'char_wb')
    test_X_char = [vect_char.transform([text]).toarray() for text in test_data]
    print 'Done!'

    # Train classifiers
    print '-' * 40
    print "Training Classifiers..."
    word_clfs = []
    word_clf_names = []
    char_clfs = []
    char_clf_names = []

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=9)
    word_clfs.append(rfc)
    word_clf_names.append("Random Forest")

    # Naive Bayes Classifier
    mnb = MultinomialNB()
    word_clfs.append(mnb)
    word_clf_names.append("Multinomial Naive Bayes")

    # SVM with word n-grams
    # Kernel can be 'linear', 'poly', 'rbf', or 'sigmoid'
    # Linear outperforms other kernels in our test
    svc_word = svm.SVC(kernel='linear')
    word_clfs.append(svc_word)
    word_clf_names.append("SVC Word-grams")

    # SVM with character n-grams
    svc_char = svm.SVC(kernel='linear')
    char_clfs.append(svc_char)
    char_clf_names.append('SVC Char-grams')

    word_clfs = trainClfs(word_clfs, word_clf_names, X_word, train_label)
    char_clfs = trainClfs(char_clfs, char_clf_names, X_char, train_label)

    # Validate results on the test data set
    print '-' * 40
    print "Validating Results..."
    validate(word_clfs, word_clf_names, test_X_word, test_label)
    #validate(char_clfs, char_clf_names, test_X_char, test_label)

    ## Build a perceptron ensemble
    #print '-' * 40
    #print "Training Perceptron as ensemble learning...",
    #nn = trainEnsemble(word_clfs, char_clfs, X_word, X_char, train_label, ())
    #print 'Done!'

    ## Validate the trained perceptron
    #print '-' * 40
    #print "Validating ensemble..."
    #validateEnsemble(nn, word_clfs, char_clfs, test_X_word, test_X_char, test_label)

main()
