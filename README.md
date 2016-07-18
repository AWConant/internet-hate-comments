# internet-hate-comments
This is an entry for the Kaggle contest ["Detecting Insults in Social Commentary"](https://www.kaggle.com/c/detecting-insults-in-social-commentary),
implemented by Andrew Conant and Fengjun Yang.

The classification program is contained in classify.py.

Usage:
    python classify.py \<training set> \<test set> \<optional: "out">

Usually we train it on training set (train.csv), and test it on one of the
two given test data sets (test_with_solutions.csv,
imperium_verification_labels.csv)

If you would like to see what the data look like after preprocessing, pass
"out" from the command line after <test set>. The files will be named as
follows:

\<training set>_data
\<test set>_data

The paper containing our final findings is paper.pdf.
