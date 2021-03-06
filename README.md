# RoomRate
The COVID-19 pandemic has forced many communications to enter the virtual setting; video calls have become the new norm for meetings, interviews and even social gatherings. This has allowed an unprecedented view into everyone's homes, which means the background of one's video becomes just as important a part of how one presents oneself as fashion choices, hair styles and language. In this project, we use ratings of video backgrounds generated by public Twitter account, [@ratemyskyperoom](https://twitter.com/ratemyskyperoom), to determine what makes for good or bad video backgrounds.

**Application: Can we use text and image classification to predict the rating of a background from @ratemyskyperoom?**

## Data Processing
Run the `Data Processing.ipynb` Jupyter notebook.

The raw data is in `data/output.csv`. The resulting cleaned text data will be in `data/cleaned.csv`.

The text based columns for the tweet will be the following columns:
- `cleaned`: remove rating and links
- `no_stopwords`: remove rating and links, all lowercase, no stopwords
- `bigrams`: remove rating and links, all lowercase, no stopwords, and bigrams

The image URL will be in the `img_url` column for the image classification, and CSV of all of the image links are in `data/images.csv`

## Data Exploration
Run the `Data Exploration.ipynb` Jupyter notebook.

Run the `Data Exploration.ipynb` Jupyter notebook. Results about most common words in tweets and user engagement measured by mean likes and retweets is printed out to review.

## Text Classification
We ran Naive Bayes, SVM, Random Forest, and Logistic Regression models as text classifiers. In the following files, we train models on the data and print out the words that are the most predictive for "good" and "bad" rooms. To run the models, run
- `Non-NB Models.ipynb`
- `NaiveBayes.ipynb`
- `LogisticRegression.ipynb`

You can change which dataset the model is fit on by changing the `X` data to be either `data.cleaned` (default, cleaned dataset), `data.no_stopwords` (cleaned and removed stopwords), or `bigram` (cleaned, no stopwords, and bigrams).

## Topic Analysis
### MALLET:
We ran MALLET (using a python wrapper) to preliminarily run LDA Topic Modeling on our dataset. The program prints out groupings of words into topics.
- Install Mallet from [here](http://mallet.cs.umass.edu/download.php)
- In `TopicAnalysis/`, pip install necessary libraries from `requirements.txt`
- Run as `python3 topicanalysis.py <num_topics> ../data/cleaned.csv <path_to_mallet-2.0.8/bin/mallet>`
### LDA Topic Modeling and Logistic Regression
We also utilized the Python package, Gensim, to run LDA Topic Modeling on our dataset, and fed the topic probability outputs as features into Sklearn's Logistic Regression package.

Run the `LDA_topic_modeling.ipynb` Jupyter notebook found in the folder `TopicAnalysis/`. Make sure to pip install the following libraries in addition to those found in `requirements.txt`: 
- gensim
- sklearn

and include the file `porter.py` in the same folder as to see Stemming results.

## Image Classification

Run the `Computer Vision.ipynb` Jupyter notebook. Make sure `torch`, `pandas`,
`numpy`, `matplotlib`, `seaborn`, `torchvision`, `torchsummary`, `scikit-learn`
and `tqdm` are installed.
