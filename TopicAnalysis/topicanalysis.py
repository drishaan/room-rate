# Install Mallet from http://mallet.cs.umass.edu/download.php

import little_mallet_wrapper
import pandas
import seaborn
import sys

# See https://melaniewalsh.github.io/Intro-Cultural-Analytics/Text-Analysis/Topic-Modeling-Text-Files.html
def model(num_topics, path_to_mallet, data_file):
    all_raw_cleaned_data = pandas.read_csv(data_file, usecols=['cleaned', 'rating'])
    rankings = all_raw_cleaned_data['rating'].tolist()
    raw_cleaned_data = all_raw_cleaned_data['cleaned'].tolist()
    data = []
    for text in raw_cleaned_data:
        processed_through_mallet = little_mallet_wrapper.process_string(str(text), numbers='remove')
        data.append(processed_through_mallet)

    # print stats about documents
    little_mallet_wrapper.print_dataset_stats(data)
    
    # train model
    training_data = data

    # set up output paths
    path_to_training_data = "training.txt"
    path_to_formatted_training_data = "training"
    path_to_model = "model"
    path_to_topic_keys = "topic_keys"
    path_to_topic_distributions = "topic_distributions"

    # import data
    little_mallet_wrapper.import_data(path_to_mallet,
                path_to_training_data,
                path_to_formatted_training_data,
                training_data)
    
    # train topic model
    little_mallet_wrapper.train_topic_model(path_to_mallet,
                      path_to_formatted_training_data,
                      path_to_model,
                      path_to_topic_keys,
                      path_to_topic_distributions,
                      num_topics)


# Run as python3 topicanalysis.py 50 ../data/cleaned.csv ../mallet-2.0.8/bin/mallet
def main():
    # Read command line arg to get directory of documents
    try:
        num_topics = int(sys.argv[1])
        data_file = sys.argv[2]
        path_to_mallet = sys.argv[3]
    except:
        print("Error: need to provide number of topics (int), path to data filve (csv), and path to Java's MALLET")
        sys.exit(2)

    model(num_topics, path_to_mallet, data_file)

    topics = little_mallet_wrapper.load_topic_keys(f"topic_keys")
    for topic_number, topic in enumerate(topics):
        print(f"Topic {topic_number}: {topic}\n\n")



if __name__ == "__main__":
    main()