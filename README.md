# log-clustering
This is a project to apply log messages clustering using BoW and K-Means clustering.
For this exercise, we started out with logs taken from a server.

# Reading log files

The logs files were generated as zipped gz files. So the files needed to be unzipped and read line by line. The log files had the following fields
1. date	
2. time	
3. severity	
4. pid	
5. process	
7. logmsg

Each Log line was read as a single string and this array of strings were then converted to dataframe to proceed with data cleaning and filtering.

First step before starting the clustering was splitting the line to columns. Data cleaning process was like below:

1- Decode utf-8 encoding t

2- Splitting str with pd.DataFrame.str.split

3- Combining the message words

# Data Cleaning

Before starting the clustering logs based on log messages, we clean the message to remove any non-aplhabetical characters and also any links. Data cleaning process was like below:

1- Get rid of all links

2- Remove non-word strings 

3- Remove words with '_'

Another column "only_words" was created from the cleaned "logmsg" column. 

# Clustering

Since error messages were relatively very less than info messages, we clustered them separately. Clustering was based on ag of Words and K-means clustering algorithm. Unsupervised clustering was first performed to create a model for error and log messages separately. 
BoW was created using Sklearn's CountVectorizer, which gave us the frequency of words in each log message. This matrix was used as feature matrix to fit the K-means clustering model. 
The clustering models are then saved as pickle files and can be used to cluster any new logs.
