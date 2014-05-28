Notes for the presentation
==========================

Introduction
------------

- Information Overload -
With hundreds of terabytes of data existing on the internet it is becoming harder and harder to access the data in a useful and orderly way. This introduces the need for new tools and algorithms to organise and search more effectively.
An intuitive way to organise documents is to group them by topic. However, since topics are abstract and one document can have many topics, this is a difficult task for a computer to automatically perform.

- Topics - 
Topic modeling aims to uncover hidden thematic structure within documents. Topics, in this case, are defined as a probability distribution of terms over the vocabulary of a corpus. Here, we have four topics, and words with a high probability are written in a large font and words with a low probability are written in a small font.


Topic Modelling
----------------

- Model of topics -
Each topic is modelled as a distribution over words.

- Model of Documents -
Each document is modelled as a distribution over topics

- Topic Modelling -
The generative process for a document begins by drawing the documents topic distribution. Then, for each position, we sample a topic assignment and a word. For each topic, we can combine the column vectors of its word distribution into a word-topic matrix.

- Word-topic Matrix -
The word-topic matrix gives us an idea of which words best describe each topic and which words are common overall. For example, the word 'online' appears in multiple topics, so would not be very useful for use in a specific description of a single topic. The aim of the task is to find a distribution of topics for each document.

- Approximate Inference & Provable Guarantees-
Posterior inference of document-topic (and topic-word) distributions is very difficult. In the worst case, it is NP-hard. This has lead researchers to use approximate inference techniques such as Singular Value Decomposition, variational inference and Markov Chain Monte Carlo. Many current approaches of these techniques are heuristic: we cannot prove good bounds on either their performance or their running time. So recent work has been on designing provably polynomial-time algorithms. They assume that each document was generated using the hypothesised model and the task of topic modelling is to statistically recover the parameters of the model.

Algorithm
---------

- Algorithm -
This paper is based off the algorithm the authors wrote in 2012. The algorithm takes the second order moment matrix of word-word co-occurrences as the input. It then finds anchor words for each topic and uses them to reconstruct topic distributions.

- Anchor Words -
In order to understand this algorithm fully, we must first go into the definition of anchor words.


###################
- Steps -
The algorithm usually proceeds as the following:
1) Treat data as observations resulting from a probabilistic process with hidden variables
2) Infer hidden structure with posterior inference
3) Situate new data into estimated model



