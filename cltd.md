---
bibliography:
- references.bib
date: August 2020
title: 'Cross-lingual toxicity detection'
---

**Cross-lingual Toxicity Detection**

Loïs Bilat\
**Master Thesis**\
École polytechnique fédérale de Lausanne\
*School of Computer and Communication Sciences*\
**Supervisor at EPFL**\
Jean-Cédric Chappelier\
![image](images/epfl.png){width="20%"}

**Supervisors at ELCA Informatique SA**\
Peter Sperisen\
Matthias Ramirez\
![image](images/elca.png){width="20%"}

August 2020

Abstract {#abstract .unnumbered}
========

With the increasing use of social media, there is a critical need for
performant automatic moderation tools. In this thesis, we present
advanced classifiers that can detect hateful and offensive content in
short texts. We study various architectures based on transformer models
such as BERT and evaluate multiple changes to those models that improve
their performance. We then tackle cross-lingual classification and
introduce new architectures that use joint-learning and data
translation. Our models are able to outperform existing multilingual
models on zero-shot and multilingual classification.

Acknowledgements {#acknowledgements .unnumbered}
================

I would like to thank everyone without whom this project would not have
been possible. To begin with, I want to thank Peter Sperisen and
Matthias Ramirez, my supervisors at ELCA, for their continuous guidance
and support during those 6 months, as well as the rest of the ELCA Data
Science team for their warm welcome and the interesting conversations we
had. I also wish to thank Jean-Cédric Chappelier for expressing interest
in this project and accepting to supervise me, and for his valuable
oversight of this thesis. Finally, I would like to thank my family and
friends for all their unconditional support during these last five
years.

Disclaimer {#disclaimer .unnumbered}
==========

Because of the nature of this work, we need to work with offensive and
hateful content directly, content that might be shocking and hurtful to
some readers. Examples of such content are presented in the report for
illustration purposes, and while the more offensive and obscene words
have been partially censored, it is still quite easy to guess the
original content. Most of the examples are taken directly from available
datasets, but some were created specifically for this report. Those
examples do not reflect the views of the authors.

Introduction
============

Motivation
----------

With the rise in popularity of social media in the past decade, an
increasing number of people are exposed to toxic and hateful comments on
the Internet, a phenomenon often amplified by the anonymity provided by
some platforms. [^1] It is crucial for the moderators of these platforms
to have an efficient way to filter and remove toxic content, for the
well-being of the users, and the moderators themselves that are often
left with psychological scars from the manual moderation of such
content. [^2]

In the past few years, we have seen increasing pressure on social media
platforms to be better at removing hateful content, with some
governments pushing for more regulations and accountability. A few
countries have tried implementing laws to force social media platforms
to report and remove hateful content or risk fines otherwise, but such
initiatives are often very controversial and considered by their
opponents as anti-constitutional and against freedom of speech. [^3]
[^4]

Fully manual moderation is not possible, for the psychological reasons
cited earlier, but also due to the sheer amount of content posted every
day. Automatic or semi-automatic moderation tools can be of huge help
and make the Internet a safer place. The largest problem when developing
such systems is to find enough data to train them. Unfortunately, there
are not a lot of resources available, and the vast majority of
obtainable data are in English, making this task even more difficult for
other languages.

In this project, we will implement models that can automatically detect
hateful and offensive content, and then try to transfer the knowledge
acquired from one language to other languages with fewer or no data. We
will study different categories of models, and starting from the best
existing models we will work towards improving them and applying them
successfully to a cross-lingual context.

Problem Statement {#problem_statement}
-----------------

We want to classify short messages into three categories: *hate speech*,
*offensive content* and *none*. These messages can originate from
various sources, such as social media, online chats or comments on
websites. We are only interested in the content of the message, but not
the metadata or other additional information.

We use the definitions from @charitidis2019countering for the three
categories:

-   `hate`: any message that fulfils the following two criteria: it
    should target a person or a group, and it must contain a hateful
    attack; a hateful attack can be violent speech, support for death,
    harm or disease, a statement of inferiority regarding a group they
    identify with or a call for segregation (e.g. racism, homophobia,
    Islamophobia, sexism);

-   `offensive`: any message that contains a personal attack,
    harassment, or insults;

-   `none`: any message that does not belong to one of the former two
    categories.

To illustrate these definitions, we show in table
[1.1](#tab:sentences){reference-type="ref" reference="tab:sentences"} a
few examples of sentences taken from various hate speech and toxic
English datasets.

::: {#tab:sentences}
  -- --------------------------------------------------------------------
     \@user May he burn in hell eternally.
     \@user Hang them and let god sort them out
     \@user \@user F\*ck that n\*gga
     \@user Build the wall! Immigrants are bring in the disease
     \@user What a fat c\*nt
     \@user This low life will do anything to hang on to power!
     \@user \@user These people are idiots. Truly.
     ur trash!
     \@user \@user \@user \@user I'm playing the lotto.
     \@user \@user pls cancel this appointment.
     RT \@user: \@user \@user You win the Twitter award tonight!! IMHO.
     \@user We're f\*cking tired.
  -- --------------------------------------------------------------------

  : [\[tab:sentences\]]{#tab:sentences label="tab:sentences"}Example of
  content from each class. Censorship was added for the report and is
  not present in the data. Mentions were simplified for clarity.
:::

In other works on the same subject, similar definitions are often used,
although some variations are not uncommon. The datasets selected for
this project (presented in section [3.1](#data){reference-type="ref"
reference="data"}) all use equivalent definitions.

Objectives
----------

There are two main objectives to this thesis:

**Monolingual Toxicity Detection**: The first objective is to implement
performant models that can classify messages from a given language into
one of the three classes presented in section
[1.2](#problem_statement){reference-type="ref"
reference="problem_statement"}. We will develop classifiers for English,
French and German data using advanced Neural Network based techniques,
and try to improve existing methods and architectures. Each model should
perform well on all three classes. We present this work in chapter
[4](#monolingual){reference-type="ref" reference="monolingual"}.

**Cross-lingual Toxicity Detection**: The second objective is to design
multilingual models that can classify messages in multiple languages.
They should perform similarly to monolingual models when evaluated on
languages with training data available, but they should also work on
languages without training data. This aspect is developed in chapter
[5](#cross_lingual){reference-type="ref" reference="cross_lingual"}.

Related work
============

There were three main steps in the development of toxic content
classification; it started with simple models that were trained on
individual datasets for the specific toxicity detection task, then
transfer learning methods were introduced to leverage information from
models already trained on more general data, and finally, multiple
studies experimented with multilingual classification to transfer
knowledge to languages with no training data.

Toxicity Detection
------------------

Traditional toxic content detection methods are built on algorithms such
as Logistic Regression and Support Vector Machines
[@davidson2017automated; @ousidhoum2019multilingual; @van-aken-etal-2018-challenges; @waseem-hovy-2016-hateful].
A feature vector is created from the text, usually by simply considering
the textual content, and encoding it into a vector (bag of words,
tf-idf) using word n-grams as well as character n-grams
[@davidson2017automated; @ousidhoum2019multilingual; @van-aken-etal-2018-challenges; @waseem-hovy-2016-hateful].
When available, external features can also be added (for instance with
tweets, the number of mentions or retweets can be used
[@davidson2017automated]) and are usually quite useful, but too reliant
on the type of data used and not applicable to text coming from more
diverse sources. More advanced features have also been used
[@davidson2017automated], such as part-of-speech tags, Flesch-Kincaid
Grade Level, Flesch Reading ease scores [^5] and sentiment score.
Unfortunately, such methods are very sensitive to the presence of some
keywords and are not able to see long-term dependencies in the text,
resulting in overall mixed results.

Neural Network approaches such as convolutional neural networks (CNN)
and long short-term memory (LSTM) networks were also applied to the
toxic content detection task [@Badjatiya_2017], as these architectures
are able to act as n-gram feature extractors. Mixes of convolutional and
recurrent models were also tried, using a model were the output of a CNN
was fed into a gated recurrent unit (GRU) layer [@zhangconvgru].
@van-aken-etal-2018-challenges added attention layers to LSTM and GRU
networks and experimented with ensemble learning. They also perform an
in-depth error analysis to understand the primary causes of
misclassification. Most errors come from ambiguous sentences (for
instance with irony or sarcasm, metaphors, rhetorical questions or rare
words), but a lot of errors also come from inconsistencies in the
annotations, and the presence of some specific keywords that drastically
influence the perceived offensiveness of the sentence.
@charitidis2019countering compared multiple models on a newly created
dataset, and @gamback-sikdar-2017-using experimented with different word
embeddings for a CNN network. Using these types of models was for a long
time the best way to perform toxic content detection, but newer methods
based on transfer learning and more complex architectures are now
consistently better. We will still take time to evaluate some of these
models and use ideas from those papers to improve the newer
architectures.

Transfer Learning
-----------------

Over time, the amount of available data for various natural language
processing tasks rapidly increased, and the models utilising those data
were getting more complex. To spare some resources, it is possible to
reuse already trained components in the target model instead of training
them completely from scratch. This is convenient to reduce the training
time, but also to improve the quality of the model that will have a
better language model representation.

Widespread use of pretrained components in natural language processing
started with the introduction of pretrained word embeddings such as
word2vec [@mikolov2013efficient], GloVe [@pennington2014glove] and
FastText [@bojanowski2016enriching]. Those were trained on large amounts
of data and could be used in a multitude of tasks, saving computation
and time resources that would have been required to create completely
new word embeddings. With those embeddings it became easier and quicker
to create complex language models, as those models were able to detect
similarities between words, unlike previous bag-of-word based methods
(words with a similar meaning or use are often close to each other in
the embedding space). However, they lack context awareness which causes
some issues with homonyms that can have different meanings depending on
their context.

With ELMo [@peters2018], a bidirectional LSTM network was used to create
contextual word embeddings that could then be used for any language
processing task. BERT (@devlin2018bert) followed with the same idea but
used stacked Transformer encoders (@vaswani2017attention), which became
state-of-the-art on eleven different language processing tasks. Many
variants followed, such as RoBERTa [@liu2019roberta], an optimised
version of BERT trained with more data, or XLNet [@yang2019xlnet] that
improves BERT by using permutations instead of masked tokens.

Iterating on models similar to BERT, various classifiers were added to
extract important features [@mozafari2019bertbased]. This process is
called fine-tuning, where during the training of the model the added
classifier and the BERT model are updated simultaneously to better fit
the data. Diverse pretraining methods were used to improve the relevance
of the BERT model before using it for fine-tuning [@sun2019finetune]; it
has been shown that continuing the original training on data relevant to
the target task can be useful in some cases.

Since the introduction of BERT, lots of researchers have focused on
improving the original version of BERT, either by developing new models
based thereon or by finding better ways to use the embeddings created by
it. Large research teams and companies regularly come up with new models
based on BERT, with more and more parameters, training data, and
optimisations. [^6] In this project we will study the second approach,
and using some ideas from previous research will try to discover better
ways to utilise pre-trained models.

Multilingual toxicity detection
-------------------------------

Most research in natural language processing is carried out on English
data, and it is hard to find resources for other languages. Multilingual
text classification was developed to allow for the creation of models in
languages with few or no training data without the need for manual data
annotation, a costly process. There are two main approaches to this
problem:

**Translation-based methods**: The first approach is to translate all
the data into a single language with many resources, and then use a
traditional monolingual model [@aluru2020deep]. Some more advanced
models that use two-way translations to train multiple models in
parallel have also been successfully used
[@wan-2009-co; @pamungkas-patti-2019-cross].

**Multilingual methods**: The other method is to use multilingual
embeddings, so that we can train a model that can comprehend multiple
languages simultaneously, such as LASER embeddings
[@artetxe2018massively] that are able to create sentence embeddings for
93 languages, all residing in the same embedding space, and MUSE
embeddings [@conneau2019unsupervised], aligned versions of the FastText
word embeddings [@bojanowski2016enriching] in 30 languages. A
multilingual version of BERT was also created, [^7] and later other
models with similar architectures like XLM [@lample2019cross] and
XLM-RoBERTa [@conneau2019unsupervised].

The choice of the approach often depends on the language and the actual
datasets used [@aluru2020deep]. No method is consistently better than
the others. Both approaches will be evaluated in this project in chapter
[5](#cross_lingual){reference-type="ref" reference="cross_lingual"}.

Background
==========

Data
----

Gathering data for toxic comment classification is a challenging task,
as manual annotation is required and it is not trivial to gather toxic
data in the first place. Hateful comments are quite sparse online,
especially since some of them are fortunately rapidly removed, either by
the author or a moderator, leading to a large class imbalance with naive
data gathering. Still, there have been numerous attempts in the past few
years to collect such data. [^8] Most data originate from Twitter, and
this is the type of content that we use in this project.

There are, however, some common problems with the available data. The
majority of research on the subject has been done on the English
language, and resources in most other languages are sparse or
non-existent, complicating the creation of models. Besides, the
annotation scheme often differs from one dataset to another; the number
of classes can vary, sometimes a binary classification between toxic and
non-toxic comments is performed, or sometimes hateful content is also
further annotated into distinct categories. We conducted an exhaustive
search of available datasets and present our selection in the following
section. This selection is based on the size of the dataset (we excluded
small datasets when we already had multiple larger datasets in one
language), the annotations available (they had to be compatible with the
`hate`, `offensive` and `none` classes of our problem, see section
[1.2](#problem_statement){reference-type="ref"
reference="problem_statement"}), and a clear documentation on the source
and the acquisition methodology of the data.

### Available datasets

Here, we present the datasets that were chosen for this project. We have
datasets in English, French and German that are used in both the first
part (*Monolingual Toxicity Detection*, chapter
[4](#monolingual){reference-type="ref" reference="monolingual"}) and the
second part (*Cross-lingual Toxicity Detection*, chapter
[5](#cross_lingual){reference-type="ref" reference="cross_lingual"}) to
train some models. There are also additional datasets in other languages
(Spanish, Italian, Turkish, Greek and Indonesian) that are only used in
the second part to evaluate multilingual models, but not for training.

#### Dachs

The first dataset, published by @charitidis2019countering for DACHS (*A
Data-driven Approach to Countering Hate Speech*), contains annotated
tweets in five languages. We focus on the English, French and German
data, but also use the Spanish and Greek data in the second part of the
project. For each language two datasets are provided, one annotated for
hate speech, the other annotated for personal attack, and for each
dataset they provide tweet ids and a binary classification (*hate or
not* for the first dataset, or *attack or not* for the second dataset).
Each tweet was annotated by one person only, and the majority of tweets
appear in both the *hate* and *attack* datasets. We used the Twitter API
[^9] and the Twython library [^10] to retrieve the text associated to
these ids. However, some of those tweets have since been deleted
(approximately 30%), we were therefore unable to retrieve all the data
corresponding to the original datasets.

The datasets were then adapted to our problem statement to match the
previously defined classes (`hate`, `offensive` and `none`). For each
language, both datasets were joined and each tweet tagged as *hate
speech* in the original dataset is tagged as `hate` in our dataset. Each
tweet that is marked as *personal attack* but not *hate speech* in the
original dataset is classified as `offensive` in our dataset. Finally,
each tweet that is neither a personal attack nor a hate speech is
classified as `none` in our dataset.

#### Founta et al.

The second dataset is provided by @founta2018large and consists of
English tweets, each annotated by five people, but only the tweet ids
are available. Since it is already provided as a multi-class problem, no
further transformation was required. In this dataset, the possible
classes are *hate speech* (corresponding to our `hate` class), *abusive
language* (`offensive`), *neutral* (`none`), and *spam*. We removed all
spam messages (there are only a few of them) as we cannot assign them to
any of the three classes we are interested in.

#### Davidson et al.

The third dataset, provided by @davidson2017automated, is also a
multi-class dataset in English with labels *hate*, *offensive* and
*neither*, corresponding to our three classes. The tweets are annotated
by at least three people each and are directly provided.

#### Jigsaw

The fourth dataset was published by Jigsaw and Google on Kaggle for a
machine learning challenge. [^11] This data, in English, comes from
Wikipedia comments and contains among others the following annotations:
*severe toxicity*, *identity attack*, *threat* and *insult*. Each of
those is a number between $0$ and $1$, representing the average between
different annotators that had to label $0$ (no) or $1$ (yes) for each
category. Messages with value *identity attack* larger than $0.5$ or
value *threat* larger than $0.5$ are labelled as `hate`, messages with
value *insult* larger than $0.5$ or value *severe toxicity* larger than
$0.5$ are labelled as `offensive`, if they are not already in the `hate`
class, and the rest goes into the `none` class.

Moreover, because of the large quantity of data in the `none` class
compared to the other two classes, it was decided to under-sample data
from this class. We arbitrarily chose to keep 100'000 messages from the
`none` class, which was then slightly reduced when removing some
duplicate messages, giving us a proportion of `hate` messages similar to
the other datasets (around 5%), while having approximately 50% of toxic
messages.

In a more recent challenge proposed on Kaggle by the Jigsaw team, [^12]
an evaluation set with data in Spanish, Italian and Turkish is provided.
The annotation only consists of a binary label indicating the presence
or not of toxicity, but we will still evaluate our models from the
second part of this project on this data.

#### MLMA

The fifth dataset, provided by @ousidhoum2019multilingual, contains data
in English, French and Arabic. It consists of tweets that are annotated
as *offensive*, *disrespectful*, *hateful*, *fearful*, *abusive* or
*normal* (6 classes), with three annotators for each message. English
and Arabic data were ignored, as the English dataset was extremely small
and obtained very bad results compared to the other English datasets,
and we are not interested in Arabic data. Any tweet that was labelled as
*hateful* or *fearful* was considered as `hate`, any tweet labelled as
*abusive* or *offensive* was considered as `offensive`, and the
remaining tweets as `none`.

#### Germeval

For the sixth dataset, we regrouped data from the GermEval2018 [^13] and
GermEval2019 [^14] challenges. For both challenges, we mapped the
*abuse*, *insult* and *other* labels to `hate`, `offensive` and `none`,
respectively, and we concatenated both datasets.

#### Indonesian Hate Speech

The final dataset, provided by @ibrohim-budi-2019-multi, consists of
Indonesian tweets with labels for *hate speech* and *abusive language*,
that we map to `hate` and `offensive`. This dataset is only used in the
second part of this project to evaluate multilingual models.

#### Other datasets

Other datasets were also available but were excluded for multiple
reasons, often due to an incompatibly with our annotation scheme, for
instance with only a binary classification scheme or missing classes
[@waseem-hovy-2016-hateful; @wulczyn2016ex; @gibert2018hate]. Additional
datasets can be found on *hatespeechdata*. [^15]

#### Summary

We summarise all available training data in figure
[3.1](#fig:data){reference-type="ref" reference="fig:data"} and table
[\[tab:data\]](#tab:data){reference-type="ref" reference="tab:data"}.

![[\[fig:data\]]{#fig:data label="fig:data"}Summary of all the available
data in English, French and German.](data){#fig:data width="90%"}

English is the language with the most resources as expected, and the
toxic classes are in minority. While this represents the inherent
distribution of this type of data, it also means it will be harder to
learn a good classification for those lower resource classes, and the
metric will need to be selected accordingly.

In addition to the English, French and German data that will be used to
train our models, we have data that will exclusively be used in the
second part of this project to evaluate the performance of our
multilingual models on unseen languages. We show a summary of available
datasets for these languages in table
[\[tab:data_multilingual\]](#tab:data_multilingual){reference-type="ref"
reference="tab:data_multilingual"}.

We observe a similar class imbalance as with the training datasets, but
it is not a problem since we will not train any models on these
datasets, as long as the metrics properly report the performance of the
model on all classes.

### Data analysis {#data_analysis}

Before applying any model to the data, it is important to analyse more
attentively the content of each dataset. To be specific, we are looking
for recurrent keywords and themes in each class, which could let us know
if there are considerable differences between the datasets. This
analysis is performed on the English, French and German datasets.

For each dataset, a feature vector is generated for each sentence using
a simple bag of unigrams approach. A Logistic Regression Classifier from
the scikit-learn [^16] library [@scikit-learn] is used to train a
classification model, and then we extract the features with the highest
coefficients. We remove stopwords and only keep one member of each word
family (words with the same lemma or alternative spellings). We show in
table
[\[tab:important_words\]](#tab:important_words){reference-type="ref"
reference="tab:important_words"} for each dataset some of the most
important words for the `hate` and `offensive` classes.

We see that the themes in each dataset are quite different. Hateful
content in the Dachs EN datasets is about ways to hurt and kill other
people, while offensive content mostly seems to be about \"fakenews\"
and politics. For the Davidson and Founta datasets, `hate` mostly
consists of comments, whereas `offensive` is mostly offensive language
and insults. With the Jigsaw dataset, we have a mix of the previous
distributions, mostly racist comments in `hate` and some lighter insults
in `offensive`. In French, MLMA-FR tends towards racism while Dachs FR
is more about ways to hurt others. The two German datasets are globally
similar. To illustrate this difference in content, we show in figure
[3.4](#fig:wc_en){reference-type="ref" reference="fig:wc_en"} wordclouds
consisting of the most important words for the `hate` and `offensive`
classes, this time with all data from each language merged. We can see
that with the datasets merged, the most important words are quite
different. In all three languages, most of the important words for
`offensive` were not present when considering the datasets separately,
indicating that the model had to find more general words that were
present in all datasets.

Due to this disparity in the type of content covered by each dataset, we
will first study them separately, as there might be a significant domain
gap that might cause problems when combining them.

![[\[fig:wc_en\]]{#fig:wc_en label="fig:wc_en"}Wordclouds of important
words in the English, French and German datasets, according to the
logistic regression coefficients.](wordcloud_en "fig:"){#fig:wc_en
width="80%"} ![[\[fig:wc_en\]]{#fig:wc_en label="fig:wc_en"}Wordclouds
of important words in the English, French and German datasets, according
to the logistic regression
coefficients.](wordcloud_fr "fig:"){#fig:wc_en width="80%"}
![[\[fig:wc_en\]]{#fig:wc_en label="fig:wc_en"}Wordclouds of important
words in the English, French and German datasets, according to the
logistic regression coefficients.](wordcloud_de "fig:"){#fig:wc_en
width="80%"}

Evaluation Metrics {#metrics}
------------------

An important consideration is the choice of the evaluation metric for
the considered task. One major problem with the data available, both in
the datasets we use in this project and real-world applications, is the
large class imbalance. Hateful content is underrepresented, but it still
needs to be correctly accounted for in the metric. We preferred a known
metric, to allow better reproducibility and comparison with previous and
future work. For these reasons, we chose the macro-F1 score, the
unweighted average of the F1-score of each class, where the F1-score for
a class is the harmonic mean of the precision and recall on that class.

This metric was chosen for its property of giving as much importance to
the minority classes as to the majority classes. Other metrics such as
accuracy, weighted-average F1-score or micro-F1 accord more importance
to the majority classes, and any sudden drop of performance on a
minority class would have almost no influence on the final score. With
the macro-F1, a performance drop on the `hate` or `offensive` class will
have as much impact as a drop of performance on the `none` class. For
some models, we will also report the recall and precision on all
classes, as it gives more insight into the strengths and weaknesses of
each model.

Using simpler metrics like the recall on the toxic classes was also
considered, but those often had the problem of ignoring one aspect of
the problem. Additionally, some extreme cases with very low precision
could happen, and we were unable to automatically detect them with these
simpler metrics. Experiments were carried out with a custom metric
tailored for this problem, its usefulness is discussed in section
[6.3](#fhate){reference-type="ref" reference="fhate"}.

Finally, before using any metric, it is important to identify some of
its limits. It is especially interesting to know about results in
extreme cases to define clear baselines. The Dachs EN dataset is used
here to illustrate this aspect, but the conclusions are similar with the
other datasets. We consider some naive models: three of them always
predict the same class, `none`, `offensive` or `hate`, one makes a
uniformly random classification, and the last one generates random
predictions, but with probabilities proportional to the prior
distribution of the classes. We report the precision and recall for all
classes, with the macro-F1 score, in table
[\[tab:naive\]](#tab:naive){reference-type="ref" reference="tab:naive"}.
Every metric reported here and in this report is in percents.

Two main conclusions can be drawn from those results. First, it is
possible to achieve a macro-F1 score of 33% with a naive model that does
not actually do anything useful. It is important to keep this in mind
when experimenting with various models, as a score around this value
that might initially indicate our model is learning something relevant
actually does not mean anything. Secondly, we can see here why a score
based uniquely on one metric (for instance the recall on `hate`) can be
problematic, as it can reach maximal values with dummy models. The
macro-F1 score does not suffer from this problem.

Tokenisers
----------

### Byte-pair encoding {#bpe}

Byte-pair encoding [@sennrich-etal-2016-neural] is a subword
tokenisation scheme that tries to optimise the amount of data required
to represent text. It works as follows:

1.  prepare a large corpus, and define a desired subword vocabulary
    size;

2.  get the word count frequency and the initial token count (i.e. the
    number of occurrences for each character);

3.  merge the two most common tokens, add the results to the list of
    tokens, remove the used tokens, and recalculate the frequency count
    for all tokens;

4.  repeat until the vocabulary size is reached.

### WordPiece tokeniser {#wordpiece}

WordPiece tokenisation [@wu2016googles] is very similar to Byte-pair
tokenisation, in the sense that it uses the same algorithm, except that
instead of merging tokens based on their frequency, we merge them based
on the increase in likelihood if merged. This difference is the
probability of the new merged pair occurring, minus the probability of
both tokens occurring individually.

Monolingual Toxicity Detection {#monolingual}
==============================

In this chapter we present our work on monolingual toxicity detection.
We use the datasets individually and create some classifiers for each of
them separately. Multiple architectures are presented, and potential
improvements are tested. The combination of multiple datasets from the
*same* language is also briefly studied.

Models
------

There exist different types of classifiers, with various levels of
complexity. For this problem, we experimented with three categories of
models. We first have a simple statistical model, here a Logistic
Regression classifier, easy to use and with a fast training time. Then
there are classical neural networks, still with relatively simple
architectures, but with a more consequent training time. Finally, there
are so-called *transformer* models, large pretrained language models
that are currently the state-of-the-art in most natural language
processing tasks [^17] but require significant training time. We
primarily focus on this type of model in this project.

### Logistic Regression {#logisitc}

As a first baseline, a Logistic Regression classifier from the
scikit-learn [^18] library [@scikit-learn] was chosen. For the features,
we used count vectorisation on unigrams and bigrams of words, only
keeping the 20'000 most frequent features.

The Logistic Regression classifier algorithm used here uses the \"one
versus rest\" approach, where each class is treated as a separate binary
classification problem. This approach was chosen as it is the default
one in the scikit-learn library.

### Convolutional Neural Network {#cnn}

We implemented a simple convolutional neural network (CNN) using the
PyTorch [^19] library [@pytorch]. Each word in the input is first mapped
to a vocabulary id. Using the Gensim [^20] library [@rehurek_lrec], we
load pretrained GloVe [@pennington2014glove] or FastText
[@bojanowski2016enriching] word embeddings, which are then loaded into
the embedding layer provided by PyTorch. This embedding layer will map
each vocabulary id to an embedding vector and can tune those embeddings
during training. This can be useful since the type of data used to
create the embeddings might be quite different from the data used for
this project, both semantically and syntactically, hence the existing
embeddings might need to be adjusted for optimal performance.

The embeddings outputted by the embedding layer are then sent into three
2D convolutional layers, each with one hundred filters of size 3,4 or 5
in the sentence length dimension and of embedding size in the other
dimension. A rectified linear unit (ReLU) is applied, followed by a
max-pool operation along the sentence dimension, and the resulting
outputs from all three sizes are concatenated. We end with a dropout
layer with probability 0.1, and a final linear layer that outputs three
values, one for each class. The class with the highest value in the
output is the predicted class. A softmax layer is applied by the
implementation of the loss function to generate a probability for each
class, therefore we do not include it ourselves. This architecture is
illustrated in figure [4.1](#fig:cnn){reference-type="ref"
reference="fig:cnn"}.

![[\[fig:cnn\]]{#fig:cnn label="fig:cnn"}CNN architecture, for a
sentence of length $N$, a word embedding of size $E$ and 100 channels
for each filter. The yellow, blue and green arrays correspond to the
outputs of the three convolutions with kernel size 3,4 and 5
respectively.](cnn_simple){#fig:cnn width="70%"}

### Bidirectional Long short-term memory network {#bilstm}

We also implemented a model that uses two stacked bidirectional long
short-term memory (biLSTM) layers. In a biLSTM, we have two classical
LSTM layers, one going from the start of the sentence to the end, and
the second going from the end to the start. The input is given to both
networks at the same time, and then the hidden outputs of the two layers
are concatenated. We subsequently reduce those to one vector by
computing the mean of all the outputs, and to another vector by taking
the maximum value for each dimension. We concatenate these two vectors,
apply some dropout with probability 0.1, send this into a linear layer
with 64 outputs, a ReLU layer and finally a linear layer with three
outputs.

The embeddings are created by loading GloVe [@pennington2014glove] or
FastText [@bojanowski2016enriching] word embeddings with Gensim
[@rehurek_lrec], and then using the PyTorch [@pytorch] embedding layer.
The softmax layer is excluded here as it is included in the
implementation of the loss function. This architecture is illustrated in
figure [4.2](#fig:lstm){reference-type="ref" reference="fig:lstm"}.

![[\[fig:lstm\]]{#fig:lstm label="fig:lstm"}biLSTM architecture, for a
sentence of length $N$, a word embedding size of $E$ and a hidden size
for the LSTM units of 100. ](lstm_simple){#fig:lstm width="45%"}

### Transformer Models {#transformer_models}

Regarding *transformer models*, we use huggingsface's transformers [^21]
library [@Wolf2019HuggingFacesTS], that let us use pretrained
transformer models such as BERT [@devlin2018bert], RoBERTa
[@liu2019roberta] and XLNet [@yang2019xlnet]. It provides a common
interface to easily use and modify all of these models.

#### BERT

BERT (*Bidirectional Encoder Representations from Transformers*) was
introduced by @devlin2018bert in 2018. It is able to create deep
bidirectional contextual embeddings using raw unsupervised text data for
training, that can then be used for numerous language processing and
understanding tasks. There are two main phases when using a BERT model,
or other similar transformer models: *pretraining* on general data and
*fine-tuning* for a downstream task.

**Pretraining**: The pretraining step is typically performed by the
authors of the model, as it requires a huge amount of computation power
and a lot of data. BERT was trained on two unsupervised tasks, Masked
Language Model (MLM) and Next Sentence Prediction (NSP). For the MLM
task, a percentage of the input words are replaced by a `[MASK]` token,
and the model is trained to predict the missing words. More precisely,
15% of the input tokens are selected, and among them 80% are replaced by
the `[MASK]` token, 10% with a random token and the remaining 10% use
the original word. For the NSP task, BERT will learn to predict the
probability of two sentences being consecutive. During training, it will
receive in 50% of cases two consecutive sentences from the training data
and in the other 50% two nonconsecutive sentences at random, and it will
learn to detect consecutive sentences.

**Fine-tuning**: In the second part, fine-tuning on the downstream task,
the pretrained model is loaded and slightly adapted, usually by adding
one or more new layers(s) extracting information from the transformer,
constructing a network that will then be trained on the target task and
data. During training, the weights of the added layer(s) as well as the
weights from the original BERT network will be updated, learning a
classifier that utilises those contextual embeddings, while also
learning a better representation of the sentence by updating the
inherent language model to be more adapted to the target domain.

#### BERT architecture

The input sentences are first converted to a specific input format. With
BERT, the WordPiece tokeniser [@wu2016googles] is used to create the
input tokens (see section [3.3.2](#wordpiece){reference-type="ref"
reference="wordpiece"}), then a `[CLS]` token is added at the start of
the sentence, and a `[SEP]` token is inserted between the first and the
second sentence. For our problem, since we do not work on a task
requiring a distinction between two sentences (this distinction is
useful with question answering for instance), all sentences in the input
are concatenated together, and the `[SEP]` token is added at the very
end of the message. We then pad the input with `[PAD]` tokens, up to a
length of 50 tokens. These tokens are then mapped to embeddings using an
existing vocabulary-embedding mapping, creating *token embeddings*. We
had to limit the maximum sentence size due to memory constraints, 50 was
chosen as a good trade-off on our machine, [^22] where we are able to
run the desired models and the majority of sentences do not need to be
shortened. It also was the value chosen by @charitidis2019countering as
their maximum sentence size, and they noticed that larger sentences did
not really improve performance. Assuming a maximum length of 25 tokens
for the example, and using simple whitespace and punctuation-based
tokenisation for convenience, a typical input using BERT would in our
case look likes this:

> `[CLS] this is a very [MASK] car that you have . I [MASK] that mine was as nice . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] `

Additionally, BERT automatically creates *segment embeddings* and
*position embeddings*. Segment embeddings are useful to differentiate
two sentences in the case where we decide to separate them with a
`[SEP]` token. A segment embedding is used for all tokens in the first
sentence, and a different segment embedding is used for all tokens in
the second sentence. Position embeddings are used to differentiate two
instances of the same word but at different locations in the text. Two
words, in two different instances, that are at the same position will
have the same position embedding. These are required as the BERT
architecture cannot understand positions by itself. An element-wise
addition is then performed between the token embeddings, the segment
embeddings and the position embeddings, giving us the *input
embeddings*, as shown in figure
[4.3](#fig:bert_input){reference-type="ref" reference="fig:bert_input"}.

![[\[fig:bert_input\]]{#fig:bert_input label="fig:bert_input"}Bert input
embeddings, illustration from
@devlin2018bert](bert_input){#fig:bert_input width="80%"}

The rest of the architecture is a series of Transformer encoder blocks
[@vaswani2017attention], illustrated in figure
[4.4](#fig:transformer){reference-type="ref"
reference="fig:transformer"}. The *input embeddings* are sent into the
first Transformer encoder block, whose outputs are then forwarded to the
next block. BERT uses 12 or 24 of these, depending on the exact model
(BERT-base or BERT-large).

![[\[fig:transformer\]]{#fig:transformer label="fig:transformer"}A
Transformer encoder block, illustration from
@vaswani2017attention](transformer){#fig:transformer width="20%"}

The first part of this encoder block is the *attention layer* that gives
more importance to some parts of the input. This concept can be
formulated using *queries*, *keys* and *values*. For some query, we want
to know which keys are relevant, so that we can mask some irrelevant
values for each query. In our case, each word is a query, and we want to
know how relevant the other words (keys) are to our main word/query. For
this, we use weights on the values of words to give them some
importance. An example of this concept is shown in figure
[4.5](#fig:attention_ex){reference-type="ref"
reference="fig:attention_ex"}.

![[\[fig:attention_ex\]]{#fig:attention_ex label="fig:attention_ex"}An
illustration of the attention mechanism. On the left the different keys,
on the right the different queries, and in blue the weights on the
values. The attention of the first instance of the word *\"park\"* is
focused on the word *\"park\"*, but also on *\"car\"*, giving some
information about the meaning of the word *\"park\"*, and \"*will*\",
giving the information that it is a verb. In the second instance of
\"*park*\", with a different meaning, the attention is focused on other
terms: \"*walk*\", \"*in*\" and \"*park*\". With an attention mechanism
like this, the embeddings of the two instances of \"*park*\" should be
significantly different, therefore creating *contextual* word
embeddings.](attention_ex){#fig:attention_ex width="30%"}

All of the attentions can be computed in parallel if we represent the
input as a matrix of shape (embedding size, sentence length). This
matrix is used for the queries, the keys and the values ($Q$, $K$ and
$V$). We can then compute the attentions using the scaled dot-product
attention:

$$\textrm{Attention}(Q, K, V) = softmax(\frac{QK^\mathsf{T}}{\sqrt{d_k}})V$$

$QK^\mathsf{T}$ computes the similarity between the queries and the
keys, which is scaled by the root of the embedding size to avoid
problems with a large embedding size. This similarity is then applied to
the values to give some of them more or less importance (see figure
[4.6](#fig:attention){reference-type="ref" reference="fig:attention"},
illustration on the left).

In the original Transformer encoder block, eight attentions are used for
each query instead of only one, allowing the model to concurrently
explot information from different representation spaces. A multi-head
attention is used, where $Q$, $K$ and $V$ are first transformed into
eight different matrices each, using linear transformations, before
going through the scaled dot-product attention. The resulting outputs
are concatenated and sent to another linear transformation. Formally,

$$\textrm{MultiHead}(Q, K, V) = \textrm{Concat}(\textrm{head}_1, \textrm{head}_2, ... , \textrm{head}_8)W^O$$
$$\textrm{with head}_i = \textrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

The projections $W_i^Q$, $W_i^K$, $W_i^V$ and $W^O$ are parameter
matrices, and they project the $Q$, $K$ and $V$ matrices into an eight
times lower dimension space so that after concatenation the output has
the same size as the input (see figure
[4.6](#fig:attention){reference-type="ref" reference="fig:attention"},
illustration on the right).

![[\[fig:attention\]]{#fig:attention label="fig:attention"}The attention
module, image from @vaswani2017attention](attention){#fig:attention
width="80%"}

The second module of this transformer block is the feed-forward layer,
where two fully connected layers are applied to each position separately
and identically, with a ReLU in between. The size in the middle of the
two layers is four times the size of the input and output. Those can be
seen as convolutions with a kernel size of one.

For each module (attention and feed-forward), a normalisation layer is
added afterwards, taking the output from the module and a skip
connection coming from before the module, concatenating and normalising
them.

BERT [@devlin2018bert] was trained on the BooksCorpus [@Zhu_2015] and
English Wikipedia, [^23] for a total raw text size of approximately
16GB. Two main versions of BERT are available:

-   BERT-base: 12 layers, hidden size of 768, 12 attention heads, 110M
    parameters

-   BERT-large: 24 layers, hidden size of 1024, 16 attention heads, 340M
    parameters

The large version performs better than the base version, but we use the
base version due to memory constraints.

#### BERT Variants

Since the introduction of BERT [@devlin2018bert], alternative models
were presented with various optimisations and changes, but always with
the same fundamental idea of creating pretrained models that generate
context-aware embeddings, and that can then be fine-tuned on a new task.

For instance, RoBERTa [@liu2019roberta] removes the Next Sentence
Prediction task, but introduces dynamic masking, meaning that the subset
of masked tokens changes at every epoch. Significantly more training
data is used, 160GB compared to 16GB for the original BERT model, and
Byte Pair Encoding [@sennrich-etal-2016-neural] (see section
[3.3.1](#bpe){reference-type="ref" reference="bpe"}) is used instead of
WordPiece [@wu2016googles] to tokenise the input.

There is also XLNet [@yang2019xlnet], that replaces the original MLM
objective by a permutation language modelling objective, and uses
Transformer-XL [@dai2019transformerxl] as a building block, a new
transformer architecture that solves the issue of limited sentence
length from the original transformer. It was trained on 139GB of data.

There are additionally models for other languages; for French, FlauBERT
[@le2019flaubert], that uses only the MLM objective and is trained on 71
GB of data, and CamemBERT [@martin2019camembert], based on RoBERTa that
uses 138GB of data. For German, we have German BERT [^24] which is
simply BERT trained with 12 GB of German data. We also have models that
understand multiple languages, such as multilingual-BERT, [^25] a BERT
model trained on multilingual data (the exact size is not documented,
but the full Wikipedia dumps for the top 100 languages was used), and
XLM-R [@conneau2019unsupervised], a RoBERTa model trained on 2.5TB of
multilingual data. A list of such models can be found on huggingface's
transformers website. [^26]

Improvements
------------

### Classification Heads {#classification_heads}

Starting from a base transformer model, we derive multiple classifiers
following some ideas from @mozafari2019bertbased. On top of the base
transformer model, we connect some additional layers that we call
*classification head*, and that can extract relevant information from
the transformer. For all of them we suppose a sentence size of $N+1$
(the `[CLS]` token plus $N$ additional input tokens, including the
\[SEP\] token and eventual padding tokens), a hidden size for the
transformer of $H$ (which is also the size of the input embeddings), and
$L$ Transformer encoder layers (the actual value of $L$ and $H$ depends
on the specific model used, there are often *small*, *base* and *large*
variants with varying sizes). Input data will be of shape $(N+1, H)$. We
ignore the softmax layer at the end of all the classification heads, as
it is included in the PyTorch [@pytorch] implementation of the loss
function.

#### Base Transformer Classifier

In the first model, illustrated in figure
[4.7](#fig:cls_head){reference-type="ref" reference="fig:cls_head"}, the
output of the `[CLS]` token (of length $H$, situated at the beginning of
the sentence) is sent into a dropout unit with probability 0.1, and a
linear layer with three outputs (corresponding to our three classes;
`none`, `offensive` and `hate`). By construction, the output of the
`[CLS]` token should be a summarised representation of the sentence and
should, therefore, contain most of the necessary information. The output
with the highest value indicates the class predicted. This is the usual
and default way to use a transformer model.

![[\[fig:cls_head\]]{#fig:cls_head label="fig:cls_head"}Transformer with
the CLS classification head. Each transformer encoder block follows the
architecture described in figure
[4.4](#fig:transformer){reference-type="ref"
reference="fig:transformer"}. There are $L$ transformer encoder blocks,
and we represent the first, the second and the last of
them.](cls){#fig:cls_head width="40%"}

#### CLS-LSTM Classifier

In the second model, we extract the values corresponding to the `[CLS]`
token at each hidden layer of the transformer. This returns $L$ vectors
of length $H$ that are sent into a LSTM layer with a hidden size of $H$
going from the front to the back of the network. We extract the final
hidden state of the LSTM, a vector of size $H$, and send it into a
dropout of 0.1 and a linear layer with three outputs, corresponding to
our three classes (illustrated in figure
[4.8](#fig:clslstm_head){reference-type="ref"
reference="fig:clslstm_head"}).

The idea is to improve on the previous classification head by including
information from the previous layers, while still only relying on the
`[CLS]` token. Hopefully, using earlier iterations of the outputs of the
`[CLS]` can provide more information. We can see the change of the
output of the `[CLS]` token over the layers as a change over time, and a
LSTM might be adapted. For instance, the earlier values might represent
the sentence generally and the later values might represent it
relatively to a toxicity aspect.

![[\[fig:clslstm_head\]]{#fig:clslstm_head
label="fig:clslstm_head"}Transformer with the CLS-LSTM classification
head. There are $L$ transformer blocks with a hidden size of $H$, and
the LSTM layer has $L$ units with a hidden size of
$H$.](cls_lstm){#fig:clslstm_head width="50%"}

#### CNN Classifier {#cnn_head}

In the third classification head, we use convolutional layers to extract
information from the transformer. In the first variant, we apply $L$
convolutional layers, one for each output of the $L$ transformer encoder
blocks. For each convolutional layer, a kernel of shape $(5, H)$ is
used, and we have 200 output channels followed by a ReLU and a max-pool
unit that only keeps one value per channel. This produces data of shape
$(200,L)$, one vector for each layer of the transformer. We then
concatenate the vectors from all layers into a vector of length $200L$,
and send it into a dropout of 0.1 followed by a linear layer with three
outputs. This variant is illustrated in figure
[4.9](#fig:cnn_head){reference-type="ref" reference="fig:cnn_head"}.

In the second variant, we apply for each layer three different
convolutional layers with kernel shapes $(3, H)$, $(4, H)$ and $(5, H)$,
with 200 output channels each. As previously, a ReLU and max-pool layer
are applied, and then we concatenate the data from all the layers, and
for the three sizes. Similarly, we go through a dropout of 0.1 and a
linear layer. We implement a similar architecture to the CNN presented
in section [4.1.2](#cnn){reference-type="ref" reference="cnn"}, where
that model is applied to each of the transformer's layers, and we then
concatenate all of the outputs together.

![[\[fig:cnn_head\]]{#fig:cnn_head label="fig:cnn_head"}Transformer with
the CNN classification head, one filter size variant, with 200 outputs
channels. There are $L$ transformer blocks, with a hidden size of
$H$.](cnn){#fig:cnn_head width="60%"}

#### CNN to LSTM Classifier

For the final architecture, we extend the CNN classification head by
adding a biLSTM layer. As previously, we apply a convolutional layer at
each level of the transformer, either the one filter or three filter
variant, and then apply the ReLU and a max-pool layer. We are left with
one vector for each layer, as before. Instead of concatenating them, we
send them into a bidirectional LSTM network. We then take the max-pool
of all the outputs of the biLSTM, and send it to a dropout of 0.1 and a
final linear layer with three outputs. Once again we have two versions,
one with a single kernel of size 3 and one with three of them (sizes 3,4
and 5). The variant with a single kernel size is illustrated in figure
[4.10](#fig:cnnlstm_head){reference-type="ref"
reference="fig:cnnlstm_head"}.

The intuition is the same as with the CLS-LSTM model, there is an
evolution of the hidden states when moving through the layers, and a
LSTM might be a more appropriate approach than a simple concatenation to
regroup data coming from different layers.

![[\[fig:cnnlstm_head\]]{#fig:cnnlstm_head
label="fig:cnnlstm_head"}Transformer with the CNN to LSTM classification
head. There are $L$ Transformer blocks with hidden size $H$, the biLSTM
has $L$ units in both directions, and a hidden size of
200.](cnn_lstm){#fig:cnnlstm_head width="60%"}

#### Model Complexity

Using advanced classification heads increases the complexity of the
models and therefore the training time. Since the classification heads
can be used with multiple models (e.g. BERT [@devlin2018bert], RoBERTa
[@liu2019roberta], FlauBERT [@le2019flaubert], etc.) which all have
different sizes, the exact number of additional parameters depends on
the actual model choice. We show in table
[\[tab:parameters\]](#tab:parameters){reference-type="ref"
reference="tab:parameters"} the relative increase in complexity
depending on the classification head, to give an idea of the additional
memory and training time these different architectures require.

It is important to remember that although the transformer is pretrained,
we still have to update its parameters during training (the
*fine-tuning* part). It will be the component with the largest impact on
the training time. It has been shown that using a frozen transformer is
detrimental to the score, although partially frozen models can reach
relatively close scores compared to the normal unfrozen transformer,
with lower training time [@lee2019elsa]. We did not experiment with such
optimisations and always update the full transformer.

### Dataset merging {#merging}

When training a machine learning model, a common method to obtain
significant improvement is to use more training data. In our case, we
cannot increase the size of the existing datasets, but we can combine
them to create a larger training set. More specifically, for a dataset
of a given language, we will train a model on more data from the same
language, and then evaluate it only on the original dataset. We expect
this to be useful especially for datasets with a very limited amount of
data (in particular in the minority classes), the assumption being that
adding external data to a small dataset with weak results cannot be
worse than using the original data, even if the data distributions do
not exactly match. It is however important to be careful not to merge
too many datasets with diverse content or themes, as the model trained
might lose its compatibility with the original dataset.

### Multitask learning {#multitask}

With multitask learning, we also want to utilise information from an
additional dataset to improve results, but in a more subtle manner. We
split the model into two sections; the first part will be common to both
datasets, and the second part will be specific to each dataset. During
training, we alternate between the two datasets for each minibatch,
sending the data through the common part of the model, and then through
the specific part corresponding to this data. We can split the model at
various levels, either having only the final linear layer specific or
the complete classification head. This idea is inspired by the work of
@rizoiu2019transfer, but to our knowledge it has not been applied to
transformer models yet.

Figure [4.11](#fig:multitask){reference-type="ref"
reference="fig:multitask"} provides an illustration of this architecture
applied on a transformer model with the CLS-LSTM classification head and
with two datasets, *yellow* and *green*. We send a minibatch of *yellow*
data through the transformer, and then into the *yellow* classification
head, and we update all the weights on the way. The same happens with
the *green* data, going through the common part of the model and then
through the *green* classification head. This process is repeated for
all training data. During the evaluation phase on the *yellow* data, the
*green* part of the network is ignored. We can imagine an experiment
where we want to use data from the Jigsaw dataset to improve results on
the Davidson dataset. In this case, we would alternate between both
datasets during training, but only use the classification head trained
on Davidson data for the actual predictions. The transformer is better
tuned for toxic content as it was also trained on the Jigsaw data, but
the classification head stays specific to the Davidson data.

![[\[fig:multitask\]]{#fig:multitask label="fig:multitask"}Example of a
multitask architecture, here with the CLS-LSTM classification head, with
data originating from two datasets, *yellow* and *green*. The
transformer is common (in blue), and the classification heads are
specific to the datasets (*yellow* or
*green*).](multitask){#fig:multitask width="60%"}

The concept is similar to dataset merging, in the sense that we train
the model with more data, except that we make sure that at least some
part of the classifier is specific to each dataset. We expect that
multitask learning will work better than merging datasets in cases where
the data is considerably different between the two datasets, with
different class distributions. The common part will be able to learn a
more general understanding of toxic language, while the specific
classifier will be able to differentiate the various types of toxic
content, depending on the specific definitions for each dataset
involved.

In our implementation of multitask learning, the larger dataset of the
two is truncated so that both datasets have the same size, but since
shuffling is applied at every epoch most of the data in the larger
dataset should still be used. If we had retained the original sizes, we
would have been unable to alternate between the two datasets for the
full epoch, and we think this could have been a problem as the common
part would have started to overfit on one of the datasets, relatively to
the other one. At the start of the following epoch, the classification
head of the other dataset might not have supported the substantial
change in outputs of the common part. For a similar reason, we
initialise both classification heads with the same weights for a better
connection to the common part.

### Further Pretraining {#further_pretraining}

Instead of using the pretrained model as-is, we can do some further
pretraining on it. This idea was explored in depth by @sun2019finetune,
and we experiment with *within-task* pretraining. More specifically,
before fine-tuning the model, we train the transformer model without the
classification head for a few epochs on the original masked language
model task, using our training data. With this method, the transformer
model should be more familiar with the type of data that will be used to
fine-tune the model, and it should help with the performance. This step
is an intermediary process to facilitate the transition between the
original pretrained model, and the fine-tuning, as illustrated in figure
[4.12](#fig:training_steps){reference-type="ref"
reference="fig:training_steps"}. However, there is the risk that if this
process is performed for too long, the transformer will start to forget
previous knowledge (from the original pretraining), and start to overfit
on the new data.

![[\[fig:training_steps\]]{#fig:training_steps
label="fig:training_steps"}The three main steps of using a transformer
model: first, the model is pretrained by its authors using general data
and the MLM task, secondly, we can optionally continue training on the
MLM task using target data, and finally, we fine-tune the model on
target data using a classification head and a loss
function.](images/training_steps.png){#fig:training_steps width="65%"}

### Data augmentation {#data_augmentation}

One operation that can be performed to counter unbalanced classes is to
augment the data in the minority classes. A naive way to achieve this is
simple random oversampling, but initial experiments showed that it never
works, and only makes the model overfit on this multiplicated data. A
common alternative is SMOTE (Synthetic Minority Over-sampling Technique)
[@chawla2011smote] that creates new samples in minority classes by
generating points in the embeddings space that are close to other
samples in the same class, but it actually does not work with the
transformer models as we have multiple embeddings for each input.

Instead, we try editing the copies slightly at a higher level. For
instance, one could replace some words with some of their synonyms,
which can hopefully bring more examples to the model while keeping
correct and accurate sentences. We can also add some noise by inserting,
deleting or swapping some words, but this is riskier as we might alter
the meaning of a sentence and move it into another class. In particular,
removing the word *\"not\"* might drastically alter the meaning of a
sentence if the model was relying on it to make the correct
classification.

We settled for synonym replacement, random insertion of synonym words
and random swapping. We did not perform random deletions, as we would
have taken the risk of removing offensive words that are important for
the classification decision. We use the EDA [^27] library [@wei2019eda],
and we adapted it to also work with French data. We use the Open
Multilingual Wordnet [^28] [@Bond:Paik:2012] to get the synonyms, in
particular we use WOLF (Wordnet Libre du Français) [@Sagot:Fiser:2008]
for French and Priceton WordNet [@_Fellbaum:1998] for English. As far as
we know, no such data were available in German, so we will not perform
this operation on the German datasets. We present a few examples of
sentences generated by this process in table
[\[tab:eda\]](#tab:eda){reference-type="ref" reference="tab:eda"}.

The primary drawback of this method is the longer training time; since
we augment data from the minority classes, the total amount of data
after augmentation will be significantly higher. The synonyms generated
are also not always accurate as the Wordnet models do not use the
context of a word to find its synonyms, which can cause problems,
especially with homonyms.

### Data cleaning and normalisation {#data_cleaning}

Some simple cleaning processes can be applied to the data. Fundamental
operations are removing every URL, mention and hashtag from the messages
(we provide more details on the simple cleaning operations used in
section [4.3.2](#data_preprocessing){reference-type="ref"
reference="data_preprocessing"}), but it is possible to perform a more
methodical cleaning. Some of the content has spelling errors, but also
abbreviations that might be unknown to the transformer models. To
counter this, we use MoNoise [^29] [@goot2017monoise], a text
normalisation tool that was trained to clean and normalise tweets, and
uses notably the Aspell spell checker, [^30] word embeddings and random
forests to determine the best candidates to replace words. For instance,
given the sentence

> *u r rly stpid cuz u dd dat 2 all the pple jst 4 fun*

MoNoise is able recover

> *you are really stupid because you did that to all the people just for
> fun*

The model is available in a few languages, but unfortunately neither in
French nor German, therefore we only experiment with this method on the
English datasets.

It does seem very good in general and is capable of correcting both
spelling mistakes (*stpid* $\rightarrow$ *stupid*) and some common
abbreviations (*u* $\rightarrow$ *you*, *rly* $\rightarrow$ *really*,
*2* $\rightarrow$ *to*, *4* $\rightarrow$ *for*), but we were still able
to find some particular cases where it does not work correctly. For
instance, with a simple sentence like *\"I have 2 cats\"*, the *\"2\"*
will be corrected to *\"to\"*. It does not always seem able to look at
the context to understand the intended use of the number *\"2\"*.
Strangely, with *\"I have 4 cats\"* we do not encounter the same
problem, and the *\"4\"* is kept as-is. It would be complex to correct
this behaviour, as we have both cases where we want to replace *\"2\"*
by *\"to\"*, and cases where we want to keep it unchanged.

### Ensemble learning {#ensemble}

Ensemble learning is a method that uses multiple models trained on the
same task to obtain better performance than each model individually. The
prediction of the ensemble is made by regrouping each model's
prediction, for instance with majority voting (illustrated in figure
[4.13](#fig:ensemble){reference-type="ref" reference="fig:ensemble"}).
It has been shown that an ensemble between multiple instances of the
same model but with random weight initialisation can improve results
[@zimmerman-etal-2018-improving]. We can train the same model multiple
times with the same data split, and due to randomness in weight
initialisation and to the shuffling of data during training the
resulting models will all be slightly different. Test data is given to
all models and majority voting is applied to make predictions.

![[\[fig:ensemble\]]{#fig:ensemble label="fig:ensemble"}Ensemble
learning with three versions of the same model and majority
voting.](ensemble){#fig:ensemble width="50%"}

### Multilevel classification {#multilevel}

Instead of considering our problem as a three-class problem, we can also
consider it as two binary subproblems. We first train a model to
differentiate `toxic` and `non-toxic` messages. Meanwhile, we train
another model to differentiate `offensive` from `hate` content. We can
then evaluate our model as follows: for a given message, give it to the
`toxic` vs `non-toxic` classifier. If it is classified as `non-toxic`,
return the class `none`. Otherwise, send it to the `hate` vs `offensive`
classifier and return the predicted class (see figure
[4.14](#fig:multilevel){reference-type="ref"
reference="fig:multilevel"}).

![[\[fig:multilevel\]]{#fig:multilevel
label="fig:multilevel"}Multi-level classification compared to
Single-level classification.](multilevel){#fig:multilevel width="80%"}

For this approach we use a simplified methodology; we take the
architecture from the multiclass model and apply it to both
classification levels. Ideally, we would need to find the optimal
architecture for each level, but this would require too much time. The
advantage of this architecture is that it helps with class imbalance, as
`offensive` and `hate` will be regrouped into one `toxic` class with
more weight compared to `none`. It should also be easier to
differentiate the subtle differences between some offensive and hateful
content by having a model fully dedicated to this task. However, this
model has the disadvantage of containing two sources of error, in the
first and second classifier, so it may be more likely to make an error.
It also requires more training time, especially for large `hate` and
`offensive` classes.

Experiments
-----------

### Loss function {#loss}

As the loss function, we use the *Cross-Entropy Loss* (CEL). It is
defined, for a multiclass problem with $M$ classes and a true class $o$,
as

$$\mathcal{L} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$

where $y_{o,c}=1$ if and only if the class label $c$ is equal to the
true class $o$, and $p_{o,c}$ is the prediction probability for the
class $c$. Note that the CEL only takes the quality of the prediction on
the true class into account, implying that a case with the correct
prediction can have the same loss as a case with a wrong prediction. For
instance, if we assume that class $0$ in a 3-class problem is the true
class, a model with predictions `[0.4, 0.5, 0.1]` will have the same
cross-entropy loss as a model with outputs `[0.4, 0.3, 0.3]` (loss of
$log(0.4)$ in both cases), despite having a different predicted class.

For some models, we alternatively use the *Weighted Cross-Entropy Loss*
(WCEL) to assign more weight to samples originating from some classes.
Since our target metric is the macro-F1 that gives the same importance
to all classes, using a loss function that can increase the importance
of minority classes can be beneficial. The WCEL is defined as

$$\mathcal{L}_w = -\sum_{c=1}^{M} w_c y_{o,c} \log(p_{o,c})$$

where $w_c$ is usually inversely proportional to the number of training
samples in class $c$. In practice, we use $w_c = \frac{1}{n_c}$, with
$n_c$ the number of samples in class $c$. In our experiments, we found
that the WCEL is beneficial with simple models like the CNN and LSTM,
giving better macro-F1 scores than with the CEL, but detrimental to the
larger transformer models, giving lower macro-F1 scores compared to when
using the CEL. Additional effects of this loss function are presented in
section [6.3](#fhate){reference-type="ref" reference="fhate"}. We
briefly experimented with an \"*F1 loss function*\" that was computed as
$1-\textrm{F1-score}$ but it was only possible to implement this for a
binary problem, so we went with the CEL instead, the loss function
usually selected for multiclass problems.

The PyTorch [@pytorch] implementations of the CEL and WCEL include a
softmax layer, therefore it was ignored in the architecture of all our
models.

### Data Preprocessing {#data_preprocessing}

For all experiments, an aggressive cleaning process was chosen. We put
all content in lowercase, remove any emojis and most of the punctuation
and symbols (we remove - \_ / : ; \" ' , \* ( ) \| \\). We also replace
a few abbreviations (\"1st\", \"2nd\", \"3rd\" become \"first\",
\"second\" and \"third\", \"\$\", \"£\", \"%\" become \"dollar\",
\"pound\", \"percent\") and a few expressions (for instance \"*fyi*\"
becomes \"*for your information*\", \"*gtfo*\" becomes \"*get the f\*ck
out*\", etc.). All the hashtags, mentions and numbers are removed. We
get rid of repeating words to mitigate the importance of some spammed
expressions and, inside words, we only keep two occurrences of a letter
appearing three or more times in a row. This should help with the
various exaggerations (e.g \"*it is sooooo cold*\" should be considered
the same as \"*it is soooooooo cold*\"). A few examples of data cleaning
can be seen in table
[\[tab:cleaning\]](#tab:cleaning){reference-type="ref"
reference="tab:cleaning"}. We experimented with other variations of the
cleaning process, where we kept and separated the hashtags into words,
or replaced the mentions by special \"$<$user$>$\" tokens, but the
difference was negligible and since it was not possible to perform a
full grid search on all the variations, we adopted the original
aggressive cleaning process. Moreover, removing all traces of
Twitter-specific elements (hashtags and mentions) should help the models
generalise better on generic text.

Examples in table [\[tab:cleaning\]](#tab:cleaning){reference-type="ref"
reference="tab:cleaning"} show that while the mentions might be useful
in some cases (mentions to *\@realDonaldTrump* or *\@FoxNews* might for
instance talk about controversial topics, with polarising opinions and
heated debates), there are also all the other cases where the presence
of such mentions does not seem to bring any meaningful information
except some noise (e.g. *\"\@okamlord\"*, *\"via \@lemondefr\"*). We
wanted a classifier based on pure textual content only and avoid bias
due to the presence of some mentions. The same is true with hashtags; we
do not want our models to be too reliant on them.

For the Logistic Regression classifier, CNN and biLSTM models (sections
[4.1.1](#logisitc){reference-type="ref" reference="logisitc"},
[4.1.2](#cnn){reference-type="ref" reference="cnn"} and
[4.1.3](#bilstm){reference-type="ref" reference="bilstm"}), we use the
*TweetTokenizer* from nltk [^31] to separate the sentences into
sequences of tokens. With the transformer models (section
[4.1.4](#transformer_models){reference-type="ref"
reference="transformer_models"}), we use each model's included
tokeniser.

### Experimental Setup {#setup}

The datasets are divided into *train*/*validation*/*test* sets with an
80%/10%/10% split ratio. We use stratified sampling to preserve the
class ratio in all sets. For each experiment we train our model on the
*train* set, randomly shuffled at each epoch. Four times per epoch the
model is evaluated on the *validation* set and we track the evolution of
the metric (macro-F1) over time. The final *test* metrics are computed
using the state of the model when the *validation* metric was at its
highest point. This process is repeated three times for each experiment,
where the *train*/*validation*/*test* sets are randomly sampled from the
data each time and we report the average and standard deviation of the
macro-F1 on the test sets over the three runs. The early stopping uses
the macro-F1 score instead of the loss function, as it was observed that
the macro-F1 score would still increase for a while after the
over-fitting syndrome started appearing on the loss function. This
probably happens since the CEL and macro-F1 do not measure the same
thing: the CEL only takes the probability of the correct class into
account, but does not report if the predicted class is actually correct,
while the macro-F1 only measures the correctness of the predicted class.
Both metrics do not measure the same aspect of the problem and thus are
not synchronised; since in the end we are interested in the macro-F1
score, we preferred it for the early stopping.

We decided against using a fixed *test* set for the experiments, as its
choice would have been arbitrary and would therefore not have been a
reliable way of comparing models. We also decided not to use the
*validation* score to compare models, as we already rely on it for the
early stopping, and did not know if the difference in architectures had
an impact on the variability of the metric during training (with more
extreme high peaks in the curves that might impact the early stopping),
and therefore on the score used to compare the models. By using the
average over three random *test* sets, data that was not used during
training in any way, the results should be more conclusive regarding the
performance of the model.

All experiments were run with PyTorch [^32] [@pytorch] on a machine
running Ubuntu 18.04, with an Intel(R) Core(TM) i5-7500 quad-core CPU,
32GB of RAM, and a Nvidia GEFORCE GTX 1080 Ti with 12GB of VRAM.
Training of the transformer models lasted between 15-20 minutes on the
smaller models and datasets and 2-3 hours on the largest models and
datasets. A large amount of VRAM was required, from 2-3 GB for the
smaller models (e.g. BERT [@devlin2018bert]) to approximately 10GB for
the largest models (XLM-RoBERTa [@conneau2019unsupervised] or XLNet
[@yang2019xlnet]). Most models exist in different sizes (with more
layers and a larger hidden size), but we always use the *base* model.
Larger models usually come with a small improvement in results but have
significantly larger training time and memory requirements.

### Results

In this section, we compare the diverse architectures presented in
sections [4.1](#models){reference-type="ref" reference="models"} and
[4.2](#improvements){reference-type="ref" reference="improvements"}. We
start with the simple models, i.e. the Logistic Regression classifier,
the CNN and the biLSTM network (sections
[4.1.1](#logisitc){reference-type="ref" reference="logisitc"},
[4.1.2](#cnn){reference-type="ref" reference="cnn"} and
[4.1.3](#bilstm){reference-type="ref" reference="bilstm"}). We use these
as a baseline to determine the usefulness of more advanced transformer
models. For the transformers, we first evaluate the base models, then we
look for a good combination of training improvements and architectural
changes that can improve their performance. We focus on the changes and
differences between architectures and methods, but not on the details of
each architecture; we did not spend a lot of time tuning hyperparameters
as we felt that the architectural and methodological changes were more
interesting.

#### Simple models

More specifically, for the simple models we have:

-   A Logistic Regression classifier with a bag of unigrams and bigrams,
    with only the 20'000 most frequent features kept (see section
    [4.1.1](#logisitc){reference-type="ref" reference="logisitc"}). We
    limit the algorithm to 2'000 iterations.

-   A CNN following the architecture presented in section
    [4.1.2](#cnn){reference-type="ref" reference="cnn"}, trained with
    the weighted cross-entropy loss and tested with GloVe
    [@pennington2014glove] and FastText [@bojanowski2016enriching]
    embeddings. The CNN has 100 output channels, kernel sizes of 3, 4
    and 5 by the embedding size, and a dropout of 0.1 before the final
    layer.

-   A biLSTM network following the architecture from section
    [4.1.3](#bilstm){reference-type="ref" reference="bilstm"}, trained
    with the weighted cross-entropy loss and tested with GloVe and
    FastText embeddings. The LSTM units use a hidden size of 100, and a
    dropout of 0.1 is applied before the final layer.

Both the CNN and biLSTM models are trained over 50 epochs, with early
stopping after 2 epochs with no improvement and using the Adam optimiser
[@kingma2014adam] with a learning rate of $10^{-4}$ (other parameters
follow the default values from the PyTorch implementation). We use a
cosine scheduler that decreases the learning rate over the 50 epochs and
a mini-batch size of 32. For the GloVe and FastText embeddings, we use
the embedding layer from PyTorch. Since it needs to be initialised
beforehand, we need to define a vocabulary; to spare memory, we do not
use the full vocabulary from the trained embeddings, but only the words
present in our data. Results should be identical, and it makes the model
smaller. If we have unknown words in the dataset, they are added to the
embeddings as random vectors.

For each model, we report the average macro-F1 score on the test set
(with the standard deviation over three runs) in table
[\[tab:simple_classifier_test\]](#tab:simple_classifier_test){reference-type="ref"
reference="tab:simple_classifier_test"}. In bold, the best model, and if
two models are so close that the difference is non-significant both will
be marked in bold.

The first conclusion that can be drawn from these results is that the
more complex models often perform better than the simple Logistic
regression model and that the biLSTM usually outperforms the CNN,
probably due to its more complex architecture, most notably its ability
to see long-term dependencies better as it is not limited to a maximum
of five consecutive tokens like the CNN. Secondly, in English GloVe
embeddings are often better than FastText embeddings, but in French and
German the FastText embeddings are typically better. There are a few
exceptions, in particular Dachs FR performs better with the CNN, and
with MLMA FR and Dachs DE the GloVe embeddings perform better than
FastText when using the CNN. However, we see in a few instances very
large standard deviations; this happens when the model has a lot of
difficulties during training, and in one or two of the trials it does
not learn anything at all and stays in its initial state, a good
indication that the model is not adapted to the dataset.

We see that the results in English are generally better than the results
in French and German. This, of course, depends on the quality of the
datasets, but except for Dachs FR the results for French and German are
really bad. For the MLMA dataset, this can be explained by the very
small size of the dataset, and for Dachs DE the low amount of `hate`
data (a little over 2.5% of the total) certainly does not help. More
generally for the German datasets, one can also hypothesise that the
complexity of German grammar might be at cause; there are often long and
complex dependencies in German, and the various word declinations
(dative, accusative, genitive, etc.) might add too many variations of
the same word, and these models might not be smart enough to detect
those variations as the same word. A more careful data prepossessing
process with lemmatisation and morphological analysis could help.

For a baseline, it is also important to have more information about the
scores on the different classes. For each dataset, we report in table
[\[tab:simple_classifier_test_full\]](#tab:simple_classifier_test_full){reference-type="ref"
reference="tab:simple_classifier_test_full"} the macro-F1, and the
recall and precision on all classes for the best simple models according
to table
[\[tab:simple_classifier_test\]](#tab:simple_classifier_test){reference-type="ref"
reference="tab:simple_classifier_test"}. This gives us a more precise
baseline, before moving on to the transformer models.

We can see that with some datasets there is a significant imbalance
between the different classes. With Founta, Davidson and MLMA FR, the
metrics on `hate` are very much lower than for the `offensive` class.
This phenomenon is also present on other datasets, but to a lesser
extent, for instance with Jigsaw. This phenomenon is expected, as the
`hate` class is usually by far the class with the least data. It still
needs to be seen if the transformer models can correct this issue.

#### Base Transformer models

We now present results for the transformer models (section
[4.1.4](#transformer_models){reference-type="ref"
reference="transformer_models"}). We first evaluate existing models
taken directly from hugginface's transformers library, [^33] and for the
best model for each dataset we try the different classification heads
(section [4.2.1](#classification_heads){reference-type="ref"
reference="classification_heads"}) and data combination methods
(sections [4.2.2](#merging){reference-type="ref" reference="merging"}
and [4.2.3](#multitask){reference-type="ref" reference="multitask"}).
Afterwards, we evaluate the different training improvements presented in
sections [4.2.4](#further_pretraining){reference-type="ref"
reference="further_pretraining"} to
[4.2.8](#multilevel){reference-type="ref" reference="multilevel"}, each
of them independently, and apply the ones that are useful to obtain our
final models.

We train all models for 10 epochs with a cosine scheduler and the Adam
optimiser [@kingma2014adam], with early stopping on the macro-F1 after
two epochs without improvement, as the model usually reaches a maximal
score after three to five epochs. The learning rate for the transformer
part is $2\times10^{-5}$ on the recommendation of @devlin2018bert, and
the learning rate for the classification head is $5\times10^{-5}$ (we
found during experimentation that a higher rate for the classification
head was beneficial). A mini-batch size of 64 was used for all models.

In table
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"} we report for each dataset
the scores for a few transformer models. Those models are taken as-is,
without any change to the classification head (we use the default CLS
classification head). We always take the *base* models, and the *cased*
instead of *uncased* version if applicable.

Firstly, we see that BERT [@devlin2018bert], the original transformer
model, underperforms compared to the newer RoBERTa [@liu2019roberta]
model. This difference is highly dependent on the dataset, with only a
0.5% difference with RoBERTa on Dachs EN and more than 8% on Jigsaw, but
RoBERTa is always the best model for English data.

Furthermore, in English and French the best monolingual models always
outperform the multilingual model. This is expected, as RoBERTa,
FlauBERT [@le2019flaubert] and XLM-RoBERTa [@conneau2019unsupervised]
are very similar architecturally, but monolingual models are more
adapted since they do not include all unnecessary data for other
languages. In the case of the German datasets, the multilingual model
here is better, presumably due to the larger amount of training data
(2.5TB in total for XLM-R, with 66GB of German data) compared to
german-BERT (only 12GB). We can already conclude that if we want to use
those multilingual models in the future to improve our results with
cross-lingual toxicity detection, we will have a disadvantage that will
need to be compensated before hoping to surpass the original monolingual
score. These results give us a baseline for the transformer models, and
they show what kind of results we can hope for when using the pretrained
models from the huggingface's transformers library directly, without any
changes.

#### Classification heads {#classification-heads}

In table
[\[tab:results_classification_head\]](#tab:results_classification_head){reference-type="ref"
reference="tab:results_classification_head"} we show for each dataset,
and using the best transformer models from table
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"}, the scores using the
different classifications heads introduced in section
[4.2.1](#classification_heads){reference-type="ref"
reference="classification_heads"}.

The first thing we observe is the similarity of the scores for the
different classification heads, with close averages and relatively large
standard deviations. Since we use cross-validation and randomly sample
the *train/validation/test* sets, we have different data for each run,
which creates some variance in the results. Besides, there is some
inherent variance during the training of a transformer model; the
initialisation of weights has an non-negligible impact
[@zimmerman-etal-2018-improving], so even if we retained the same data
in all trials we would get some variation.

Secondly, although the best model is not the same for all datasets, CLS
and CNN are often among the better classification heads. For each of
these heads, there are a lot of optimisations that could be done; we
could add some layers, some regularisation, change kernel sizes, change
the learning rate, etc., so it might be possible to achieve different
and better results. The results here show that there is some potential
and that the way information is extracted from the transformer has some
importance, even if the improvement here is not as large as hoped.
According to results from @mozafari2019bertbased, this improvement seems
to be highly dependant on the dataset and classification task. On one of
their datasets, they achieve significant improvement by using a
specialised classification head, while the difference is negligible on
another dataset.

Finally, we note a few instances with very low scores, these happen when
in one or more of the three trials the model is not able to learn
meaningful information. It sometimes gets stuck in a state close to the
initial random state, giving a score of around 33%, sometimes lower,
equivalent to a random prediction. It only happens with harder datasets,
here MLMA FR and Dachs DE. The large standard deviation in the case of
Dachs DE shows that the model was still able to fit to the data at least
one time. We considered ignoring those very bad trials but decided
against as it is a valuable indication of the quality of the model and
its compatibility with a dataset.

Overall, these results show that the most important part of the
architecture is still the transformer itself and not the way the data is
extracted. Since all classification heads also have access to the output
of the `[CLS]` token, which is supposed to give all necessary
information regarding the input, the additional information from the
more complex classification heads might be redundant in some cases.

#### Dataset Combining

We now try to combine data from multiple datasets. This can be done
either by merging directly all the datasets of the language (section
[4.2.2](#merging){reference-type="ref" reference="merging"}) or by using
multitask learning (two versions possible as described in section
[4.2.3](#multitask){reference-type="ref" reference="multitask"}, with
either the final layer or the classification head specific to each
dataset). We try combining each dataset *(primary* dataset) with a new
dataset (*secondary* dataset) created from all remaining data in the
language (e.g. with Dachs EN, we combine it with the combination of
Davidson, Founta and Jigsaw data). Better results might be achievable by
testing other subsets of data for the secondary datasets, but we did not
have time to consider all possible combinations. The goal here was to
put as much diversity as possible in the training set to hopefully
better fit the test data.

The different methods can be seen as different levels of data
combination, and we can place them on a scale; on one side, we have a
model completely specific to the actual dataset, with no combining
(*None*). Then, when we do multitasking at the classification head level
(*MT-class*), the model still has quite a large part that is specific to
the dataset. With the multitasking only at the final layer (*MT-final*),
the model has less specific parts, only the final layer, and most of it
is common. Finally, with full dataset merging (*Merge*), none of the
model is specific to the dataset. We show those four approaches in table
[\[tab:results_combining\]](#tab:results_combining){reference-type="ref"
reference="tab:results_combining"}. Results in the *None* column are
taken from table
[\[tab:results_classification_head\]](#tab:results_classification_head){reference-type="ref"
reference="tab:results_classification_head"}.

The performance of data combination is highly dependent on the datasets
themselves, but in most cases combining data is detrimental to the
score. It might be slightly useful for the datasets with the lowest
scores in each language (Dachs EN, Founta, MLMA FR and Germeval) since
we include data from better datasets, but it is detrimental to datasets
with better scores (as we add data from datasets that do not perform as
well). Most of the time we are within a margin of error, meaning that
combining data is not very useful especially considering the added
training time. The only clear improvement is with data merging for MLMA,
where we gain a few percentage points, but the score still stays very
low.

When combining datasets, multitasking is usually a better approach
compared to naive merging, except for MLMA FR and Dachs FR. This is due
to the disparity in sizes of the two datasets, and the fact that we
reduce the largest dataset to the size of the smallest dataset with
multitask learning. The Dachs FR dataset will be significantly reduced
when combined with the smaller MLMA FR dataset, leading to a smaller
total amount than with Dachs FR only. For MLMA FR, adding the Dachs FR
data with multitask will only double the total size, while merging the
two datasets will result in a remarkably larger total dataset, which is
extremely valuable.

Ultimately, the fundamental problem of data combination is the domain
gap between the datasets. They do not all have the same exact
definitions and data distribution, and the model will have to generalise
more to fit all regrouped datasets, therefore the score on a single
dataset will be worse. Dataset combining might still be useful in a
zero-shot approach where the evaluation dataset is not used during
training; if we used multiple datasets during training, there should be
a higher chance of generalising correctly on unseen evaluation data. We
did not verify this claim due to time constraints.

All these architectural changes (*transformer type*, *classification
head* and *data combining*) have some non-negligible influence. We
cannot conclude that a method is always effective, as the results vary
significantly between datasets and there is a lot of variance. However,
it is apparent from the results in tables
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"},
[\[tab:results_classification_head\]](#tab:results_classification_head){reference-type="ref"
reference="tab:results_classification_head"} and
[\[tab:results_combining\]](#tab:results_combining){reference-type="ref"
reference="tab:results_combining"} that it is possible to improve
results from the original model by combining some of those options. This
gives us a new baseline, summed up in table
[\[tab:baseline_arch\]](#tab:baseline_arch){reference-type="ref"
reference="tab:baseline_arch"}, from which we can then try the remaining
training improvements.

#### Further training improvements

We now evaluate the effects of each of the remaining training
improvements individually, then combine the options that were beneficial
to create a final model. More specifically, we consider the following
methods, with results reported in table
[\[tab:results_options\]](#tab:results_options){reference-type="ref"
reference="tab:results_options"}:

-   *Pretraining* (section
    [4.2.4](#further_pretraining){reference-type="ref"
    reference="further_pretraining"}): pretrain the transformer model on
    the training data. [^34] This pretrained model is then used as a
    starting point for the usual classification. In the cases of data
    combining, we use all combined data for the pretraining.

-   *Data augmentation* (section
    [4.2.5](#data_augmentation){reference-type="ref"
    reference="data_augmentation"}): augment the minority classes to
    reach a similar size to the majority class (for a class $C$, and a
    majority class $M$, augment by a factor $\floor*{\frac{|M|}{|C|}}$).

-   *Data Normalisation* (section
    [4.2.6](#data_cleaning){reference-type="ref"
    reference="data_cleaning"}): use MoNoise [@goot2017monoise] to
    normalise the data on top of the existing cleaning process. Perform
    synonym replacement, random insertions and random swaps, all with
    probability 0.3.

-   *Ensemble* (section [4.2.7](#ensemble){reference-type="ref"
    reference="ensemble"}): make a majority vote between three versions
    of the same model.

-   *Multilevel classification* (section
    [4.2.8](#multilevel){reference-type="ref" reference="multilevel"}):
    train a model to differentiate toxic and non-toxic content, and
    train another model to differentiate hateful and offensive content.

Unfortunately, it seems that the vast majority of these options do not
work. The standard deviation shows that there is not a large difference
between the options, thus we cannot conclude with confidence that they
never work, but it also shows that there is not a lot of potential. We
can think of a few reasons why these methods are ineffective:

-   For *pretraining*, the problem might come from the pretraining
    duration. If we pretrain the model for too long, the transformer
    might start to overfit on the training data and will then encounter
    some difficulties with the test data. No option to reduce the
    training time was found in the used script. We could maybe reduce
    the size of the data, but the model might start to overfit even
    more. Another solution could be to pretrain on significantly more
    data to avoid this overfit.

-   For *augmentation*, we think that too much noise is added. The
    WordNet models do not use context to generate synonyms, so a lot of
    them might be inaccurate and alter the meaning of the sentence.

-   For *data normalisation*, we think that some of the incorrectly
    spelt words have some use for the classification and that correcting
    them can have a negative impact. There is perhaps a correlation
    between the level of grammatical and orthographical correctness of a
    message and its toxicity, and such precious information is lost with
    normalisation.

-   For *ensemble learning*, the problem might come from the distinction
    between the test and validation sets. The three models for the
    ensemble are selected to maximise the validation score but do not
    consider the test score. When ensembling three models optimised for
    the validation set, we might start to overfit in some way on this
    validation set, and the ensemble will be less capable of
    generalising with other data, here the test set.

-   For *multilevel classification*, a source of error might be the
    grouping of `hate` and `offensive` content. The first level of the
    classification (binary classification `toxic` or not) might struggle
    to find common patterns for the `hate` and `offensive` classes, as
    `offensive` content might be more similar to `none` than `hate` in
    some cases, leading to misclassifications.

#### Summary of results

We summarise in table
[\[tab:results_withoptions\]](#tab:results_withoptions){reference-type="ref"
reference="tab:results_withoptions"} the architecture and options chosen
for each dataset. We report in table
[\[tab:summary_metrics\]](#tab:summary_metrics){reference-type="ref"
reference="tab:summary_metrics"} more detailed results for the models,
where we compare the best simple model (among Logistic Regression, CNN
and biLSTM), with the base transformer model (i.e. a transformer model
with the CLS classification head and no other improvement, see models in
table
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"}), and the improved
transformer model (a model with all options enabled, see table
[\[tab:results_withoptions\]](#tab:results_withoptions){reference-type="ref"
reference="tab:results_withoptions"}).

It is clear that the transformer models outperform the traditional
biLSTM and CNN based methods, often by a significant margin. Besides, in
the cases where the improvement on the macro-F1 is not very big, we
usually see one metric where the improvement is considerable. For
instance on Jigsaw, with only 1.7% of improvement on the macro-F1, we
have more than 6% of improvement for the recall on `hate`, an important
metric for this task. Similar behaviours can be observed with the Founta
and Davidson datasets. With the improved transformer models, the results
are slightly higher than with the base transformer model. Not all
datasets are similarly affected, but once again we have some cases where
specific metrics are particularly improved. On Dachs EN, we gain a lot
of recall on `hate` and only lose a small amount on the `offensive`
metrics. Similar behaviours were observed with the Founta and Davidson
datasets. With French and German data, we also improve the scores from
the simple model to the base transformer, and then to the improved
transformer (Dachs DE being the exception, no option improved the
results for the transformer model).

For the most part, the improvements to the transformer models are not
extremely useful. In most cases, they provide better scores on some
metrics, but it is relatively unpredictable and we cannot significantly
improve the scores with these methods. The largest improvement comes
from the use of transformer models in the first place.

In figure [4.15](#fig:p_r_plot){reference-type="ref"
reference="fig:p_r_plot"} we plot the evolution of metrics between the
simple models and advanced transformer models (values taken from table
[\[tab:summary_metrics\]](#tab:summary_metrics){reference-type="ref"
reference="tab:summary_metrics"}).

![[\[fig:p_r\_plot\]]{#fig:p_r_plot label="fig:p_r_plot"}Evolution of
metrics from the simple models to the improved transformers for the
`none`, `offensive` and `hate` classes. The tail of the arrow shows the
precision and recall of the simple model, the head shows the metrics for
the improved transformer.](p_r_plot){#fig:p_r_plot width="\\textwidth"}

We observe a clear trend of improvement for all models, as most arrows
tend to move towards the top right corner (representing the perfect
model). More interestingly, the arrows often point towards the grey line
representing the equilibrium between precision and recall, which
corresponds to an improvement of recall on the `offensive` and `hate`
classes, and an improvement of precision on the `none` class.

Finally, we report in table
[\[tab:results_binary\]](#tab:results_binary){reference-type="ref"
reference="tab:results_binary"} results obtained when considering a
binary classification problem. We use for each dataset the best model
for the multiclass problem (table
[\[tab:results_withoptions\]](#tab:results_withoptions){reference-type="ref"
reference="tab:results_withoptions"}), but we train it with only two
classes, `toxic` if the label was `hate` or `offensive`, and `none`
otherwise.

We observe a significant increase in the scores with Founta and
Davidson; these two datasets had problems differentiating `hate` and
`offensive`, but when regrouping the two classes this problem disappears
and we have very good results. While we used to have better results for
the recall of `hate` and `offensive` with Dachs FR than with MLMA FR,
now that we have regrouped the two classes MLMA FR gets better recall on
`toxic` than Dachs FR. The score on the `none` class is still as bad,
but this also shows that one problem of MLMA FR was the differentiation
of `hate` and `offensive`.

The problem of differentiating the `hate` and `offensive` classes does
not only come from the models, but also the datasets. For some of them,
we observed some inconsistencies in the annotations of content in the
`offensive` and `hate` classes. It was observed that the presence of
some keywords would make the annotator immediately label the tweet as
`hate`, without considering the context the word is used in
[@davidson2017automated], which can lead to similar content being in
both classes if there is too much bias with some annotators.

Globally, we get acceptable results on the `toxic` class, and in the end
it is the most important thing. For most applications, we first want to
detect all toxic content (i.e. maximise the recall on `toxic`) and not
let anything through. Differentiating `hate` and `offensive` is less
important in most cases. Naturally, one could imagine an application
where offensive content is allowed but not hateful content, in which
case the multiclass approach is useful. One could also adapt the model
into one that gives a toxicity score, from 0 if offensive only to 1 if
hateful, we would then be able to easily tune the system for the desired
application. We opted for the multiclass approach as it gave more
understandable results (with a multiclass problem, the recall and
precision metrics are directly mapped to real-world performance on a
moderation system), but also since all annotations in the datasets and
previous works already follow the multiclass approach.

### Comparison with previous results

It is important to compare our results with previously published results
on the same datasets to know if our models are working correctly.
Unfortunately, it is often hard to find studies that use exactly the
same data, processing, classification task and similar models to allow
for precise comparison. Sometimes some classes are ignored, sometimes
datasets are merged, and we do not always have all the original data at
our disposal.

We were still able to find two datasets where we had a similar enough
classification task to compare: the Dachs and Davidson datasets. For the
others, we could not find adequate experiments; there were no
experiments exclusively on Founta (it was merged with other datasets
[@aluru2020deep]), MLMA FR had experiments with more classes and mixed
with the other languages of that dataset [@ousidhoum2019multilingual]
(English and Arabic) and Jigsaw and Germeval were used for machine
learning competitions, again with different goals, so we could not
compare with them either.

#### Dachs dataset

In the original Dachs paper [@charitidis2019countering], the task
differs from ours; they perform two binary classification tasks, `hate`
or `not-hate` and `attack` or `not-attack`. They tested a few models,
but not the transformers, so we would like to see how these perform
compared to their models. We unfortunately do not have all the data that
was used in the original experiment; table
[\[tab:dachs_data\]](#tab:dachs_data){reference-type="ref"
reference="tab:dachs_data"} reports the amount of data we have for the
Dachs dataset, with the percentage of the original amount we were able
to collect. We still hope to improve their results as we have more
advanced architectures.

For all languages, we use models with the best architectural choices and
options, except for self-ensemble and data combining, as we do not have
a secondary dataset with the same annotations to use (see table
[\[tab:results_withoptions\]](#tab:results_withoptions){reference-type="ref"
reference="tab:results_withoptions"}), and compare the score with their
best non-ensemble model.

For the two classification tasks, @charitidis2019countering reported the
macro-precision and macro-recall (the averages of the precision or
recall on the two classes), the macro-F1 and toxic-F1 (i.e. the F1 score
on the `hate` class for the `hate` problem and the F1 score on the
`attack` class for the `attack` problem). Since it is unclear if they
used some type of average on their results, we use our usual method with
the average on the test set over three runs. We report our results
alongside the results provided by @charitidis2019countering in table
[\[tab:dachs_metrics\]](#tab:dachs_metrics){reference-type="ref"
reference="tab:dachs_metrics"}.

To begin with, it is important to note that the models used here were
optimised for the multiclass problem (`hate`, `offensive` and `none`)
and not on the two binary classification problems presented here.
Repeating the same process as previously (evaluating all options to
determine the best combination) but directly on the target tasks might
give better results. Still, we see that in English and French we are
able to get better results than the authors of the dataset. We have more
complex models, but also clearly less data. This is a good sign that
transformer models are indeed useful and better than the simpler models.
In German, we are not able to match their scores, probably due to the
very limited amount of data at our disposal, especially on the `hate`
class. On the `attack` class, with a larger percentage of the original
data, we are closer although we still do not match their results.

#### Davidson dataset

Here, we compare our models to those of @mozafari2019bertbased on the
Davidson dataset. In their paper the weighted F1-score is used as a
metric, which by itself does not give a lot of information especially
about the minority classes, but they also report a confusion matrix
which allows us to recompute all other metrics for a better comparison.
The metrics given in table
[4.1](#tab:davidson_metrics){reference-type="ref"
reference="tab:davidson_metrics"} are for their best model, BERT
[@devlin2018bert] using a custom classification head based on a CNN, and
we compare those with our best model for this dataset, RoBERTa
[@liu2019roberta] with the CNN to LSTM classification head (see section
[4.2.1](#classification_heads){reference-type="ref"
reference="classification_heads"}). Once again, since we do not know
their methodology for reporting results, we report the average over
three runs on random test sets for our models.

::: {#tab:davidson_metrics}
                         P      R     F1
  -- -------------- ------ ------ ------
     none             91.6   92.2   91.9
     offensive        94.0   97.3   95.6
     hate             59.2   29.5   39.4
     macro avg        81.6   73.0   75.6
     weighted avg     91.6   92.6 
     none             88.9   90.3   89.6
     offensive        95.3   94.3   94.8
     hate             48.7   52.0   50.0
     macro avg        77.6   78.9 
     weighted avg     91.5   91.2   91.3

  : [\[tab:davidson_metrics\]]{#tab:davidson_metrics
  label="tab:davidson_metrics"}Comparison on the Davidson dataset. They
  use BERT with a CNN-based classification head, we use RoBERTa with the
  CNN to LSTM classification head.
:::

If we look at the main metric they report, the weighted F1-score, we do
not quite match their results. However, our model has a better macro-F1
score. The `hate` class performs significantly better with our model,
and this has a larger influence on the macro-F1 score compared to the
weighted-F1 score. We score a little lower with the `none` and
`offensive` classes, but we think the gain in recall on `hate` makes it
worth.

Cross-lingual Toxicity Detection {#cross_lingual}
================================

In this chapter, we present our work on Cross-lingual Toxicity
Detection. Three different aspects of this problem were considered.

**Cross-lingual combination**: The first use of cross-lingual models is
to offer the possibility of regrouping datasets from multiple languages
to form a larger dataset with more training data, and potentially get
better results on individual datasets. This is an extension of the work
on dataset merging and multitask learning introduced in sections
[4.2.2](#merging){reference-type="ref" reference="merging"} and
[4.2.3](#multitask){reference-type="ref" reference="multitask"} (with
results in table
[\[tab:results_combining\]](#tab:results_combining){reference-type="ref"
reference="tab:results_combining"}). These methods could work when
providing additional data to a weak dataset, and a similar principle can
be applied across languages, although combing datasets with too much
content difference will still be detrimental. Results would be
completely dependent on the choice of datasets and their combination,
requiring an exhaustive search to determine the optimal combination. We
could not find any cases were cross-lingual data combination was
beneficial. This task was already challenging with monolingual models,
but here the addition of a language gap makes this task even more
difficult.

**Zero-shot classification**: The second aspect is zero-shot
classification, with the goal of designing models able to classify data
in a given language, while only possessing training data in other
languages. This aspect is especially important for languages with small
populations where no data are available, so that classification models
can still be used by them.

**Multilingual classification**: Finally, as an extension of zero-shot
classification, we might want to build a single multilingual model that
can classify multiple languages simultaneously. It should work correctly
on the language used for training, but also on unknown languages.

Models {#cross_lingual_models}
------

We experiment with two categories of models for the cross-lingual tasks:
section [5.1.1](#multilingual_models){reference-type="ref"
reference="multilingual_models"} presents models that are purely
multilingual, directly understanding data in multiple languages, while
section [5.1.2](#translation_models){reference-type="ref"
reference="translation_models"} presents models that require data
translation at some point in the pipeline. The difference between these
two types of models is illustrated in figure
[5.1](#fig:multi_vs_translation){reference-type="ref"
reference="fig:multi_vs_translation"}.

![[\[fig:multi_vs_translation\]]{#fig:multi_vs_translation
label="fig:multi_vs_translation"}On the left, a purely multilingual
architecture, on the right, a multilingual architecture that requires
translations, illustrated here with English and German training data,
and French test data. The purely multilingual model uses the data
directly, while the other model needs to translate the German and French
data into English to use the monolingual English
model.](multi_vs_translation){#fig:multi_vs_translation width="70%"}

### Purely multilingual models {#multilingual_models}

We start with *purely* multilingual models that can utilise data from
multiple languages without any additional preprocessing required. We use
three types of purely multilingual models; models that use LASER
[@artetxe2018massively] sentence embeddings, models that use MUSE
[@conneau2017word] words embeddings, and models based on transformers
(see section [4.1.4](#transformer_models){reference-type="ref"
reference="transformer_models"}).

#### LASER embeddings {#laser}

LASER embeddings [@artetxe2018massively] are multilingual sentence
embeddings generated by a biLSTM encoder, trained on 93 languages as
part of the encoder-decoder architecture illustrated in figure
[5.2](#fig:laser){reference-type="ref" reference="fig:laser"}.

![[\[fig:laser\]]{#fig:laser label="fig:laser"}Architecture used to
train the LASER embeddings encoder, illustration from
@artetxe2018massively](laser){#fig:laser width="80%"}

Byte-pair encoding [@gage1994new] (see section
[3.3.1](#bpe){reference-type="ref" reference="bpe"}) is used on the
input data using a vocabulary acquired on the concatenation of all
learning corpora (the encoder does not know the language of the input).
A max-pool unit creates a sentence embedding using the output of a
biLSTM network. The sentence embedding is then used in the decoder,
trained to generate an English or Spanish translation of the original
sentence, depending on the language specified in the input of the
decoder. In our experiments, LASER embeddings are coupled with a
multi-layer perceptron (MLP), a sequence of linear and non-linear
layers. We use the *laserembeddings* Python library [^35] to generate
the embeddings. Unlike with the transformer models, there is no
fine-tuning happening to the biLSTM network that generates the
embeddings.

#### MUSE embeddings {#muse}

MUSE embeddings [@conneau2017word] are multilingual word embeddings with
support for 30 languages, generated by aligning FastText
[@bojanowski2016enriching] word embeddings into a global embedding space
using a bilingual dictionary that provides anchor points for the
alignments of the two embeddings spaces. We use Gensim [@rehurek_lrec]
to load them into the embedding layer from PyTorch [@pytorch], with
embedding tuning disabled to avoid a dissociation of the embedding space
of different languages in the case of zero-shot learning. Those
embeddings are used in conjunction with the biLSTM model presented in
section [4.1.3](#bilstm){reference-type="ref" reference="bilstm"}.

#### Multilingual transformer {#ml_transformer}

For the transformer model we use XLM-RoBERTa [@conneau2019unsupervised],
based on RoBERTa [@liu2019roberta] and trained on 2.5TB of data from 100
languages. We connect it to the CNN classification head (section
[4.2.1.3](#cnn_head){reference-type="ref" reference="cnn_head"}), as it
usually performs well (see table
[\[tab:results_classification_head\]](#tab:results_classification_head){reference-type="ref"
reference="tab:results_classification_head"}), and using a common
classification head will make the comparison between all models easier.
The same CNN classification head is also used for all other transformer
models in this chapter.

### Translation-based models {#translation_models}

A second approach to cross-lingual classification is to translate all
available data into a single language, and then use a monolingual model.
In this section we present multiple architectures based on this idea.

#### Data translation

To generate the translations, we use MarianNMT [@mariannmt], a neural
machine translation model trained on the OPUS [@tiedemann-2012-parallel]
parallel corpus by a research team at Helsinki University [^36]
[@TiedemannThottingal:EAMT2020]. We use a Python implementation of the
model available in *huggingface's transformers* library. [^37] All
translations are done in advance on the cleaned data (cleaning according
to section [4.3.2](#data_preprocessing){reference-type="ref"
reference="data_preprocessing"}), and it takes approximately 0.1 to 0.2
seconds per sentence on a NVIDIA GeForce GTX 1080TI. We need a specific
model for each language pair, but since there is not a model for every
language pair [^38] a series of translations is sometimes necessary.
There are also a few models that understand language groups (e.g.
Romance/Celtic/Nordic languages) that can be useful to avoid having to
load too many translation models concurrently.

Before using the translations, it is important to know their quality. We
show in table
[\[tab:translations\]](#tab:translations){reference-type="ref"
reference="tab:translations"} a few experiments conducted with
translations between English and French sentences. Censorship of extreme
words was added for the report but was not present during the
experiments.

Overall the translations are quite good and the original sense is
usually maintained. For instance, the presence and position of the word
*f\*ck* have a large impact on the translated sentences (1-6). The
structure of the sentence changes to accommodate this word, instead of
simply having the literal translation inserted in the output. Some rare
expressions and words are not translated by the model (7-9), and the
English word is simply kept in French. In these cases, a French model
that sees this word might still be able to learn some patterns, as this
non-translation should happen in multiple cases. For some homonyms, the
quality of the translation depends on the whole sentence, and the
original meaning might only be preserved in specific cases, e.g. (10-11)
where the presence of *\"such\"* has an impact on the final meaning.
With other homonyms, the translation seems to work without issues
(13-14). If we look at French sentences translated to English, we have
similar patterns; adding an offensive word will usually fundamentally
change the structure of the translation (17-20), it does work with some
homonyms (21-26) and common expressions are correctly translated (27).
Finally, there are a few cases where the translation just does not make
sense (12, 16), but it usually happens with a single word, and adding
more context helps (10, 15).

Overall these models seem to work quite well, at least with common
sentences and insults, we therefore think that translation-based models
have some potential.

#### Simple translation models {#simple_translation_models}

Using Opus-MT [@TiedemannThottingal:EAMT2020] we can now translate the
available data into many languages, allowing for the use of the
following architectures that utilise data translation and a monolingual
transformer model with the CNN classification head (see section
[4.2.1.3](#cnn_head){reference-type="ref" reference="cnn_head"}):

-   translate all data into English and use RoBERTa [@liu2019roberta]
    with the CNN classification head;

-   translate all data into French and use FlauBERT [@le2019flaubert]
    with the CNN classification head;

-   translate all data into German and use german-BERT [^39] with the
    CNN classification head.

We did not experiment with simpler models (Logistic Regression
Classifier, CNN or biLSTM) in this translation approach, as we saw in
the first part of this project that transformers models always
outperformed them (comparison in table
[\[tab:summary_metrics\]](#tab:summary_metrics){reference-type="ref"
reference="tab:summary_metrics"}).

Improvements {#multilingual_improvments}
------------

Starting from the idea of using monolingual models with translations
(section [5.1.2.2](#simple_translation_models){reference-type="ref"
reference="simple_translation_models"}), we can create more advanced
architectures where data translation is used to train multiple models in
parallel. This idea is inspired by the work of @wan-2009-co
[@pamungkas-patti-2019-cross] and adapted to transformer models. We
present here three different architectures that use translations and
joint learning.

### Joint transformers {#joint_transformers}

Table
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"} showed that monolingual
models usually outperform multilingual models, since they are trained on
more data from the target languages, and they do not have to learn the
specificities of multiple languages simultaneously. We propose here a
joint architecture, *joint transformers*, where we use a monolingual
transformer model (see section
[4.1.4](#transformer_models){reference-type="ref"
reference="transformer_models"}) for each language in the data. The goal
is to utilise the advantages of monolingual models while still being
compatible with multiple languages. Such a model is illustrated in
figure [5.3](#fig:joint_transformer){reference-type="ref"
reference="fig:joint_transformer"}.

![[\[fig:joint_transformer\]]{#fig:joint_transformer
label="fig:joint_transformer"}*joint transformer* architecture, here
with English training data and French test data. We train an English
transformer with the English data, but also a French transformer by
translating English data into French. During the evaluation, the French
model takes the French test data directly, and the English model takes
the French data translated into
English.](joint_transformer){#fig:joint_transformer width="90%"}

A monolingual transformer is initialised for each language of the data,
and their outputs are sent into a single linear layer to make a
prediction. During training, all data are translated into all other
languages and the original and translated data are given to all
transformers in parallel. All models receive a different representation
of the original sentence and are trained jointly. For the evaluation the
same process applies, with multidirectional data translations to feed
all transformers in parallel.

The largest drawback of this architecture is its scalability, as we need
a transformer model for every language in the data, but also need to
perform all the multidirectional translations, and it becomes
impractical for more than two or three languages. We can mitigate this
problem by selecting two or three main languages that will have a
transformer, and any additional language will be translated into the
main languages. This technique is also useful if a language simply does
not have a monolingual model.

### Joint LASER {#joint_laser}

The second proposed approach, *joint LASER*, is similar to the *joint
transformers* model (section
[5.2.1](#joint_transformers){reference-type="ref"
reference="joint_transformers"}), except that we use a multi-layer
perceptron with LASER sentence embeddings [@artetxe2018massively] for
each language in the dataset in place of the transformers. Their outputs
are also concatenated and sent to linear layers to make a prediction
(see figure [5.4](#fig:joint_laser){reference-type="ref"
reference="fig:joint_laser"}).

![[\[fig:joint_laser\]]{#fig:joint_laser label="fig:joint_laser"}*joint
LASER* architecture, with English train data and French test data. Each
LASER+MLP module receives data from a unique
language.](joint_laser){#fig:joint_laser width="95%"}

Both LASER+MLP blocks operate in a monolingual regime, but their outputs
are still in the same embedding space. The scalability problem is less
pronounced than with the *joint transformers*, due to the smaller size
of LASER+MLP blocks. We also do not need to find monolingual models for
each language, as we directly use multilingual models. If the target
language is not supported by LASER embeddings (which should be rare as
LASER embeddings support more than 93 languages), we can always
translate the data into a supported language.

### Joint transformer and LASER {#joint_transformer_laser}

With the *joint transformers* and *joint LASER* architectures, similar
models but in different languages are connected, hoping that the
translations will generate multiple representations of the same
sentence. We can also join completely different models together to
achieve a similar effect. The base idea is similar to ensemble learning,
in the sense that adding some variety to the models might result in
better results, the difference being that the combination of all models
is done more carefully than with a simple majority vote.

We propose to combine a transformer with a LASER+MLP block. Both will
receive the same data and produce different representations of the
input, then joined together to make a prediction. During training, the
final layer, the transformer and the MLP are all updated together. For
the transformer, there are two choices; use a monolingual transformer,
in which case translations are required to send data to the transformer,
or use a multilingual transformer model so that no translations are
required. The LASER+MLP block will work in a multilingual regime by
receiving all data directly. We illustrate these two variants in figures
[5.5](#fig:joint_purely_ml){reference-type="ref"
reference="fig:joint_purely_ml"} and
[5.6](#fig:joint_transformer_laser){reference-type="ref"
reference="fig:joint_transformer_laser"}.

![[\[fig:joint_purely_ml\]]{#fig:joint_purely_ml
label="fig:joint_purely_ml"}*joint transformer and LASER* architecture,
variant with the multilingual transformer, with English train data and
French test data. Both the LASER+MLP and transformer modules receive
training and test data as-is, without any
translations.](images/joint_purely_ml.png){#fig:joint_purely_ml
width="95%"}

![[\[fig:joint_transformer_laser\]]{#fig:joint_transformer_laser
label="fig:joint_transformer_laser"}*joint transformer and LASER*
architecture, variant with the monolingual transformer, with English
train data and French test data. The monolingual RoBERTa transformer
model only receives English data (original during training, translated
from French during evaluation), and the LASER+MLP block will receive
English data during training, and French data during
evaluation.](images/joint_transformer_laser.png){#fig:joint_transformer_laser
width="95%"}

Since we join two radically different models (transformer and
LASER+MLP), each model should interpret the input with two different
points of view, maybe focus on different concepts, and the concatenation
of those representations should give a better summary for the sentence,
and therefore a more accurate classification.

### Hurtlex

Iterating on the previous joint architectures, we propose variants to
the simple translation models (section
[5.1.2.2](#simple_translation_models){reference-type="ref"
reference="simple_translation_models"}) and the *joint transformer and
LASER* approach (section
[5.2.3](#joint_transformer_laser){reference-type="ref"
reference="joint_transformer_laser"}), where we also concatenate a
representation of the presence of offensive and hateful words in the
sentence.

We use Hurtlex [^40] [@hurtlex], a multilingual lexicon that classifies
offensive and hateful words into 17 categories (cognitive disabilities
and diversity, moral and behavioural defects, words related to social
and economic disadvantage, animals, male genitalia, etc.) For each
sentence, we create a vector of length $17$ (the number of Hurtlex
categories), indicating how many words from each category are in the
sentence, and this vector is directly concatenated to the output of the
transformer (and the output of the LASER+MLP block for the joint model).
Since the Hurtlex categories are the same across all languages, this
representation is multilingual and might be able to detect rare insults
and swearwords whose sense might otherwise be lost in translation.

Experiments
-----------

We now present results for the models introduced in sections
[5.1](#cross_lingual_models){reference-type="ref"
reference="cross_lingual_models"} and
[5.2](#multilingual_improvments){reference-type="ref"
reference="multilingual_improvments"} for the two different aspects of
cross-lingual classification we are interested in: *zero-shot*
classification and *multilingual* classification.

### Experimental setup {#experimental-setup}

We report the averaged macro-F1 score on the test sets in our results,
with the standard deviation over three runs, each with a different
randomised *train*/*validation*/*test* split (with 80%/10%/10%
proportions), and early stopping on the validation score. Unless
specified otherwise, all other cleaning, training and architecture
parameters are unchanged from the first part of the project (section
[4.3](#experiments){reference-type="ref" reference="experiments"}).

### Zeroshot classification {#ml_zeroshot}

In this section, we consider only one dataset for each language and try
to improve the macro-F1 score for each language in a zero-shot approach.
We are interested in obtaining the best possible score in a language by
only using data from other languages for training. We only experiment
with the Dachs datasets as we found that they were relatively clean,
with good scores in French and English, and we expect the annotations
and class distributions to be quite similar between the three languages
as they come from the same authors. Excluding the other datasets from
the start removes an important variable for the results (the selection
of the best combination of datasets) and we can therefore concentrate on
comparing the different architectures.

#### Multilingual models

We first experiment with the *purely* multilingual models, the ones that
recognise input data in any language. We have a multi-layer perceptron
(MLP) that uses LASER sentence embeddings [@artetxe2018massively], a
biLSTM with MUSE word embeddings [@conneau2017word], and the
multilingual XLM-RoBERTa transformer model [@conneau2019unsupervised]
with the CNN classification head (all three models presented in section
[5.1.1](#multilingual_models){reference-type="ref"
reference="multilingual_models"}). There are also the joint approaches
from sections [5.2.3](#joint_transformer_laser){reference-type="ref"
reference="joint_transformer_laser"} and
[5.2.4](#hurtlex){reference-type="ref" reference="hurtlex"}, where the
output of the transformer is joined to the output of the LASER+MLP block
and/or the Hurtlex [@hurtlex] representation. For each target language,
we show in table
[\[tab:zero_shot_dachs\]](#tab:zero_shot_dachs){reference-type="ref"
reference="tab:zero_shot_dachs"} the macro-F1 scores obtained by
training the model on either one or two of the other languages.

The XLM-R model uses the CNN classification head; we train it for 10
epochs with a learning rate of $2\times10^{-5}$ for the transformer, a
learning rate of $5\times10^{-5}$ for the classification head, and the
cross-entropy loss. The biLSTM has a hidden size of 100, and we train it
for 50 epochs, with a learning rate of $10^{-4}$, and the weighted
cross-entropy loss). For the LASER+MLP model, we use a MLP with layers
of sizes 128, 64 and 3, separated by LeakyReLUs with a slope of 0.01, a
learning rate of $10^{-3}$ and the weighted cross-entropy loss. The
embedding tuning for MUSE embeddings [@conneau2017word] is disabled, to
avoid dissociation of the train and test embedding spaces. For the joint
models, the classification head of the transformer has 64 outputs
instead of the usual 3, and if applicable are concatenated to the LASER
and/or Hurtlex representations (the LASER embeddings first go through a
linear layer with 64 outputs), before going through a LeakyReLU with
slope 0.1, and a final linear layer with 3 outputs. For the joint
models, the learning rate of the transformer is $2\times10^{-5}$ and the
learning rate of all other layers is $5\times10^{-5}$. All models use
the Adam optimiser [@kingma2014adam].

The MUSE embeddings do not work very well, with significantly lower
scores than the other architectures, and for German test data it is as
bad as a random prediction (see table
[\[tab:naive\]](#tab:naive){reference-type="ref"
reference="tab:naive"}). Despite having a simpler architecture, the
LASER+MLP model is competitive with the transformer models, surpassing
XLM-R in multiple cases. It seems that XLM-R encounters some
difficulties when there are German data in the training sets; in both
cases where we only train on German, the score on XLM-R is significantly
lower than with the other languages for training. This large difference
is not present with the other two models, nor in the opposite direction
with German as the test language. Finally, when comparing the basic
XLM-R model with the joint models, we see that the joint models are
typically better. In all cases, there is at least one joint model
performing better than the base architecture. The better model is not
the same for all languages, but overall we see a clear benefit in using
the joint architecture with additional representations.

#### Translation-based models {#translation-based-models}

The second approach to zero-shot classification is to use translations
and monolingual models. For the translations into English, French and
German, we use RoBERTa [@liu2019roberta], FlauBERT [@le2019flaubert] and
german-BERT [^41], respectively (they are the best *monolingual* models
in table
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"}, we did not use XLM-R for
German to avoid mixing conceptually different models). We also test the
*joint transformers* architecture (section
[5.2.1](#joint_transformers){reference-type="ref"
reference="joint_transformers"}). For this model, in the experiments
with three languages, we could not use the original transformer models
due to memory limitations. We used distilRoBERTa, FlauBERT-small and
distilBERT-german instead of the original models, [^42] which puts these
models at a small disadvantage. We compare all translation-based models
with the best purely multilingual models taken from table
[\[tab:zero_shot_dachs\]](#tab:zero_shot_dachs){reference-type="ref"
reference="tab:zero_shot_dachs"}, and show all results in table
[\[tab:translate_dachs\]](#tab:translate_dachs){reference-type="ref"
reference="tab:translate_dachs"}.

Despite some potential noise and loss of meaning due to the
translations, the models that use translations perform better than
purely multilingual models in multiple cases. There are some exceptions
when there are German training data, but in those cases there is usually
a model using translation not too far behind. Interestingly, the
language chosen for the translation has some importance; when we want to
classify English data, the best choice is to train an English model with
English translations, and when we want to classify French data, the best
choice is to use a French model and French translations. This pattern
does not apply to the experiments with German test data, probably
because german-BERT is a simpler model than RoBERTa and FlauBERT, with
less training data used in the original pretraining (for the same reason
that XLM-R was better than german-BERT on German data in table
[\[tab:results_transformer_model\]](#tab:results_transformer_model){reference-type="ref"
reference="tab:results_transformer_model"}). The results with the *joint
transformers* architecture are not conclusive, with no significant gain
in performance especially when considering the increased cost in
resources, but it might be caused by the simplified transformers used
due to technical limitations.

**Joint LASER**: We report in table
[\[tab:joint_laser\]](#tab:joint_laser){reference-type="ref"
reference="tab:joint_laser"} results for the second joint learning
approach, *joint LASER* (section
[5.2.2](#joint_laser){reference-type="ref" reference="joint_laser"}),
with one LASER+MLP block used for each language in the data and
multidirectional translations. We compare those with the simple
multilingual LASER+MLP model from table
[\[tab:zero_shot_dachs\]](#tab:zero_shot_dachs){reference-type="ref"
reference="tab:zero_shot_dachs"}. Here, the LASER embeddings from each
language each go trough a linear layer with 128 outputs, a LeakyReLU
with slope 0.01, and a dropout layer with probability 0.1, before being
concatenated and going through a linear layer with 64 outputs, a
LeakyReLU with slope 0.01 and a final linear layer with 3 outputs, one
for each class. We use a learning rate of $10^{-3}$ and the weighted
cross-entropy loss.

For all cases, we have a non-negligible improvement over the original
LASER model, while keeping a relatively simple model (still less complex
than the transformer models). We add the requirement for data
translation, and the complexity of the model doubles, but we think it is
a good trade-off when looking at the improvements we have in some cases.
The model only needs to learn the parameters for a few linear layers,
which can be done rapidly on a GPU, although we still need to generate
the LASER embeddings, which can take some time. For most cases, the
transformer models in table
[\[tab:translate_dachs\]](#tab:translate_dachs){reference-type="ref"
reference="tab:translate_dachs"} still perform better, but we are able
to get quite close in multiple scenarios, with significantly lower
training time.

**Joint transformer + LASER + Hurtlex**: Finally, table
[\[tab:joint_transformer_laser\]](#tab:joint_transformer_laser){reference-type="ref"
reference="tab:joint_transformer_laser"} shows results for the third
type of joint models, where we join the outputs of a transformer model
with a LASER+MLP block and/or the Hurtlex [@hurtlex] representation
(sections [5.2.3](#joint_transformer_laser){reference-type="ref"
reference="joint_transformer_laser"} and
[5.2.4](#hurtlex){reference-type="ref" reference="hurtlex"}). For both
RoBERTa and FlauBERT, with English or French translations, we show
results for the transformer by itself, the transformer with LASER
embeddings, the transformer with the Hurtlex representation and the
transformer with both LASER embeddings and the Hurtlex representation.
We do not perform this experiment with german-BERT, [^43] as we saw in
table
[\[tab:translate_dachs\]](#tab:translate_dachs){reference-type="ref"
reference="tab:translate_dachs"} that it is always worse than the
English and French models.

Using additional representations jointly with the transformer let us
improve all results over the base models, but the best variant depends
on the languages used. On average, the Hurtlex representation seems to
be slightly more beneficial to the score than the LASER embeddings, but
as usual there is a lot of variance with some models, making confident
conclusions difficult. Joining both the Hurtlex and LASER
representations is usually not useful.

We summarise all *zero-shot* results in table
[\[tab:zero_shot_best\]](#tab:zero_shot_best){reference-type="ref"
reference="tab:zero_shot_best"}, reporting the best results for purely
multilingual models and for models with translations. We consider those
two approaches different enough to warrant separate reporting.
Translation-based models require more processing before training the
model, and most importantly require constant translation of new data
while using them in practice. To support multiple languages, multiple
Opus-MT [@TiedemannThottingal:EAMT2020] translation models will be
required per language, which can get costly very quickly. We do not have
this problem with the purely multilingual model.

In all cases, the models that utilise translations perform better than
the multilingual models. The best models are also usually joint models;
the use of diverse representations of the input is clearly useful. It is
however important to remember that this is an ideal case, where the data
used for zero-shot evaluation is sometimes in the same language as the
monolingual transformer, and that translations to English and French are
probably better than average due to the common use of these languages.

We conclude this section with detailed results for the best models from
table [\[tab:zero_shot_best\]](#tab:zero_shot_best){reference-type="ref"
reference="tab:zero_shot_best"}. We show those results in table
[\[tab:zero_shot_best_detailed\]](#tab:zero_shot_best_detailed){reference-type="ref"
reference="tab:zero_shot_best_detailed"}, alongside the detailed results
of the monolingual models on each language (using models from table
[\[tab:results_classification_head\]](#tab:results_classification_head){reference-type="ref"
reference="tab:results_classification_head"}). There is still a large
gap between the scores obtained on the monolingual classification, and
what can be achieved with zero-shot classification. The recall on `hate`
and `offensive` is particularly low, with the largest difference in
French. On Dachs DE, however, we get closer results; since the dataset
itself is not very good, it is less problematic to use other data for
training. With French and English, the domain gap has more impact.

### Multilingual classification {#ml_multilingual}

In this section, the goal is to develop a unique model compatible with
multiple languages that delivers comparable performance to individual
monolingual models, while also performing well on other unknown
languages. We train multiple models on English, French, and German data
and evaluate them on these three languages but also additional languages
with no training data.

For the training languages, we keep half of the datasets for each
language that are in our opinion the best. For English we keep and merge
Jigsaw and Dachs EN, as both the Davidson and Founta datasets have
problems with the distinction between `hate` and `offensive`. We keep
Dachs FR for French and Germeval for German. We create a balanced
validation set, consisting of one third English data, one third French
data and one third German Data, that we use for early stopping, but the
evaluation on the test sets is still done for each language
independently.

We use the same models as in zero-shot classification (section
[5.3.2](#ml_zeroshot){reference-type="ref" reference="ml_zeroshot"}):
the multilingual XLM-R and LASER+MLP models (section
[5.1.1](#multilingual_models){reference-type="ref"
reference="multilingual_models"}), the monolingual models using
translations (Translate EN + RoBERTa, Translate FR + FlauBERT, section
[5.1.2](#translation_models){reference-type="ref"
reference="translation_models"}) and the joint variants with the LASER
and/or Hurtlex representations joined to the transformers (section
[5.2.3](#joint_transformer_laser){reference-type="ref"
reference="joint_transformer_laser"}). We also show the *joint
transformers* model (section
[5.2.1](#joint_transformers){reference-type="ref"
reference="joint_transformers"}) and *joint LASER* model (section
[5.2.2](#joint_laser){reference-type="ref" reference="joint_laser"}),
both with three modules (one for each training language). In table
[\[tab:multilingual_comparison\]](#tab:multilingual_comparison){reference-type="ref"
reference="tab:multilingual_comparison"} we show the macro-F1 score on
the three training languages, but also on additional datasets in unseen
languages (Dachs GR, Dachs ES and Indonesian, see section
[3.1](#data){reference-type="ref" reference="data"}).

For the results on the training languages, we observe similar behaviours
as in zero-shot classification (section
[5.3.2](#ml_zeroshot){reference-type="ref" reference="ml_zeroshot"}); in
English and French the translation-based methods are overall better, and
the chosen translation language is important. In German, the XLM-R model
is this time better than any of the translation models. For both
categories of models, the joint variants are generally better than the
base models, although what variant is the best depends on the language.
On average over the training languages, the XLM-R models are better than
the translation-based methods (mostly due to the German scores), with a
slight advantage for the model joined with the LASER and Hurtlex
representations. The LASER+MLP models are clearly worse than the
transformer models, but the advantage of the *joint LASER* model is
still present. Finally, the scores on some of the joint models are
equivalent to the best individual monolingual models, showing the
usefulness of the joint methods that could also be applied to a
monolingual context. However, there is not one single multilingual model
reaching maximal performance on the three languages, so for optimal
results on these languages one would still need to use multiple
individual models.

For the additional languages evaluated in a zero-shot context, it is
quite hard to formulate a definitive conclusion of which model is the
best. On Indonesian for instance, the single LASER+MLP model is the only
one that reaches results better than a random prediction. For Spanish,
the *joint transformers* is by far the best model, with not a lot of
difference between the purely multilingual and translations-based
models. In Greek the translation models are usually better. For the
three languages, the additional LASER and/or Hurtlex representation are
often useful, but overall the scores are very low, especially compared
to English and French, making these models not really usable in
practice.

#### Binary classification

Since some of the misclassifications come from a confusion between
`hate` and `offensive`, we look at the binary classification scores
(`toxic` or `non-toxic`) to get a better grasp of the performance of the
models. We use the exact same models, but regroup the `hate` and
`offensive` predictions into a unique `toxic` class, which now also let
us evaluate three new Jigsaw datasets (Spanish, Italian and Turkish,
presented in section [3.1](#data){reference-type="ref"
reference="data"}). We show the macro-F1 scores in table
[\[tab:multilingual_comparison_binary\]](#tab:multilingual_comparison_binary){reference-type="ref"
reference="tab:multilingual_comparison_binary"}.

We observe similar patterns to those of table
[\[tab:multilingual_comparison\]](#tab:multilingual_comparison){reference-type="ref"
reference="tab:multilingual_comparison"}; LASER+MLP is usually the worst
model, and for multiple languages we can improve the score of the
transformer models by joining additional representations. Altogether,
the best approach seems to be to translate all data into English, and
then use a RoBERTa model jointly with LASER embeddings. This gives us
the second-best average in table
[\[tab:multilingual_comparison\]](#tab:multilingual_comparison){reference-type="ref"
reference="tab:multilingual_comparison"} and best average in table
[\[tab:multilingual_comparison_binary\]](#tab:multilingual_comparison_binary){reference-type="ref"
reference="tab:multilingual_comparison_binary"}. For a purely
multilingual model, XLM-R joined with LASER embeddings is the best
option and is actually very close to the performance of the translation
models on the binary classification task, and most crucially does not
require any translations. The final choice will depend on the languages
that need to be supported, the exact classification task (two or three
classes), and the available resources.

This time the Indonesian data score better than the Spanish data from
Dachs with most models, a good indication that many misclassifications
came from a confusion between `hate` and `offensive`. Perhaps most
importantly, the impact of the choice of datasets is very clear; Dachs
ES and Jigsaw ES, while both in the same language, have vastly different
scores. It is hard to know exactly the cause of this difference; it can
be a problem with the quality of annotations in Dachs ES or a larger
domain gap from Dachs ES to the training data than from Jigsaw ES. This
shows that it is extremely challenging to predict the performance of our
models on general data, as all of the scores are more dependant on the
dataset chosen rather than the language or model selected. Nevertheless,
since we evaluated a lot of datasets, some patterns and conclusions
should still apply relatively well to other datasets. With the three
datasets from Jigsaw we also observe higher variations in the scores;
this happens because we have less data for them, therefore taking a
random test set from them will result in more variance, but we still
think that the differences between the values are large enough to draw
some conclusions.

Discussion
==========

In this chapter, we study in greater details a few aspects of the
toxicity detection task to highlight some weaknesses of the models
presented in the previous two chapters. We also present a few additional
relevant experiments.

Error Analysis {#error_analysis}
--------------

We consider here a few incorrectly classified messages to highlight
factors that lead to erroneous predictions. There are two primary
causes: there can be problems with the datasets and problems with the
models. We mostly look at errors caused by the datasets, as they are
easier to uncover and understand.

The major problem with some of the available datasets is the
inconsistency between the annotations of various sentences. Two very
similar sentences can sometimes be found in multiple classes. As a first
example, we have the following sentences:

> *geta life f\*ggot* $\rightarrow$ `hate`\
> *stick to hockey f\*ggot. you're finished* $\rightarrow$ `offensive`

The term *f\*ggot* does not seem to be used as an attack against
homosexual people here and should be in the `offensive` class in both
cases according to the definitions presented in section
[1.2](#problem_statement){reference-type="ref"
reference="problem_statement"}. It can be argued about whether or not
the use of this term should be considered `hate` or `offensive` in this
context, but it should be the same class for both sentences here since
the term is used in the exact same way. Similarly, we consider the
following sentences:

> *what a f\*cking ret\*rd* $\rightarrow$ `offensive`\
> *lmao madonnas music sucks? yeah you're f\*cking stupid if you think
> gaga is better. ret\*rd* $\rightarrow$ `hate`

In both cases the term *ret\*rd* is used as a general insult and does
not seem to target people with mental disabilities, but they were
assigned to different labels. In more extreme cases, sentences that were
extremely similar also had inconsistent annotations:

> *\@latimes Go back to your own country* $\rightarrow$ `hate`\
> *\@FoxNews Nope. Go back to your own country* $\rightarrow$ `none`\
> *\@Newsweek Go back to your own country. \#BuildTheWallNow*
> $\rightarrow$ `none`

All three messages come from the Dachs dataset, where each message only
has one annotator, meaning that personal opinions and bias will have a
huge effect in the annotations. The models will then be given
contradictory information, hurting their performance.

We were also able to find some annotations that in our opinion were
simply incorrect, for instance:

> *I hate aubergines* $\rightarrow$ `hate`\
> *Just need to make it through the next weeks and then I can relax and
> by relax I mean start parking to move\...* $\rightarrow$ `hate`

Manual research of these tweets on Twitter showed no special context
that might indicate hate speech. Both messages were classified as `none`
by our model.

All of those errors and inconsistencies in the annotations makes it hard
to precisely know the performance of our models, and if
misclassifications are caused by problems with the annotations or the
model. The amount of such problems depends on the dataset, as the
annotation methodologies and number of annotators are not always the
same (see section [3.1](#data){reference-type="ref" reference="data"}),
but it is impossible to know the full extent of this problem without a
time-consuming review of all annotations.

The second type of misclassification happens when there is simply not
enough training data to learn the toxicity of some content. For
instance,

> *burn in h3ll s8n*

was classified as `none` by our model, simply because it never observed
the words *h3ll* and *s8n* written this way before, therefore cannot
infer anything from them. Similarly, when we have sentences with words
that are rarely present in the training set, it is challenging to infer
the correct class. For instance, the word *slits* in the sentence

> *i'm surprised you could read it with slits for eyes*

only appear three times in the Davidson dataset, above example included,
and it is associated twice with the `none` class. More data are required
to make the co-occurrence of *slits* and *eyes* recognisable as racism
against Asian people.

Even with enough data to learn all currently hateful and offensive
expressions, there would still be problems with unknown expressions. New
words and expressions are sometimes created, especially on the internet,
and with them novel ways to be toxic. Moreover, the meanings of some
terms change over time and some become offensive, such as the word
*Boomer* now considered as ageist and offensive by some people due to
the rise in popularity of the expression \"*OK Boomer*\" used to mock
some behaviours of people from the Baby Boomer generation. The same is
true with *Karen*, nothing more than a name a few years ago but now a
mocking term for a certain type of women perceived as entitled. These
words, almost not present in our datasets, are now very common in online
communities, therefore the models trained here will probably not
properly detect this type of offensive content. The opposite can also
happen, with terms like *queer* or *geek* that used to be pejorative now
accepted by the communities they were once attacking [^44].

Finally, it is extremely hard to define offensive and hateful words. If
a word is considered offensive by only 1% of the population, should it
be universally considered offensive? There is ordinarily no official
authority that defines offensive words, so any classifier system that
relies on manually annotated data will depend on the personal opinion of
the annotators and authors, and the context in which those messages were
written, thus it is extremely important for the data used to represent
the general opinion of the population the model will be applied to.

Domain gap
----------

The first problem that contributes to relatively low scores when
performing zero-shot classification remains the insufficient quality of
some of the datasets; if we do not have a good score on a dataset when
training it directly, we will not be capable of creating a decent
zero-shot model with it. Not a lot can be done to fix this, except
giving particular attention to dataset selection.

In the case of cross-lingual classification, there is on top of that the
problem of the language gap. Some expressions might be considered rude
in one language, but when translated literally will lose the original
sense and any negative connotation. In practice, the quality of Opus-MT
translations [@TiedemannThottingal:EAMT2020] seems reasonably good (as
shown in table
[\[tab:translations\]](#tab:translations){reference-type="ref"
reference="tab:translations"}) but there are always exceptions that
might lower the performance of the models.

Most importantly, as shown in section
[3.1.2](#data_analysis){reference-type="ref" reference="data_analysis"},
the themes present in the different datasets are fairly different, which
causes a non-negligible domain gap. In some datasets we have a lot of
racism, in others it is more general hate (e.g. wish for harm). As an
experiment, we tried zero-shot classification on English data only using
a simple RoBERTa [@liu2019roberta] with the default CLS classification
head (see section [4.2.1](#classification_heads){reference-type="ref"
reference="classification_heads"}). We show results in table
[\[tab:zeroshot_monolingual\]](#tab:zeroshot_monolingual){reference-type="ref"
reference="tab:zeroshot_monolingual"}.

It is clear that the domain gap between the datasets exerts a
considerable influence on the scores. Each dataset has an individual
score that is relatively high, a sign that the dataset itself is quite
good and has sufficient data. However, with zero-shot classification the
results are clearly lower, and it is extremely hard to utilise knowledge
acquired in one dataset to classify other data. This is also the main
cause behind the lack of good results with dataset merging (section
[4.2.2](#merging){reference-type="ref" reference="merging"}).

Even with similar types and distributions of `hate` and `offensive`
content in all datasets, there is still a difference in subjects and
targets across languages. In the English (American) datasets, racism
might typically be directed towards the Mexican and African-American
populations, while in the French and German datasets racism might
instead focus on North-African and Middle-Eastern countries. It will be
challenging for a model trained uniquely on French data to learn the
concept of hate against Mexicans. There are also subjects that despite
being present in multiple datasets/regions do not have the same
perceived offensiveness; for instance attacks against homosexuals will
be looked at completely differently in Sweden or Iran. If we then mix
the two sources, a model will have to employ contradictory data. This
gap is present in all societal domains, from politics to religion and
economy, which makes it extremely challenging to successfully combine
datasets gathered in different countries.

All these examples fall into the domain gap category, and we think that
this is the most critical problem and limitation to any model that
regroups different datasets (in particular zero-shot classification
models). The language gap is mostly crossed by the multilingual models
or the translations, but there is not a lot that can be done about the
domain gap. If we were to test these models on datasets in multiple
languages, but from a similar geographical region and with the same
collection methodology, we think that the scores might be better; we
could for instance gather French, German and Italian comments on the
respective *20 Minutes* (a Swiss newspaper) comment sections, in which
case the themes of each dataset should be quite close.

To illustrate the impacts of the language gap and domain gap, we
performed a few experiments. Supposing a dataset $A$ in language $L_A$
and a dataset $B$ in language $L_B$, we train a XLM-R+LASER multilingual
model (section [5.6](#fig:joint_transformer_laser){reference-type="ref"
reference="fig:joint_transformer_laser"}) in four different ways, each
of them corresponding to a different combination of language and domain
gap, and evaluate it on dataset A:

-   train the model on dataset $A$;

-   train the model on dataset $A$ translated to language $L_B$:
    language gap;

-   train the model on dataset $B$ translated to language $L_A$: domain
    gap;

-   train the model on dataset $B$: domain gap and language gap.

We report in table
[\[tab:domain_vs_language_gap\]](#tab:domain_vs_language_gap){reference-type="ref"
reference="tab:domain_vs_language_gap"} a few results using the Dachs
EN, Dachs FR and Germeval datasets.

Two conclusions come from these results. We first see that even though
we use a multilingual model, translations have some influence. In
particular when the training dataset is different from the test dataset,
it is useful to translate the training data into the test language (see
the last two columns of table
[\[tab:domain_vs_language_gap\]](#tab:domain_vs_language_gap){reference-type="ref"
reference="tab:domain_vs_language_gap"}). It is better to have no
language gap, even if this means adding some noise during translations).

Moreover, as stated previously, the domain gap is clearly more
problematic than the language gap. The same pattern applies for all
examples, with a drop of performance due to the domain gap at least
twice as important as the drop of performance due to the language gap.
It shows that *cross-domain* classification is actually harder than
*cross-lingual* classification, and meticulous care should be given to
gather a good selection of compatible datasets. This also means that in
some cases training data from other languages might be better than
training data in the target language and should not be excluded.

Tuning the models for a target application {#fhate}
------------------------------------------

Until now, we have used the macro-F1 score to measure the general
performance of our models, for the reasons cited in section
[3.2](#metrics){reference-type="ref" reference="metrics"}. It accords
the same importance to all classes, and to the recall and precision of
those classes, but depending on the nature of the problem we might want
to give more importance to the recall or precision of some classes. This
is often the case with hate speech detection, where it is usually more
important to have a high recall on the `hate` class rather than a high
precision.

One solution would be to measure only the recall on `hate`, but then it
is not possible to detect extreme cases where the model does not learn
anything (see in section [3.2](#metrics){reference-type="ref"
reference="metrics"}, if we have a model that always predict `hate` we
will have a score of 100%). We wanted a metric that could offer a
general overview of the performance of a model while giving more
importance to the recall on `hate` and `offensive`, and that can still
detect and avoid extreme cases. We define a custom metric that we call
*F-hate*, which can be used to optimise a given model for a specific
application, with a certain degree of severity required to block toxic
content. This is a variant of the F1-score, with weights given to the
precision and recall. We define

$$\begin{aligned}
   \textrm{F-hate}_\textit{none} &= (1+\beta^2)*\frac{\textrm{precision}_\textit{none}*\textrm{recall}_\textit{none}}{\textrm{precision}_\textit{none} + \beta^2*\textrm{recall}_\textit{none}}\\[2pt]
    \textrm{F-hate}_\textit{offensive} &= (1+\beta^2)*\frac{\textrm{precision}_\textit{offensive}*\textrm{recall}_\textit{offensive}}{\beta^2*\textrm{precision}_\textit{offensive} + \textrm{recall}_\textit{offensive}}\\[2pt]
    \textrm{F-hate}_\textit{hate} &= (1+\beta^2)*\frac{\textrm{precision}_\textit{hate}*\textrm{recall}_\textit{hate}}{\beta^2*\textrm{precision}_\textit{hate} + \textrm{recall}_\textit{hate}}\end{aligned}$$

where, for $\beta > 1$, we give on the `none` class more importance to
the precision than to the recall (when we mark a message as `none` we
want to be sure that it is indeed `none`), and on the `offensive` and
`hate` classes we give more importance to the recall (we do not want to
miss toxic content). From this we can derive the global F-hate metric,
defined as

$$\textrm{F-hate} = \frac{\textrm{F-hate}_\textit{hate} + \textrm{F-hate}_\textit{offensive} + \textrm{F-hate}_\textit{none}}{3}$$

If we take for instance $\beta = 2$, it means that the recall on `hate`
is twice as important as the precision on `hate`, the recall on
`offensive` is twice as important as the precision on `offensive`, and
the precision on `none` is twice as important as the recall on `none`.
This would be a system where we do not want to let toxic messages pass,
and we prefer to have to manually white-list acceptable content.
Increasing $\beta$ will make this system more strict. If we take
$\beta = 1$, this corresponds to the macro-F1 score, where recall and
precision have the same importance.

Choosing $\beta > 1$ is usually the best choice since toxic comment
classification cannot be considered a symmetrical problem. It naturally
depends on the final application, but most of the time one would rather
have a system where no toxic content is missed than a system were toxic
content easily gets through, even if there are a few false positives.

With $\beta < 1$, it would create a system that tries not to block any
normal content, even if it means letting toxic content through more
often. This case is probably less common, but the results are still
interesting and might be applicable to some specific applications (for
instance to avoid being accused of censoring free-speech it is necessary
to let all non-toxic content through). More generally, this method of
creating a custom metric based on the macro-F1 can be used on other
classification tasks, and conclusions drawn in this section should be
applicable to other contexts.

With this metric defined we can now train models, but instead of using
the macro-F1 as a metric for the early stopping we use the F-hate metric
with $\beta > 1$, which should make the model stop training at a point
with a high recall on `hate` and `offensive`. We look at results for the
Dachs EN dataset in table
[\[tab:metric_change_en\]](#tab:metric_change_en){reference-type="ref"
reference="tab:metric_change_en"}, with a simple transformer model,
RoBERTa [@liu2019roberta] with the CNN classification head (section
[4.2.1.3](#cnn_head){reference-type="ref" reference="cnn_head"}), once
using the macro-F1 score as the early stopping metric, and once using
the F-hate metric with $\beta=2$. We also report results using the
weighted cross-entropy loss (WCEL) instead of the cross-entropy loss
(CEL) (see section [4.3.1](#loss){reference-type="ref"
reference="loss"}), as it causes a similar effect to the use of the
F-hate.

With the original version (macro-F1 and CEL), recall and precision are
relatively similar. The use of the F-hate metric increases the recall on
`hate` and `offensive` by 6-8%, and the precision on `none` increases by
3%. To compensate, the three other metrics, recall on `none` and
precision on `hate` and `offensive` decrease. Going from the unweighted
CEL with the macro-F1 to the WCEL with the macro-F1 has a similar effect
but to a lesser extend, while using the WCEL with the F-hate increases
this effect even more. If we perform the same experiment with Dachs FR,
we obtain very similar results (see table
[\[tab:metric_change_fr\]](#tab:metric_change_fr){reference-type="ref"
reference="tab:metric_change_fr"}). To accord more importance to the
recall of the toxic classes, we should use the WCEL with the F-hate
metric. The CEL with the macro-F1 is at the opposite, with the lowest
recall on toxic classes. The two other options (CEL with F-hate and WCEL
with macro-F1) once again find themselves in between the two extremes.

If the recall on `hate` was used a the sole metric, we would see
something similar to the score obtained with the WCEL and the F-hate
metric, with a recall on `hate` significantly higher than the precision
on `hate`. However, this difference would be larger, and we might lose a
lot of precision for only an insignificant improvement in recall. For
instance, a model with $recall_{hate}=85\%$ and $precision_{hate}=40\%$
would be chosen over a model with $recall_{hate}=84\%$ and
$precision_{hate}=70\%$. This is problematic, as we do not want to
sacrifice too much of any metric. We also observed some cases where the
initial prediction of the model was that all content was `hate`, due to
the random initialisation of the network weights, and if we tracked only
the recall on `hate` the training would quickly stop as we would already
have reached a maximal value of the metric.

It is straightforward to change the *F-hate* score with the desired
ratio to give more importance to some classes. This choice is mostly
arbitrarily, and one would likely need to test a few values of $\beta$
to find a good compromise. We show in figure
[6.1](#fig:beta_fhate){reference-type="ref" reference="fig:beta_fhate"}
an experiment on Dachs EN (RoBERTa with the CNN classification head)
with the CEL, where we change the value of $\beta$ in the F-hate
metrics. We plot for $\beta  \in \{1/3, 1/2, 2/3, 1, 3/2, 2, 3\}$ the
metrics computed from the average of three runs over random test sets.
We also report the standard error for each value, computed as
$\frac{\sigma}{\sqrt{3}}$, where $\sigma$ is the standard deviation.

![[\[fig:beta_fhate\]]{#fig:beta_fhate label="fig:beta_fhate"}Evolution
of the recall, precision and macro metrics on Dachs EN, depending on the
$\beta$ value in the F-hate metric, using RoBERTa with a CNN
classification head.](fhate){#fig:beta_fhate width="70%"}

When we increase the $\beta$ coefficient, the recall on `hate` and
`offensive` and the precision on `none` increase, while the precision on
`hate` and `offensive` and the recall on `none` decrease, and the
opposite happens when decreasing the $\beta$ coefficient. This is a
symmetrical problem, with $\beta = 2$ having the opposite effect of
$\beta = 1/2$. When $\beta = 1$, the macro-F1 is the same as the
*F-hate*. The further from $1$ this value is, the lower the macro-F1
gets, and the higher the *F-hate* gets.

The previous results show that the *F-hate* is useful to tune the model
to a desired level of severity in a monolingual context, but we still
need to see if the same conclusions apply to a multilingual context. We
experimented with XLM-R [@conneau2019unsupervised] with the CNN head
(section [5.1.1.3](#ml_transformer){reference-type="ref"
reference="ml_transformer"}) joined with the LASER+MLP block (section
[5.2.3](#joint_transformer_laser){reference-type="ref"
reference="joint_transformer_laser"}). We trained the model on Jigsaw
and Dachs FR, and evaluated it on Germeval, Dachs ES, Dachs GR and the
Indonesian dataset. We trained two versions on this model, one with the
macro-F1 used as the early stopping and once with the F-hate with
$\beta=2$ (twice as much importance to recall on `hate` and `offensive`
and to precision on `none`).

Going from the macro-F1 to the F-hate, we were able to observe the
expected effect on Jigsaw and Dachs FR, the training languages. The
pattern was clear, and in both cases the recall on `hate` and
`offensive` increased and the precision on `none` also increased.
However, with the other datasets that were evaluated in the zero-shot
context we did not consistently observe this effect. For some classes
and datasets it was present, but for others not, and we could not
identify any pattern that might explain the results. Since the F-hate
does not actually change the architecture but only stops the training at
a point with optimal scores on the validations dataset, we would need
the datasets in the zero-shot languages to be extremely close to the
early stopping dataset to allow for a transfer of this effect. In
practice it is often not the case, and the F-hate does not properly
apply to zero-shot classification.

Multilingual model or multiple monolingual models
-------------------------------------------------

As seen in the table
[\[tab:multilingual_comparison\]](#tab:multilingual_comparison){reference-type="ref"
reference="tab:multilingual_comparison"}, when multiple languages have
available training data, there is not one unique model that performs
optimally on all languages, and for each language a different
architecture would be better suited. From these observations, we propose
in figure [6.2](#fig:ml_pipeline){reference-type="ref"
reference="fig:ml_pipeline"} a pipeline that could be beneficial for
multilingual classification.

![[\[fig:ml_pipeline\]]{#fig:ml_pipeline label="fig:ml_pipeline"}On the
left, the original pipeline with a single model to classify data from
any language. One the right, the advanced pipeline where we suppose
French, English and German training data. Data in any other language
falls in the *other* category.](ml_pipeline){#fig:ml_pipeline
width="80%"}

Given a set $S$ of languages for which we have some training data, we
create $|S|$ individual monolingual classifiers, one for each language
in $S$, and a language identification module. One multilingual
classifier is also trained using any of the architectures presented in
chapter [5](#cross_lingual){reference-type="ref"
reference="cross_lingual"}. When presented with a new sentence to
classify, if it is written in one of the languages in $S$ the
architecture sends the sentence to the associated monolingual
classifier, otherwise it goes to the multilingual classifier. This gives
us scores on known languages as good as possible using monolingual
models, while still being able to classify any other language. There is
a trade-off between performance and cost; using a single multilingual
model for all languages will require less memory but will have lower
scores; whereas for a more advanced pipeline more memory will be
required but the scores will be significantly better.

The final choice of architecture also comes down to the data the model
will be applied to. If for instance most of the expected data is in
English, with only a few messages in French or German, a model with good
performance in English (equivalent to a monolingual English model) but
slightly worse on other languages can be used. On average, the
performance will still be better than with a purely multilingual model,
and adding monolingual models for French and German might not be worth
the added cost.

Model selection
---------------

Due to the chosen training methodology and inherent variance when
training transformers, all of our results have quite a lot of variance,
which often makes the comparison between multiple models complex. In
some cases it is still possible to say that an architecture is better
than another, but we expect that one might get different results when
trying to reproduce some of our experiments. Overall, we think we were
able to demonstrate the potential of advanced classification heads, and
that for cross-lingual classification the joint models are useful and
translating data into a single language to use monolingual models can
work, depending on the targeted languages. The largest variable here
remains the choice of datasets. Depending on the available training data
and the desired target language and evaluation dataset, some
architectures will be more adapted than others.

We therefore cannot confidently select a best architecture. For
monolingual models, we can confidently say that RoBERTa
[@liu2019roberta], FlauBERT [@le2019flaubert] and XLM-RoBERTa
[@conneau2019unsupervised] are the best transformers for our English,
French and German datasets, but we are less confident with the other
architectural changes. For cross-lingual classification, with our
datasets and with a specific subset of languages, we found that the
joint RoBERTa+LASER model with translations (section
[5.2.3](#joint_transformer_laser){reference-type="ref"
reference="joint_transformer_laser"}) was on average better (table
[\[tab:multilingual_comparison\]](#tab:multilingual_comparison){reference-type="ref"
reference="tab:multilingual_comparison"} shows the usefulness of
translations and joint models in at least some cases). The XLM-R+LASER
model was also competitive, without requiring translations, although the
translation-based models are still overall better than the purely
multilingual models with our datasets.

When tasked with developing a multilingual model based on the present
work, one should evaluate the various models for all target languages to
determine the best choice (and this choice should also depend on the
expected proportion of data from each language). Ideally, this
evaluation data should not originate from the datasets mentioned in this
project but should be gathered specifically for the task at hand.
Additionally, the different options presented in the first chapter (in
section [4.2](#improvements){reference-type="ref"
reference="improvements"}) should also be evaluated for cross-lingual
models.

Finally, this choice will also come down to a compromise between
performance and cost. Some of the evaluated options and architectures
require significantly more computation power, and a consequent amount of
time would need to be dedicated to finding the optimal model. This
investment might not be worth for potentially only a few percents of
improvements in the final score compared to a simple base transformer
model.

Conclusions and Future work
===========================

Conclusions
-----------

In this thesis we tackled monolingual and cross-lingual classification.
In both cases, we evaluated existing methods on multiple available
datasets, and then proposed various approaches to improve the
performance of these methods.

With monolingual classification, we showed that the recently introduced
transformer-based models clearly outperform classical methods, and that
we can improve results slightly by using different classification heads.
We also showed the difficulty of combining multiple datasets; using a
multitask approach is usually better than naively merging datasets, but
the domain gap between them is usually too wide to lead to any
improvement. Other methods were less successful, but the high standard
deviations of results make it hard to make confident conclusions so we
cannot exclude the possibility that these methods might work with other
datasets.

With cross-lingual classification, we used data translation to introduce
multiple joint-learning architectures. We combined transformer-based
models with LASER embeddings and the Hurtlex representation, leading to
better results than the transformer models by themselves. These
approaches were evaluated on multiple languages, including languages for
which we had no training data. The translation-based models with joint
learning usually yield slightly better results than purely multilingual
models, but come with the added translation cost. Regrettably, we were
not able to create one single multilingual model that performed as good
on English, French and German as the monolingual models trained on each
language individually, but we still reached relatively close results
considering the difficulties introduced when merging datasets.

We also proposed a custom metric, the *F-hate*, that can be adapted and
used to give more importance to the precision or the recall on some
classes. Using this metric for early stopping, it is possible to create
models with varying levels of severity in their classification, and it
is therefore relatively easy to adapt them to specific target
applications with various levels of severity desired. We were not able
to transfer this method to a zero-shot context, in part due to the
domain gap.

The domain gap seems to be the largest challenge that needs to be
overcome before being able to create performant multilingual models.
This effect is already important when comparing multiple datasets in the
same language and is sometimes even more present across languages due to
large differences in societal and cultural issues between populations.
We showed that the domain gap is in fact a larger problem than the
language gap to implement cross-lingual models. The datasets chosen for
any multilingual problem need to most importantly have a low domain gap
between them if we want decent zero-shot classification scores, and the
difference of languages is not as problematic.

Future Work
-----------

There are still a lot of problems with the methods presented in this
thesis that lead to scores that are often not good enough for a
sensitive task like hate speech detection. We list here some leads that
we did not have time to investigate but might improve the models.

**Multitask-learning for cross-lingual data**: In chapter
[4](#monolingual){reference-type="ref" reference="monolingual"}, we
presented two approaches to data combination: data merging and multitask
learning. However, in the second part on cross-lingual classification
(chapter [5](#cross_lingual){reference-type="ref"
reference="cross_lingual"}), we only used data merging to combine data
from multiple languages. We think there is some potential in combining
data using multitask learning, having for instance specific
classification heads for each language in the training data, and one
classification head used for the other unknown languages, which could
partially alleviate the domain gap.

**LASER fine-tuning**: With the transformer models (section
[4.1.4](#transformer_models){reference-type="ref"
reference="transformer_models"}), we perform fine-tuning with our
training data to improve the quality of the language model
representation inside the transformer. However, with LASER embeddings
(section [5.1.1.1](#laser){reference-type="ref" reference="laser"}), we
use a frozen version of the encoder network to generate the embeddings.
Performing fine-tuning on this encoder network might lead to sentence
embeddings more adapted to toxic content, and better classification
scores.

**Other transformer models**: New transformer models are regularly
introduced, with for instance T5 [@raffel2019exploring], ELECTRA
[@clark2020electra] or LongFormer [@beltagy2020longformer]. All these
models improve in some ways the models used in this project and are all
available in the transformers library [^45] for easy use and combination
with a classification head. There are also some older transformer models
that we did not try, such as GPT-2 [@Radford2019LanguageMA] or BART
[@lewis2019bart]. It would be interesting to evaluate their performance
on our task.

**Continuous model tuning**: It was discussed that the use of language
changes over time, with new expressions and insults added over time, and
changes to the meaning of some words. The models trained for this
project might perform well when applied to data gathered recently, but
they might not be relevant anymore in a few years. This would require
training a new model every few years, with up-to-date data to keep up
with the evolution of language, which can get expensive. Instead, it
might be possible to re-use an existing model and adapt it to new
expressions, trends and topics. One could for instance remove the
classification head from a trained model, while keeping the transformer
itself, and then train a new model using a fresh classification head. We
could then preserve all the useful information in the transformer
regarding the language model, adapted to toxic content, while having a
classification head more adapted to current data. This additional data
could be gathered over time from manually reported messages, and allow
for continuous adaptation of models to new data, without requiring a
completely new training.

**Bridging the domain gap**: The largest challenge with cross-lingual
classification is the domain gap, and the discovery of methods to reduce
it would probably lead to a significant improvement in the performance
of current models. As a temporary solution, we could find techniques to
check the compatibly of datasets. Semi-supervised clustering could for
instance be used to properly detect the themes present in multiple
datasets, and we could then select only subsets of toxic content present
in all datasets for better data compatibility. Depending on the target
task, one could automatically select the adapted toxic training data;
the domain gap would then be less problematic which could improve the
quality of the classifier.

References {#references .unnumbered}
==========

[^1]: <https://medium.com/@empathyprojectonline/whats-behind-our-bad-behavior-and-lack-of-empathy-online-8b63e46b76b8>

[^2]: <https://www.theguardian.com/technology/2019/sep/17/revealed-catastrophic-effects-working-facebook-moderator>

[^3]: <https://en.wikipedia.org/wiki/Netzwerkdurchsetzungsgesetz>

[^4]: <https://edition.cnn.com/2020/05/13/tech/french-hate-speech-social-media-law>

[^5]: <https://en.wikipedia.org/wiki/Flesch-Kincaid_readability_tests>

[^6]: Recently, the OpenAI team presented GPT-3, a language model with
    175 billion parameters, compared to BERT's 110 million. It is to our
    knowledge the largest language model to this date
    [@brown2020language].

[^7]: <https://github.com/google-research/bert/blob/master/multilingual.md>

[^8]: An extensive collection of such datasets can be found on
    <http://hatespeechdata.com>

[^9]: <https://developer.twitter.com/en/docs>

[^10]: <https://twython.readthedocs.io/en/latest/>

[^11]: <https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification>

[^12]: <https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification>

[^13]: <https://github.com/uds-lsv/GermEval-2018-Data>

[^14]: <https://projects.fzai.h-da.de/iggsa/projekt/>

[^15]: <http://hatespeechdata.com/>

[^16]: <https://scikit-learn.org>

[^17]: State-of-the-art models with scores and associated papers are
    referenced on
    <https://paperswithcode.com/area/natural-language-processing>. In
    many natural language processing tasks, the best model is usually a
    transformer model

[^18]: <https://scikit-learn.org>

[^19]: <https://pytorch.org/>

[^20]: <https://radimrehurek.com/gensim/>

[^21]: <https://huggingface.co/transformers>

[^22]: NVIDIA GeForce GTX 1080Ti with 12GB of VRAM, 32GB of RAM

[^23]: <https://dumps.wikimedia.org/>

[^24]: <https://deepset.ai/german-bert>

[^25]: <https://github.com/google-research/bert/blob/master/multilingual.md>

[^26]: <https://huggingface.co/transformers/pretrained_models.html>

[^27]: <https://github.com/jasonwei20/eda_nlp>

[^28]: <http://compling.hss.ntu.edu.sg/omw/>

[^29]: <https://bitbucket.org/robvanderg/monoise>

[^30]: <http://aspell.net/>

[^31]: <https://nltk.org/>

[^32]: <http://pytorch.org>

[^33]: <https://huggingface.co/transformers/pretrained_models.html>

[^34]: We use an utility from huggingface's transformers:
    <https://github.com/huggingface/transformers/tree/master/examples/language-modeling>

[^35]: <https://github.com/yannvgn/laserembeddings>

[^36]: <https://github.com/Helsinki-NLP/Opus-MT>

[^37]: <https://huggingface.co/transformers/model_doc/marian.html>

[^38]: List of all language pairs listed on
    <https://huggingface.co/Helsinki-NLP>

[^39]: <https://deepset.ai/german-bert>

[^40]: <https://github.com/valeriobasile/hurtlex>

[^41]: <https://deepset.ai/german-bert>

[^42]: More details on these models on
    <https://huggingface.co/transformers/pretrained_models.html>

[^43]: <https://deepset.ai/german-bert>

[^44]: Examples of linguistic reappropriation:
    <https://en.wikipedia.org/wiki/Reappropriation>

[^45]: List of supported architectures on
    <https://huggingface.co/transformers/model_summary.html>
