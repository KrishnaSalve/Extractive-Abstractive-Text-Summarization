import spacy   
from spacy.lang.en.stop_words import STOP_WORDS   
from string import punctuation
from collections import Counter
import pandas as pd
from heapq import nlargest


# Loading text

text = """There is over 7.6 billion people in this world, meaning there is also a lot of different personalities.
One of those are the negative people we all encounter in our lives.
They make everyone around them miserable, and these people can impact your life in a negative way.
For example: not being happy, having negative thoughts, and loosing friendships.
Even if you try to stay positive someone’s negativity will always drain you.
These people just complain and nag instead of being appreciative of the good they do have in their life.
Negativity is everywhere in your daily life like school, work place, or just in public.
It is a disease and it’s contagious which is why you need to cut it out of your life.
I used to be the happiest person, but the negativity got to me and just took it away from me.
I didn’t do anything about it nor took control of my own happiness.
Which is why I know negativity is contagious and there is people out there who want to bring you down with them.
In a sense that they aren’t happy, so they don’t want to see anyone else happy.
They usually lack self-esteem, aren’t happy, or feel trapped with their feelings.
They want to manipulative people to get what they want.
"""


def summarizer(rawdocs):
  nlp = spacy.load('en_core_web_sm')

  doc = nlp(rawdocs)
  # print(doc)

  stop_words = list(STOP_WORDS)
  tokens = [token.text.lower() for token in doc
            if not token.is_stop and
            not token.is_punct and
            token.text != '\n']
  # print(tokens)
  # print(len(tokens))

  word_freq = Counter(tokens)
  # print(word_freq)

  max_freq = max(word_freq.values())
  # print(max_freq)

  # Normalization
  for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq
    # print(word_freq)

  sent_token = [sent.text for sent in doc.sents]
  # print(sent_token)


  sent_score = {}
  # looping through list of sentences from sent_token.
  for sent in sent_token:
    # split each word from sentences.
    for word in sent.split():
      # checking for lowercase words from word_freq.
      if word.lower() in word_freq.keys():
        # It will check if the word exists or not in word_freq
        if sent not in sent_score.keys():
          # if it doesn't exists it will add the sentence as key and value as word from word_freq to sent_score
          sent_score[sent] = word_freq[word]
        else:
          # if it exists it increments the score of sentence with word of word_freq.
          sent_score[sent] += word_freq[word]
      # print(word)

  # print(sent_score)


  sent_scores = {}
  for key, value in sent_score.items():
    clean_key = key.replace('\n', '')
    sent_scores[clean_key] = value
    # print(sent_scores)


  df = pd.DataFrame(list(sent_scores.items()), columns = ['Sentence', 'Score'])
  # print(df)

  num_sentences = int(len(sent_token) * 0.3)

  final_summary =  nlargest(num_sentences, sent_scores, key = sent_scores.get)
  # print(final_summary)
  summary = " ".join(final_summary)

  return summary, doc, len(rawdocs.split()), len(summary.split())
