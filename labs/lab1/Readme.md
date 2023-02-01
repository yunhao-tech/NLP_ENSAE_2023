Something interesting here:

- `pandas-profiling` library, automatically generate a analysis report
- `NLTK` tokenize: Different tokenizers 
- Zipf's law: For an entity $\omega$, its frequency $f_{\omega}(k)=\frac{1}{k^\theta}$ where k is its frequency rank. In log-scale, it is a linear relationship.
- `gensim` library: unsupervised tool, to learn vector representations of topics in text. Including TF-IDF, LSA, LDA, word2vec, ect. Need to spend some time on it. 
- Stop words: words to be filtered out (i.e. stopped) before or after processing of NLP. No universal list of stop words. 

    `nltk.download('stopwords')` provides a list of stop words. Usually we combine it with punctuations. We can get it from: `from string import punctuation` and convert it to list by `list(punctuation)`. 
