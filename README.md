# NLP ENSAE 2023: 

Here is the repo for the class in ENSAE: Machine Learning for Natural Language Processing. 

Main Instructor: Pierre COLOMBO <colombo.pierre@gmail.com>


---

# What have we learned?

1. A project on Sequence labelling task (Dialogue Act and Emotion/Sentiment classification). The pre-trained RoBERTa and DeBERTa have been used to tackle this task. And the contextual embeddings pre-trained on written corpus are proven useful for spoken language. You can also find the paper on [OpenReview](https://openreview.net/forum?id=Et8MZ0_e4i). 
    
    Team members: Yunhao CHEN, Hélène RONDEY


2. Four labs which reveal the basic pipeline in NLP. You can find some interesting points below:

---

## visualization tools

- `pandas-profiling` library, automatically generate a analysis report
- [Projector tensorflow](https://projector.tensorflow.org): interactive visualization and analysis of high-dimensional data. PCA, t-SNE, etc
- `torchinfo.summary` could display the model summary as in TensorFlow
- `sklearn.metrics.classification_report`: Build a text report showing the main classification metrics.

## NLP packages

- `gensim` library: unsupervised tool, to learn vector representations of topics in text. Including TF-IDF, LSA, LDA, word2vec, ect. Need to spend some time on it. Refer to its documentation: https://radimrehurek.com/gensim/apiref.html
- Stop words: words to be filtered out (i.e. stopped) before or after processing of NLP. No universal list of stop words. 

    `nltk.download('stopwords')` provides a list of stop words. Usually we combine it with punctuations. We can get it from: `from string import punctuation` and convert it to list by `list(punctuation)`. 

- `gensim.models.phrases`: Automatically detect common phrases (ex: multi-word expressions, word n-gram collocations) from a stream of sentences. Ex: we'd like "New York" as one expression instead of "New" and "York".

- `nltk punkt`: Sentence Tokenizer that divides a text into a list of sentences. The NLTK data package includes a pre-trained Punkt tokenizer for English.

- The `torchtext` package is a part of Pytorch, consisting of data processing utilities for NLP. We can load the pre-trained token vectors via `GloVe` or `FastText`.
    
---

## Pipeline in NLP

- Text cleaning process: 
    1. Divide samples into sentences, using `nltk punkt`
    2. Tokenize each sentences, using tokenizers in `nltk.tokenize`. Then clean the obtained tokens (E.g. remove the HTML tags...). 
    3. Detect and combine the multi-word expression, using `Phrases`


- Creating a `vocab` instance from a OrderedDict. Then we can insert the special tokens (UNK, PAD...), set default index, etc.

    Combining the vocab instance and tokenizers, one can represente a sentence by a list of index. Also Possible to pad it. **Refer to the section `Sequence Classification` in lab3 notebook.** Note: the pre-trained `AutoTokenizer` in transformers can achieve automatically this step: convert text to a list of index and padding it. `AutoTokenizer` is always combined with its corresponding pre-trained model. The latter contains the appropriate Embedding layer. 

    Then, one can feed this list of index into a pre-trained Embedding layer.
    
    ps: `torch.nn.Embedding.from_pretrained` permits to load pretrained embeddings in a (customized) model
    

- Learn how to build a Vocabulary from scratch. Define `stoi()` (str to index) and `itos()` (index to str), and special tokens; Add special tokens into sentences.  

    Attention! **To define a Vocubulary, we use the tokens, instead of the words**. We always depart from the tokenized text! 
    
- Padding the text at 2 levels: dialogue level and sentence level. Refer to the section `Sequence Classification with Conversational Context` in lab3.

---

## About Python

- `input()` is a built-in function allowing user to interactively input. 

- use `multiprocessing.cpu_count()` to get the number of available CPU cores, in order to set `n_workers`. **Setting workers to number of cores is a good rule of thumb**.

- `nvidia-smi` bash command to check GPU info; **`lscpu` to check CPU info**.


## About Training 

- `nn.CrossEntropyLoss` permits to define the weights, to treat the imbalanced data. Less frequent, more important.

``` python
b_counter = Counter(batch['label'].detach().cpu().tolist())
b_weights = torch.tensor([len(batch['label'].detach().cpu().tolist()) /
                          b_counter[label] if b_counter[label] > 0 else 0 
                          for label in list(range(args['num_class']))])
b_weights = b_weights.to(device)

loss_function = nn.CrossEntropyLoss(weight=b_weights)
```

- BERT uses `AdamW` as optimizer, which is based on L2 regularization of Adam. Pay Attention to it if fine-tune BERT. Moreover, BERT uses a **linear** `lr_scheduler` with warming up steps.



