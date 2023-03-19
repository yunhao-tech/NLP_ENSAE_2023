Something interesting here:

- `gensim` provides various models and embedding algorighms, such as TF-IDF, word2Vec, Fasttext... Refer to its documentation: https://radimrehurek.com/gensim/apiref.html

- `gensim.models.phrases`: Automatically detect common phrases (ex: multi-word expressions, word n-gram collocations) from a stream of sentences. Ex: we'd like "New York" as one expression instead of "New" and "York".

- `nltk punkt`: Sentence Tokenizer that divides a text into a list of sentences. The NLTK data package includes a pre-trained Punkt tokenizer for English.

- Text cleaning process: 
    1. Divide samples into sentences, using `nltk punkt`
    2. Tokenize each sentences, using tokenizers in `nltk.tokenize`. Then clean the obtained tokens (E.g. remove the HTML tags...). 
    3. Detect and combine the multi-word expression, using `Phrases`

- `sklearn` provides a `TfidfVectorizer` for TF-IDF textual vectorization method

- `input()` is a built-in function allowing user to interactively input. 

- use `multiprocessing.cpu_count()` to get the number of available CPU cores, in order to set `n_workers`. **Setting workers to number of cores is a good rule of thumb**.

- [Projector tensorflow](https://projector.tensorflow.org): interactive visualization and analysis of high-dimensional data. PCA, t-SNE, etc