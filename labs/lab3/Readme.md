Something interesting here:

- The `torchtext` package is a part of Pytorch, consisting of data processing utilities for NLP. We can load the pre-trained token vectors via `GloVe` or `FastText`.

Creating a `vocab` instance from a OrderedDict. Then we can insert the special tokens (UNK, PAD...), set default index, etc.

Combining the vocab instance and tokenizers, one can represente a sentence by a list of index. Also Possible to pad it. **Refer to the section `Sequence Classification` in lab3 notebook.** Note: the pre-trained `AutoTokenizer` in transformers can achieve automatically this step: convert text to a list of index and padding it. `AutoTokenizer` is always combined with its corresponding pre-trained model. The latter contains the appropriate Embedding layer. 

Then, one can feed this list of index into a pre-trained Embedding layer.
ps: `torch.nn.Embedding.from_pretrained` permits to load pretrained embeddings in a (customized) model

- Padding the text at 2 levels: dialogue level and sentence level. Refer to the section `Sequence Classification with Conversational Context` in lab3.

---

- `torchinfo.summary` could display the model summary as in TensorFlow

- `nn.CrossEntropyLoss` permits to define the weights, to treat the imbalanced data. Less frequent, more important.

``` python
b_counter = Counter(batch['label'].detach().cpu().tolist())
b_weights = torch.tensor([len(batch['label'].detach().cpu().tolist()) /
                          b_counter[label] if b_counter[label] > 0 else 0 
                          for label in list(range(args['num_class']))])
b_weights = b_weights.to(device)

loss_function = nn.CrossEntropyLoss(weight=b_weights)
```

---

- `sklearn.metrics.classification_report`: Build a text report showing the main classification metrics.

- BERT uses `AdamW` as optimizer, which is based on L2 regularization of Adam. Pay Attention to it if fine-tune BERT. Moreover, BERT uses a **linear** `lr_scheduler` with warming up steps.