Something interesting here:

- `torchinfo.summary` could display the model summary as in TensorFlow

- `torch.nn.Embedding.from_pretrained` permits to load pretrained embeddings in a (customized) model

- `nn.CrossEntropyLoss` permits to define the weights, to treat the imbalanced data. Less frequent, more important.

``` python
b_counter = Counter(batch['label'].detach().cpu().tolist())
b_weights = torch.tensor([len(batch['label'].detach().cpu().tolist()) /
                          b_counter[label] if b_counter[label] > 0 else 0 
                          for label in list(range(args['num_class']))])
b_weights = b_weights.to(device)

loss_function = nn.CrossEntropyLoss(weight=b_weights)
```

- `sklearn.metrics.classification_report`: Build a text report showing the main classification metrics.

- BERT uses `AdamW` as optimizer, which is based on L2 regularization of Adam. Pay Attention to it if fine-tune BERT. Moreover, BERT uses a **linear** `lr_scheduler` with warming up steps.