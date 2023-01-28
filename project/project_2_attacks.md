# Safe AI: Protecting AI Systems Against Adversarial Attacks

## General Context

Advanced techniques in Deep Learning (DL) have brought significant improvements over previous state-of-the-art methods
in Natural Language Processing (NLP). The transformers architectures are becoming increasingly scalable, facilitating
their use in applications to critical systems. However, many concerns have been raised about their potential failures:
large neural networks are not trustworthy enough, limiting their adoption in high-risk applications. This is because the
behaviors of large neural networks remain poorly understood, and an essential line of search consists in designing tools
to make them more reliable. However, robustness is instrumental for the responsible adoption of promising NLP methods.
This project is about building more robust algorithms and making NLP systems more robust to data drifts and harmful
adversary attacks.

## Our Setting

Despite the observed high performances of large neural networks, these state-of-the-art models are, as for any machine
learning models, limited to input data whose probability distribution is close to the training dataset on which the
models were trained. This commonplace can be an essential source of dysfunction in NLP contexts. Indeed, languages
characteristics are constantly evolving and particularly subject to distributional shifts. This inherent variability
encountered by deployed NLP systems is a significant barrier to their adoption. Therefore, it is instrumental in
developing tools to measure and detect distributional shifts of different corpus/sentences. The main difficulty comes
from describing the statistical proximity at the level of tokens is not satisfactory. The new paradigm consists in
measuring this proximity through their latent representations, which lend themselves to classical discrepancy measure
tools, which have to be adapted to the high-dimensional nature of transformers layers.

## Our Scenario:

At test time, two main types of situations can be distinguished depending on the source of the abnormal example. The
abnormal test sample comes from a malicious opponent wbicb generates the test sample according to one or multiple
attacks in adversarial attack detection. This scenarios have received much attention in the computer vision community
while remaining overlooked in the text community. To solve the aforementioned problems, related works either rely on
simple heuristics using the output of the networks (black-box scenario) or assume complete knowledge of the network to
compute a similarity between the test sample and the training distribution (white-box scenario).

Several lines of work exist to address this problem.

The first line of work can be assimilated to robust training techniques which consist in incorporating regularization
terms that are smoothing the variability of predictions. This line is compute intensive and unfeasible for us.

The second line of work (the one we are interrested in) corresponds to the design of {\it detectors} that are able to
decide, based on an already existing system, whether an input sample is OOD/a malicious attack or not. This paradigm is
appealing because it does not require any change during the learning phase, making it ready-to-use for an already
deployed system.

## Your Task:

A generated Benchmark for Text can be found [here](https://github.com/anoymous92874838/text-adv-detection) [1]

## Your reads:

[1] KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwak Detection of Word Adversarial Examples in Text Classification

[2] Zhouhang Xie, Jonathan Brophy, Adam Noack, Wencong You, Kalyani Asthana, Carter Perkins, Sabrina Reis, Sameer Singh,
Daniel Lowd Identifying Adversarial Attacks on Text Classifiers

[3] Pierre Colombo, Eduardo D. C. Gomes, Guillaume Staerman, Nathan Noiry, Pablo Piantanida Beyond Mahalanobis-Based
Scores for Textual OOD Detection NeurIPS 2022

[4] Maxime Darrin, Pablo Piantanida, Pierre Colombo, Rainproof: An Umbrella To Shield Text Generators From
Out-Of-Distribution Data

[5] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-ofdistribution examples in neural
networks. arXiv preprint arXiv:1610.02136, 2016.

[6] Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and Dawn Song. Pretrained transformers
improve out-of-distribution robustness. arXiv preprint arXiv:2004.06100, 2020.

[7] Nuno Guerreiro, Pierre Colombo, Pablo Piantanida, André Martins, Optimal Transport for Unsupervised Hallucination
Detection in Neural Machine Translation. arXiv preprint arXiv:2212.09631

[8] Marine Picot, Guillaume Staerman, Federica Granese, Nathan Noiry, Francisco Messina, Pablo Piantanida, Pierre
Colombo A Simple Unsupervised Data Depth-based Method to Detect Adversarial Images

[9] Marine Picot, Nathan Noiry, Pablo Piantanida, Pierre Colombo Adversarial Attack Detection Under Realistic
Constraints

[10] Marine Picot, Francisco Messina, Malik Boudiaf, Fabrice Labeau, Ismail Ben Ayed, Pablo Piantanida Adversarial
Robustness via Fisher-Rao Regularization

[11] Eduardo Dadalto Câmara Gomes, Pierre Colombo, Guillaume Staerman, Nathan Noiry, Pablo Piantanida A Functional
Perspective on Multi-Layer Out-of-Distribution Detection

## Note:

This project is highly similar to project 1.

