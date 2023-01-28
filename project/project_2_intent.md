Anomaly Detection For NLP: OOD detection

General Context

The identification of both Dialog Acts (DA) and Emotion/Sentiment (E/S) in spoken language is an important step toward
improving model performances on spontaneous dialogue task. Especially, it is essential to avoid the generic response
problem, i.e., having an automatic dialog system generate an unspecific response — that can be an answer to a very large
number of user utterances. DA and emotion identificatio are done through sequence labelling systems that are trained in
a supervized manner. DA and Emotion have been particularly usefull for training ChatGPT.

Problem Statement:

We start by introducing the notations. We have a set $D$ of contexts (truncated conversations), i.e., $D
= (C_1,C_2,\dots,C_{|D|})$. Each context $C_i$ is composed of utterances $u$, i.e $C_i= (
u^{L_1}_1,u^{L_2}_2,\dots,u^{L_{|C_i|}}_{|C_i|})$ where $L_i$ is the language of utterance $u_i$. At the lowest level, each
utterance $u_i$ can be seen as a sequence of tokens, i.e $u^{L_i}_i = (\omega^i_1, \omega^i_2, \dots, \omega^i_{|u_i|})
$. For \texttt{DA} classification $y_i$ is the unique dialog act tag associated to $u_i$. In our setting, we work with a
shared vocabulary $\mathcal{V}$ thus $\omega^i_j \in \mathcal{V}$ and $\mathcal{V}$ is language independent.


Your Task:

Build an intent classifier. Several benchmark have been released involving english or multlingual setting

Your reads:
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

Note:
This project is highly similar to project 1.

