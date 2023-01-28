# Intents Classification for Neural Text Generation

## General Context

The identification of both Dialog Acts (DA) and Emotion/Sentiment (E/S) in spoken language is an important step toward
improving model performances on spontaneous dialogue task. Especially, it is essential to avoid the generic response
problem, i.e., having an automatic dialog system generate an unspecific response — that can be an answer to a very large
number of user utterances. DAs and emotions are identified through sequence labeling systems that are trained in a
supervised manner DAs and emotions have been particularly useful for training ChatGPT.

## Problem Statement:

We start by formally defining the Sequence Labelling Problem. At the highest level, we have a set $D$ of conversations
composed of utterances, i.e., $D = (C_1,C_2,\dots,C_{|D|})$ with $Y= (Y_1,Y_2,\dots,Y_{|D|})$ being the corresponding
set of labels (e.g., DA,E/S). At a lower level each conversation $C_i$ is composed of utterances $u$, i.e $C_i= (
u_1,u_2,\dots,u_{|C_i|})$ with $Y_i = (y_1, y_2, \dots, y_{|C_i|})$ being the corresponding sequence of labels: each
$u_i$ is associated with a unique label $y_i$. At the lowest level, each utterance $u_i$ can be seen as a sequence of
words, i.e $u_i = (\omega^i_1, \omega^i_2, \dots, \omega^i_{|u_i|})$.

The goal is to predict Y from D !

## Examples

| Utterances                                                                          | DA                    |
|-------------------------------------------------------------------------------------|-----------------------|
| How long does that take you to get to work?                                         | Question (WH)         |
| Uh, about forty-five, fifty minutes.                                                | Declarative Satetemnt |
| How does that work, work out with, uh, storing your bike and showering and all that |  Question (WH)                       |
|       Yeah                                                                              | backchanel            |
|       It can be a pain .                                                                              | Declarative datetemnt |

## Your Task:

Build an intent classifier. Several benchmark have been released involving english [1] or multlingual setting [2]

## Your reads:

[1] Emile Chapuis,Pierre Colombo, Matthieu Labeau, and Chloé Clavel. Code-switched inspired losses for generic spoken
dialog representations. EMNLP 2021

[2] Emile Chapuis,Pierre Colombo, Matteo Manica, Matthieu Labeau, and Chloé Clavel. Hierarchical pre-training for
sequence labelling in spoken dialog. Finding of EMNLP 2020

[3]Tanvi Dinkar, Pierre Colombo , Matthieu Labeau, and Chloé Clavel. The importance of fillers for text representations
of speech transcripts. EMNLP 2020

[4] Hamid Jalalzai, Pierre Colombo , Chloe Clavel, Eric Gaussier, Giovanna Varni, Emmanuel Vignon, and Anne Sabourin.
Heavy-tailed representations, text polarity classification & data augmentation. NeurIPS 2020

[5] Pierre Colombo, Emile Chapuis, Matteo Manica, Emmanuel Vignon, Giovanna Varni, and Chloé Clavel. Guiding attention
in sequence-to-sequence models for dialogue act prediction. (oral) AAAI 2020

[6] Alexandre Garcia,Pierre Colombo, Slim Essid, Florence d’Alché-Buc, and Chloé Clavel. From the token to the review: A
hierarchical multimodal approach to opinion mining. EMNLP 2020

[7] Pierre Colombo, Wojciech Witon, Ashutosh Modi, James Kennedy, and Mubbasir Kapadia. Affect-driven dialog generation.
NAACL 2019


