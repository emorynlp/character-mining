# Character Mining

The Character Mining project challenges machine comprehension on multiparty dialogue.
The objective of this project is to infer explicit and implicit contexts about individual characters through their conversations.
This is an open-source project led by the [Emory NLP](http://nlp.mathcs.emory.edu) research group that provides resources for the following tasks:

* [Character Identification](../../../character-identification) (since May 2016).
* [Emotion Detection](../../../emotion-detection) (since May 2017).
* [Reading Comprehension](../../../reading-comprehension) (since May 2018).
* [Questiong Answering](../../../FriendsQA) (since May 2019).
* [Personality Detection](../../../personality-detection) (since Sep 2019).

We welcome feedbacks and contributions from the community.
Most of our annotation are crowdsourced; implying that, errors are expected to be found.
Please make pull requests if you wish to fix errors in our datasets.

## Dataset

Our dataset is based on the popular TV show called [*Friends*](https://en.wikipedia.org/wiki/Friends).
Transcripts for all 10 seasons of the show as well as manual and crowdsourced annotation for subparts of the show are provided.
All text data are available in the [JSON files](json); please visit the individual task pages to retrieve datasets specifically designed for those tasks.

## Statistics

Each season consists of episodes, each episode is divided into scenes, each scene comprises utterances, each utterance is a list of sentences where tokens are split.

| Season ID | Episodes | Scenes | Utterances | Sentences |  Tokens | Speakers |
|:---------:|---------:|-------:|-----------:|----------:|--------:|---------:|
|    s01    |       24 |    326 |      5,968 |    10,790 |  81,453 |      107 |
|    s02    |       24 |    293 |      5,747 |     9,337 |  81,910 |      107 |
|    s03    |       25 |    348 |      6,495 |    10,858 |  90,753 |      108 |
|    s04    |       24 |    338 |      6,318 |    10,889 |  87,289 |      100 |
|    s05    |       24 |    311 |      6,220 |    11,133 |  83,907 |      107 |
|    s06    |       25 |    350 |      6,458 |    11,496 |  90,384 |      112 |
|    s07    |       24 |    332 |      6,314 |    11,340 |  84,974 |       94 |
|    s08    |       24 |    288 |      6,220 |    11,714 |  86,164 |      107 |
|    s09    |       24 |    302 |      6,322 |    11,831 |  93,773 |       99 |
|    s10    |       18 |    219 |      5,247 |     9,345 |  69,493 |       78 |
|   Total   |      236 |  3,107 |     61,309 |   108,733 | 850,100 |      700 |

Some utterances include action notes.
In the following example, extracted from `s01_e01_c01_u028`, the speaker is talking *to Ross*, which is indicated by the action note:

```
"transcript": "Let me get you some coffee.",
"transcript_with_note": "(to Ross) Let me get you some coffee.",
```

The followings show the statistics including action notes:

| Season ID | Utterances | Sentences |    Tokens |
|:---------:|-----------:|----------:|----------:|
|    s01    |      6,626 |    12,088 |   100,773 |
|    s02    |      6,048 |    10,565 |    97,763 |
|    s03    |      7,267 |    12,288 |   117,912 |
|    s04    |      7,119 |    12,811 |   116,703 |
|    s05    |      7,082 |    13,540 |   118,509 |
|    s06    |      7,235 |    13,506 |   120,471 |
|    s07    |      7,019 |    13,363 |   116,341 |
|    s08    |      6,845 |    13,321 |   109,984 |
|    s09    |      6,653 |    13,548 |   119,090 |
|    s10    |      5,479 |    11,029 |    93,390 |
|   Total   |     67,373 |   126,059 | 1,110,936 |


## Documentations

* How to retrieve information from the JSON files: [`load_json.ipynb`](doc/load_json.ipynb).

## References

* [Transformers to Learn Hierarchical Contexts in Multiparty Dialogue for Span-based Question Answering](https://github.com/emorynlp/FriendsQA/blob/master). Changmao Li and Jinho D. Choi. In Proceedings of the Conference of the Association for Computational Linguistics, ACL'20, 2020.
* [Modeling Personality with Attentive Networks and Contextual Embeddings](https://arxiv.org/abs/1911.09304). Hang Jiang, Xianzhe Zhang, and Jinho D. Choi. In Proceedings of the AAAI Student Abstract and Poster Program, AAAI:SAP'20, 2020 ([poster]()).
* [FriendsQA: Open-Domain Question Answering on TV Show Transcripts](https://www.aclweb.org/anthology/W19-5923). Zhengzhe Yang and Jinho D. Choi. In Proceedings of the Annual Conference of the ACL Special Interest Group on Discourse and Dialogue, SIGDIAL'19, 2019 ([slides](https://www.slideshare.net/jchoi7s/friendsqa-opendomain-question-answering-on-tv-show-transcripts-154329602)).
* [They Exist! Introducing Plural Mentions to Coreference Resolution and Entity Linking](http://aclweb.org/anthology/C18-1003). Ethan Zhou and Jinho D. Choi. In Proceedings of the 27th International Conference on Computational Linguistics, COLING'18, 2018 ([slides](https://www.slideshare.net/jchoi7s/they-exist-introducing-plural-mentions-to-coreference-resolution-and-entity-linking)).
* [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](http://aclweb.org/anthology/S18-1007), Jinho D. Choi and Henry Y. Chen, Proceedings of the International Workshop on Semantic Evaluation, SemEval'18, 2018 ([slides](https://www.slideshare.net/jchoi7s/semeval-2018-task-4-character-identification-on-multiparty-dialogues)).  
* [Challenging Reading Comprehension on Daily Conversation: Passage Completion on Multiparty Dialog](http://aclweb.org/anthology/N18-1185). Kaixin Ma, Tomasz Jurczyk, and Jinho D. Choi. In Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics, NAACL'18, 2018 ([poster](https://www.slideshare.net/jchoi7s/challenging-reading-comprehension-on-daily-conversation-passage-completion-on-multiparty-dialog), [source](https://github.com/Mayer123/Multiparty-Dialog-RC)). 
* [Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks](https://arxiv.org/abs/1708.04299). Sayyed Zahiri and Jinho D. Choi. In The AAAI Workshop on Affective Content Analysis, AFFCON'18, 2018.
* [Cross-domain Document Retrieval: Matching between Conversational and Formal Writings](http://www.aclweb.org/anthology/W17-5407). Tomasz Jurczyk and Jinho D. Choi. In Proceedings of the EMNLP Workshop on Building Linguistically Generalizable NLP Systems, of BLGNLP'17, 2017 ([slides](https://www.slideshare.net/jchoi7s/crossdomain-document-retrieval-matching-between-conversational-and-formal-writings)). 
* [Robust Coreference Resolution and Entity Linking on Dialogues: Character Identification on TV Show Transcripts](http://www.aclweb.org/anthology/K17-1023), Henry Y. Chen, Ethan Zhou, and Jinho D. Choi. Proceedings of the 21st Conference on Computational Natural Language Learning, CoNLL'17, 2017 ([slides](https://www.slideshare.net/jchoi7s/robust-coreference-resolution-and-entity-linking-on-dialogues-character-identification-on-tv-show-transcripts)).
* [Text-based Speaker Identification on Multiparty Dialogues Using Multi-document Convolutional Neural Networks](http://aclweb.org/anthology/P17-3009.pdf). Kaixin Ma, Catherine Xiao, and Jinho D. Choi. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, ACL:SRW'17, 2017 ([poster](https://www.slideshare.net/jchoi7s/textbased-speaker-identification-on-multiparty-dialogues-using-multidocument-convolutional-neural-networks)).
* [Character Identification on Multiparty Conversation: Identifying Mentions of Characters in TV Shows](http://www.aclweb.org/anthology/W16-3612), Henry Y. Chen and Jinho D. Choi. Proceedings of the 17th Annual SIGdial Meeting on Discourse and Dialogue, SIGDIAL'16, 2016 ([poster](https://www.slideshare.net/jchoi7s/character-identification-on-multiparty-conversation-identifying-mentions-of-characters-in-tv-shows)).

## Contact

* [Jinho D. Choi](http://www.cs.emory.edu/~choi).

<script src="https://bibbase.org/show?bib=https://bibbase.github.io/pubs.bib&jsonp=1"></script>