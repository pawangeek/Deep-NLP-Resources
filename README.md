Awesome Resource for NLP
====

Table of Contents
----
  - [Table of Contents](#table-of-contents)
  - [Libraries](#libraries)
  - [Dictionary](#dictionary)
  - [Lexicon](#lexicon)
  - [Parsing](#parsing)
  - [Discourse](#discourse)
  - [Language Model](#language-model)
  - [Machine Translation](#machine-translation)
  - [Text Generation](#text-generation)
  - [Text Classification](#text_classification)
  - [Sentiment](#sentiment)
  - [Word Representation](#word-representation)
  - [Question Answer](#question-answer)
  - [Information Extraction](#information-extraction)
  - [Natural Language Inference](#natural-language-inference)
  - [Commonsense](#commonsense)
  - [Other](#other)
  - [Contribute](#contribute)

<span id='libraries'>Useful Libraries</span>
  - [NumPy](http://cs231n.github.io/python-numpy-tutorial/) Stanford's lecture CS231N deals with NumPy, which is fundamental in machine learning calculations.
  - [NLTK](https://github.com/nltk/nltk) It's a suite of libraries and programs for symbolic and statistical natural language processing 
  - [Tensorflow](https://www.tensorflow.org/tutorials/text/word_embeddings) A tutorial provided by Tensorflow. It gives great explanations on the basics with visual aids. Useful in Deep NLP
  - [PyTorch](https://pytorch.org/tutorials/) An awesome tutorial on Pytorch provided by Facebook with great quality.
  - [tensor2tensor](https://github.com/tensorflow/tensor2tensor) Sequence to Sequence tool kit by Google written in Tensorflow.
  - [fairseq](https://github.com/pytorch/fairseq) Sequence to Sequence tool kit by Facebook written in Pytorch.
  - [Hugging Face Transformers](https://github.com/huggingface/transformers) A library based on Transformer provided by Hugging Face that allows easy access to pre-trained models. One of the key NLP libraries to not only developers but researchers as well.
  - [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) A tokenizer library that Hugging Face maintains. It boosts fast operations as the key functions are written in Rust. The latest tokenizers such as BPE can be tried out with Hugging Face tokenizers.
  - [spaCy](https://course.spacy.io/) A tutorial written by Ines, the core developer of the noteworthy spaCy.
  - [torchtext](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/) A tutorial on torchtext, a package that makes data preprocessing handy. Has more details than the official documentation.
  - [SentencePiece](https://github.com/google/sentencepiece) Google's open source library that builds BPE-based vocabulary using subword information.
  - [Gensim](https://github.com/RaRe-Technologies/gensim) Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
  - [polyglot](https://github.com/aboSamoor/polyglot) Polyglot is a natural language pipeline which supports massive multilingual applications.
  - [Textblob](https://github.com/sloria/TextBlob/)Textblob provides simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, WordNet integration, parsing, word inflection
  - [Quepy](https://github.com/machinalis/quepy) A python framework to transform natural language questions to queries in a database query language.
  - [Pattern](https://github.com/clips/pattern) Web mining module for Python, with tools for scraping, natural language processing, machine learning, network analysis and visualization

<span id='dictionary'>Dictionary</span>
----
- Bilingual Dictionary
  - [CC-CEDICT](https://cc-cedict.org/wiki/start) A bilingual dictionary between English and Chinese.
- Pronouncing Dictionary
  - [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) The Carnegie Mellon University Pronouncing Dictionary is an open-source machine-readable pronunciation dictionary for North American English that contains over 134,000 words and their pronunciations. 

<span id='lexicon'>Lexicon</span>
----
  - [PDEV](http://pdev.org.uk) Pattern Dictionary of English Verbs. 
  - [VerbNet](http://verbs.colorado.edu/~mpalmer/projects/verbnet.html) A lexicon that groups verbs based on their semantic/syntactic linking behavior.
  - [FrameNet](http://framenet.icsi.berkeley.edu) A lexicon based on frame semantics.
  - [WordNet](http://wordnet.princeton.edu) A lexicon that describes semantic relationships (such as synonymy and hyperonymy) between individual words.
  - [PropBank](http://en.wikipedia.org/wiki/PropBank) A corpus of one million words of English text, annotated with argument role labels for verbs; and a lexicon defining those argument roles on a per-verb basis.
  - [NomBank](https://nlp.cs.nyu.edu/meyers/NomBank.html)  A dataset marks the sets of arguments that cooccur with nouns in the PropBank Corpus (the Wall Street Journal Corpus of the Penn Treebank), just as PropBank records such information for verbs.
  - [SemLink](https://verbs.colorado.edu/semlink) A project whose aim is to link together different lexical resources via set of mappings. (VerbNet, PropBank, FrameNet, WordNet)
  - [Framester](https://lipn.univ-paris13.fr/framester/) Framester is a hub between FrameNet, WordNet, VerbNet, BabelNet, DBpedia, Yago, DOLCE-Zero, as well as other resources. Framester does not simply creates a strongly connected knowledge graph, but also applies a rigorous formal treatment for Fillmore's frame semantics, enabling full-fledged OWL querying and reasoning on the created joint frame-based knowledge graph.

<span id='parsing'>Parsing</span>
----
  - [PTB](https://catalog.ldc.upenn.edu/LDC99T42) The Penn Treebank (PTB).
  - [Universal Dependencies](http://universaldependencies.org) Universal Dependencies (UD) is a framework for cross-linguistically consistent grammatical annotation and an open community effort with over 200 contributors producing more than 100 treebanks in over 60 languages.
  - [Tweebank](https://github.com/Oneplus/Tweebank) Tweebank v2 is a collection of English tweets annotated in Universal Dependencies that can be exploited for the training of NLP systems to enhance their performance on social media texts.
  - [SemEval-2016 Task 9](https://github.com/HIT-SCIR/SemEval-2016) SemEval-2016 Task 9 (Chinese Semantic Dependency Parsing) Datasets.

<span id='discourse'>Discourse</span>
----
  - [PDTB2.0](https://catalog.ldc.upenn.edu/LDC2008T05) PDTB, version 2.0. annotates 40600 discourse relations, distributed into the following five types: Explicit, Implicit, etc.
  - [PDTB3.0](https://catalog.ldc.upenn.edu/LDC2019T05) In Version 3, an additional 13,000 tokens were annotated, certain pairwise annotations were standardized, new senses were included and the corpus was subject to a series of consistency checks. 
  - [Back-translation Annotated Implicit Discourse Relations](http://www.sfb1102.uni-saarland.de/?page_id=2582) This resource contains annotated implicit discourse relation instances. These sentences are annotated automatically by the back-translation of parallel corpora. 
  - [DiscourseChineseTEDTalks](https://github.com/tjunlp-lab/Shallow-Discourse-Annotation-for-Chinese-TED-Talks) This dataset includes annotation for 16 TED Talks in Chinese.

<span id='lm'>Language Model</span>
----
  - [PTB](https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data) Penn Treebank Corpus in LM Version.
  - [Google Billion Word dataset](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) 1 billion word language modeling benchmark.
  - [WikiText](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger. 

<span id='mt'>Machine Translation</span>
----
  - [Europarl](http://www.statmt.org/europarl) The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek.
  - [UNCorpus](https://conferences.unite.un.org/UNCorpus) The United Nations Parallel Corpus v1.0 is composed of official records and other parliamentary documents of the United Nations that are in the public domain.
  - [CWMT](http://nlp.nju.edu.cn/cwmt-wmt/)  The Zh-EN data collected and shared by China Workshop on Machine Translation (CWMT) community. There are three types of data for Chinese-English machine translation: Monolingual Chinese text, Parallel Chinese-English text, Multiple-Reference text.
  - [WMT](http://www.statmt.org/wmt16/translation-task.html#download) Monolingual language model training data, such as Common Crawl\News Crawl in CS\DE\EN\FI\RO\RU\TR and Parallel data.
  - [OPUS](http://opus.nlpl.eu) OPUS is a growing collection of translated texts from the web. In the OPUS project we try to convert and align free online data, to add linguistic annotation, and to provide the community with a publicly available parallel corpus. 

<span id='textgeneration'>Text Generation</span>
----
  - [Tencent Automatic Article Commenting](http://ai.tencent.com/upload/PapersUploads/article_commenting.tgz) A large-scale Chinese dataset with millions of real comments and a human-annotated subset characterizing the comments’ varying quality. This dataset consists of around 200K news articles and 4.5M human comments along with rich meta data for article categories and user votes of comments.
  - Summarization
    - [BigPatent](https://evasharma.github.io/bigpatent) A summarization dataset consists of 1.3 million records of U.S. patent documents along with human written abstractive summaries.
  - Data-to-Text
    - [Wikipedia Person and Animal Dataset](https://eaglew.github.io/patents/) This dataset gathers 428,748 person and 12,236 animal infobox with description based on Wikipedia dump (2018/04/01) and Wikidata (2018/04/12).
    - [WikiBio](https://github.com/DavidGrangier/wikipedia-biography-dataset) This dataset gathers 728,321 biographies from wikipedia. It aims at evaluating text generation algorithms. For each article, it provide the first paragraph and the infobox (both tokenized).
    - [Rotowire](https://github.com/harvardnlp/boxscore-data) This dataset consists of (human-written) NBA basketball game summaries aligned with their corresponding box- and line-scores. 
    - [MLB](https://github.com/ratishsp/data2text-entity-py) Details in *Data-to-text Generation with Entity Modeling, ACL 2019*

<span id='text_classification'>Text Classification</span>
---------
  - [20Newsgroups](http://qwone.com/~jason/20Newsgroups) The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.
  - [AG's corpus of news articles](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) AG is a collection of more than 1 million news articles.
  - [Yahoo-Answers-Topic-Classification](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset) This corpus contains 4,483,032 questions and their corresponding answers from Yahoo! Answers service.
  - [Google-Snippets](http://jwebpro.sourceforge.net/data-web-snippets.tar.gz) This dataset contains the web search results related to 8 different domains such as business, computers and engineering.
  - [BenchmarkingZeroShot](https://github.com/yinwenpeng/BenchmarkingZeroShot) This repository contains the code and the data for the EMNLP2019 paper "Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach".

<span id='sentiment'>Sentiment</span>
---------
  - [MPQA 3.0](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/) This corpus contains news articles and other text documents manually annotated for opinions and other private states (i.e., beliefs, emotions, sentiments, speculations, etc.). The main changes in this version of the MPQA corpus are the additions of new eTarget (entity/event) annotations.
  - [SentiWordNet](http://sentiwordnet.isti.cnr.it) SentiWordNet is a lexical resource for opinion mining. SentiWordNet assigns to each synset of WordNet three sentiment scores: positivity, negativity, objectivity.
  - [NRC Word-Emotion Association Lexicon ](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) The NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). 
  - [Stanford Sentiment TreeBank](https://nlp.stanford.edu/sentiment/index.html) SST is the dataset of the paper: Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
  - [SemEval-2013 Twitter](https://www.cs.york.ac.uk/semeval-2013/task2/index.html) SemEval 2013 Twitter dataset, which contains phrase-level sentiment annotation. 
  - [Sentihood](https://github.com/uclmr/jack/tree/master/data/sentihood) SentiHood is a dataset for the task of targeted aspect-based sentiment analysis, which contains 5215 sentences. *SentiHood: Targeted Aspect Based Sentiment Analysis Dataset for Urban Neighbourhoods, COLING 2016*.
  - [SemEval-2014 Task 4](http://alt.qcri.org/semeval2014/task4/) This task is concerned with aspect based sentiment analysis (ABSA). Two domain-specific datasets for laptops and restaurants, consisting of over 6K sentences with fine-grained aspect-level human annotations have been provided for training.

<span id='wordrepresentation'>Word Representation</span>
--------------
- Word Embedding
  - [Google News Word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) The model contains 300-dimensional vectors for 3 million words and phrases which trained on part of Google News dataset (about 100 billion words).
  - [GloVe Pre-trained](https://nlp.stanford.edu/projects/glove/) Pre-trained word vectors using GloVe. Wikipedia + Gigaword 5, Common Crawl, Twitter.
  - [fastText Pre-trained](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)  Pre-trained word vectors for 294 languages, trained on Wikipedia using fastText.
  - [BPEmb](https://github.com/bheinzerling/bpemb) BPEmb is a collection of pre-trained **subword embeddings** in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. 
  - [Dependency-based Word Embedding](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) Pre-trained word embeddings based on **Dependency** information, from *Dependency-Based Word Embeddings, ACL 2014.*.
  - [Meta-Embeddings](http://cistern.cis.lmu.de/meta-emb/) performs ensembles of some pretrained word embedding versions, from *Meta-Embeddings: Higher-quality word embeddings via ensembles of Embedding Sets, ACL 2016.*
  - [LexVec](https://github.com/alexandres/lexvec) Pre-trained Vectors based on the **LexVec word embedding model**. Common Crawl, English Wikipedia and NewsCrawl.
  - [MUSE](https://github.com/facebookresearch/MUSE) MUSE is a Python library for multilingual word embeddings, which provide multilingual embeddings for 30 languages and 110 large-scale ground-truth bilingual dictionaries .
  - [CWV](https://github.com/Embedding/Chinese-Word-Vectors) This project provides 100+ Chinese Word Vectors (embeddings) trained with different representations (dense and sparse), context features (word, ngram, character, and more), and corpora.
  - [charNgram2vec](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/) This repository provieds the re-implemented code for pre-training character n-gram embeddings presented in Joint Many-Task (JMT) paper, *A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks, EMNLP2017*.
- Word Representation with Context
  - [ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) Pre-trained contextual representations from large scale bidirectional language models provide large improvements for nearly all supervised NLP tasks.
  - [BERT](https://github.com/google-research/bert) **BERT**, or **B**idirectional **E**ncoder **R**epresentations from **T**ransformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. (2018.10)
  - [OpenGPT](https://github.com/openai/gpt-2) GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text.

<span id="qa">Question Answer</span>
----
- Machine Reading Comprehension
  - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) Stanford Question Answering Dataset (SQuAD) is a new reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage.
  - [CMRC2018](https://github.com/ymcui/cmrc2018) CMRC2018 is released by the Second Evaluation Workshop on Chinese Machine Reading Comprehension. The dataset is composed by near 20,000 real questions annotated by hu- man on Wikipedia paragraphs.
  - [DCRD](https://github.com/DRCKnowledgeTeam/DRCD) Delta Reading Comprehension Dataset is an open domain traditional Chinese machine reading comprehension (MRC) dataset, it contains 10,014 paragraphs from 2,108 Wikipedia articles and 30,000+ questions generated by annotators.
  - [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. This dataset is from the Wikipedia domain and Web domain.
  - [NewsQA](https://datasets.maluuba.com/NewsQA) NewsQA is a crowd-sourced machine reading comprehension dataset of 120K Q&A pairs. 
  - [HarvestingQA](https://github.com/xinyadu/harvestingQA/tree/master/dataset) This folder contains the one million paragraph-level QA-pairs dataset (split into Train, Dev and Test set) described in: *Harvesting Paragraph-Level Question-Answer Pairs from Wikipedia* (ACL 2018).
  - [ProPara](http://data.allenai.org/propara/) ProPara aims to promote the research in natural language understanding in the context of procedural text. This requires identifying the actions described in the paragraph and tracking state changes happening to the entities involved. 
  - [MCScript](http://www.sfb1102.uni-saarland.de/?page_id=2582) MCScript is a new dataset for the task of machine comprehension focussing on commonsense knowledge. It comprises 13,939 questions on 2,119 narrative texts and covers 110 different everyday scenarios. Each text is annotated with one of 110 scenarios.
  - [MCScript2.0](http://www.sfb1102.uni-saarland.de/?page_id=2582) MCScript2.0 is a machine comprehension corpus for the end-to-end evaluation of script knowledge. It contains approx. 20,000 questions on approx. 3,500 texts, crowdsourced based on a new collection process that results in challenging questions. Half of the questions cannot be answered from the reading texts, but require the use of commonsense and, in particular, script knowledge.
  - [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers.
  - [NarrativeQA](https://github.com/deepmind/narrativeqa) NarrativeQA includes the list of documents with Wikipedia summaries, links to full stories, and questions and answers. For a detailed description of this see the paper "The NarrativeQA Reading Comprehension Challenge".
  - [HotpotQA](https://hotpotqa.github.io) HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.
- <span id="SimilarQuestionIden">Duplicate/Similar Question Identification</span>
  - [Quora Question Pairs](http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv) Quora Question Pairs dataset consists of over 400,000 lines of potential question duplicate pairs. [[Kaggle Version Format]](https://www.kaggle.com/c/quora-question-pairs/data)
  - [Ask Ubuntu](https://github.com/taolei87/askubuntu) This repo contains a preprocessed collection of questions taken from AskUbuntu.com 2014 corpus dump. It also comes with 400\*20 mannual annotations, marking pairs of questions as "similar" or "non-similar", from *Semi-supervised Question Retrieval with Gated Convolutions, NAACL2016*.

<span id="ie">Information Extraction</span>
----
- Entity
  - [Shimaoka Fine-grained](http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip) This dataset contains two standard and publicly available datasets for Fine-grained Entity Classification, provided in a preprocessed tokenized format, details in *Neural architectures for ﬁne-grained entity type classiﬁcation, EACL 2017*.
  - [Ultra-Fine Entity Typing](https://homes.cs.washington.edu/~eunsol/_site/open_entity.html) A new entity typing task: given a sentence with an entity mention, the goal is to predict a set of free-form phrases (e.g. skyscraper, songwriter, or criminal) that describe appropriate types for the target entity.
  - [Nested Named Entity Corpus](https://github.com/nickyringland/nested_named_entities) A fine-grained, nested named entity dataset over the full Wall Street Journal portion of the Penn Treebank (PTB), which annotation comprises 279,795 mentions of 114 entity types with up to 6 layers of nesting.
  - [Named Entity Recognition on Code-switched Data](https://code-switching.github.io/2018/#shared-task-id) Code-switching (CS) is the phenomenon by which multilingual speakers switch back and forth between their common languages in written or spoken communication. It contains the training and development data for tuning and testing systems in the following language pairs: Spanish-English (SPA-ENG), and Modern Standard Arabic-Egyptian (MSA-EGY).
  - [MIT Movie Corpus](https://groups.csail.mit.edu/sls/downloads/) The MIT Movie Corpus is a semantically tagged training and test corpus in BIO format. The eng corpus are simple queries, and the trivia10k13 corpus are more complex queries.
  - [MIT Restaurant Corpus](https://groups.csail.mit.edu/sls/downloads/) The MIT Restaurant Corpus is a semantically tagged training and test corpus in BIO format.
- Relation Extraction
  - [Datasets of Annotated Semantic Relationships](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets) **RECOMMEND** This repository contains annotated datasets which can be used to train supervised models for the task of semantic relationship extraction.
  - [TACRED](https://nlp.stanford.edu/projects/tacred/) TACRED is a large-scale relation extraction dataset with 106,264 examples built over newswire and web text from the corpus used in the yearly TAC Knowledge Base Population (TAC KBP) challenges. Details in *Position-aware Attention and Supervised Data Improve Slot Filling, EMNLP 2017*.
  - [FewRel](http://www.zhuhao.me/fewrel/) FewRel is a Few-shot Relation classification dataset, which features 70, 000 natural language sentences expressing 100 relations annotated by crowdworkers.
  - [SemEval 2018 Task7](https://lipn.univ-paris13.fr/~gabor/semeval2018task7/) The training data and evaluation script for SemEval 2018 Task 7: Semantic Relation Extraction and Classification in Scientific Papers.
  - [Chinese-Literature-NER-RE](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset) A discourse-level Named Entity Recognition and Relation Extraction dataset for Chinese literature text. It contains 726 articles, 29,096 sentences and over 100,000 characters in total.
- Event
  - [ACE 2005 Training Data](http://catalog.ldc.upenn.edu/LDC2006T06) The corpus consists of data of various types annotated for entities, relations and events was created by Linguistic Data Consortium with support from the ACE Program, across three languages: English, Chinese, Arabic.
  - [Chinese Emergency Corpus (CEC)](https://github.com/shijiebei2009/CEC-Corpus) Chinese Emergency Corpus (CEC) is built by Data Semantic Laboratory in Shanghai University. This corpus is divided into 5 categories – earthquake, fire, traffic accident, terrorist attack and intoxication of food.
  - [TAC-KBP](https://tac.nist.gov) Event Evaluation is a sub-track in TAC Knowledge Base Population (KBP), which started from 2015. The goal of TAC Knowledge Base Population (KBP) is to develop and evaluate technologies for populating knowledge bases (KBs) from unstructured text.
  - [Narrative Cloze Evaluation Data](https://www.usna.edu/Users/cs/nchamber/data/chains)  Evaluate understanding of a script by predicting the next event given several context events. Details in *Unsupervised Learning of Narrative Schemas and their Participants, ACL 2009*.
  - [Event Tensor](https://github.com/StonyBrookNLP/event-tensors/tree/master/data) A evaluation dataset about Schema Generation/Sentence Similarity/Narrative Cloze, which is proposed by *Event Representations with Tensor-based Compositions, AAAI 2018.*. 
  - [SemEval-2015 Task 4](http://alt.qcri.org/semeval2015/task4/) TimeLine: Cross-Document Event Ordering. Given a set of documents and a target entity, the task is to build an event TimeLine related  to that entity, i.e. to detect, anchor in time and order the events involving the target entity.
  - [RED](https://catalog.ldc.upenn.edu/LDC2016T23) Richer Event Description  consists of coreference, bridging and event-event relations (temporal, causal, subevent and reporting relations) annotations over 95 English newswire, discussion forum and narrative text documents, covering all events, times and non-eventive entities within each document.
  - [InScript](http://www.sfb1102.uni-saarland.de/?page_id=2582) The InScript corpus contains a total of 1000 narrative texts crowdsourced via Amazon Mechanical Turk. It is annotated with script information in the form of scenario-specific events and participants labels.
  - [AutoLabelEvent](https://github.com/acl2017submission/event-data) The data of the work in *Automatically Labeled Data Generation for Large Scale Event Extraction, ACL2017*.
  - [EventInFrameNet](https://github.com/liushulinle/events_in_framenet) The data of the work in *Leveraging FrameNet to Improve Automatic Event Detection, ACL2016*.
  - [MEANTIME](http://www.newsreader-project.eu/results/data/wikinews/) The MEANTIME Corpus (the NewsReader Multilingual Event ANd TIME Corpus) consists of a total of 480 news articles: 120 English Wikinews articles on four topics and their translations in Spanish, Italian, and Dutch. It has been annotated manually at multiple levels, including entities, events, temporal information, semantic roles, and intra-document and cross-document event and entity coreference.
  - [BioNLP-ST 2013](http://2013.bionlp-st.org/tasks) BioNLP-ST 2013 features the six event extraction tasks: Genia Event Extraction for NFkB knowledge base construction, Cancer Genetics, Pathway Curation, Corpus Annotation with Gene Regulation Ontology, Gene Regulation Network in Bacteria, and Bacteria Biotopes (semantic annotation by an ontology).
  - Event Temporal and Causal Relations
    - [CaTeRS](http://cs.rochester.edu/nlp/rocstories/CaTeRS/) Causal and Temporal Relation Scheme (CaTeRS),which is unique in simultaneously capturing a com- prehensive set of temporal and causal relations between events. CaTeRS contains a total of 1,600 sentences in the context of 320 five-sentence short stories sampled from ROCStories corpus.
    - [Causal-TimeBank](https://hlt-nlp.fbk.eu/technologies/causal-timebank) Causal-TimeBank is the TimeBank corpus taken from TempEval-3 task, which puts new information about causality in the form of C-SIGNALs and CLINKs annotation. 6,811 EVENTs (only instantiated events by MAKEINSTANCE tag of TimeML), 5,118 TLINKs (temporal links), 171 CSIGNALs (causal signals), 318 CLINKs (causal links).
    - [EventCausalityData](https://cogcomp.seas.upenn.edu/page/resource_view/27) The EventCausality dataset provides relatively dense causal annotations on 25 newswire articles collected from CNN in 2010.
    - [EventStoryLine](https://github.com/tommasoc80/EventStoryLine) A benchmark dataset for the temporal and causal relation detection.
    - [TempEval-3](https://www.cs.york.ac.uk/semeval-2013/task1/index.html) The TempEval-3 shared task aims to advance research on temporal information processing.
    - [TemporalCausalReasoning](https://github.com/qiangning/TemporalCausalReasoning) A dataset with both temporal and causal relations annotation. The temporal relations were annotated based on the scheme proposed in "A Multi-Axis Annotation Scheme for Event Temporal Relations" using CrowdFlower; the causal relations were mapped from the "EventCausalityData".
    - [TimeBank](https://catalog.ldc.upenn.edu/LDC2006T08) TimeBank 1.2 contains 183 news articles that have been annotated with temporal information, adding events, times and temporal links(TLINKs) between events and times.
    - [TimeBank-EventTime Corpus](https://www.ukp.tu-darmstadt.de/data/timeline-generation/temporal-anchoring-of-events/) This dataset is a subset of the TimeBank Corpus with a new annotation scheme to anchor events in time. [Detailed description](https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2016/2016_Reimers_Temporal_Anchoring_of_Events.pdf).
  - Event Factuality
    - [UW Event Factuality Dataset](https://bitbucket.org/kentonl/factuality-data/src) This dataset contains annotations of text from the TempEval-3 corpus with factuality assessment labels.
    - [FactBank 1.0](https://catalog.ldc.upenn.edu/ldc2009t23) FactBank 1.0, consists of 208 documents (over 77,000 tokens) from newswire and broadcast news reports in which event mentions are annotated with their degree of factuality.
    - [CommitmentBank](https://github.com/mcdm/CommitmentBank) The CommitmentBank is a corpus of 1,200 naturally occurring discourses whose final sentence contains a clause-embedding predicate under an entailment canceling operator (question, modal, negation, antecedent of conditional).
    - [UDS](http://decomp.io/projects/factuality/) Universal Decompositional Semantics It Happened Dataset, covers the entirety of the English Universal Dependencies v1.2 (EUD1.2) treebank, a large event factuality dataset.
    - [DLEF](https://github.com/qz011/dlef/tree/master/dlef_corpus) A document level event factuality (DLEF) dataset, which includes the source (English and Chinese), detailed guidelines for both document- and sentence-level event factuality.
  - Event Coreference
    - [ECB 1.0](http://adi.bejan.ro/data/ECB1.0.tar.gz) This corpus consists of a collection of Google News documents annotated with within- and cross-document event coreference information. The documents are grouped according to the Google News Cluster, each group of documents representing the same seminal event (or topic).
    - [EECB 1.0](http://nlp.stanford.edu/pubs/jcoref-corpus.zip) Compared to ECB 1.0, this dataset is extended in two directions: (i) fully annotated sentences, and (ii) entity coreference relations. In addition, annotators removed relations other than coreference (e.g., subevent, purpose, related, etc.).
    - [ECB+](http://www.newsreader-project.eu/results/data/the-ecb-corpus) The ECB+ corpus is an extension to the ECB 1.0. A newly added corpus component consists of 502 documents that belong to the 43 topics of the ECB but that describe different seminal events than those already captured in the ECB.
- Open Information Extraction
  - [oie-benchmark](https://github.com/gabrielStanovsky/oie-benchmark#converting-qa-srl-to-open-ie) This repository contains code for converting QA-SRL annotations to Open-IE extractions and comparing Open-IE parsers against a converted benchmark corpus.
  - [NeuralOpenIE](https://onedrive.live.com/?authkey=%21AHj1kHDE5TSS0e8&cid=C826C2D6F4C7D993&id=C826C2D6F4C7D993%213193&parId=C826C2D6F4C7D993%213189&action=locate) A training dataset from *Neural Open Information Extraction*, ACL 2018. here are a total of 36,247,584 hsentence, tuplei pairs extracted from Wikipedia dump using OPENIE4.
- Other
  - [WikilinksNED](https://github.com/yotam-happy/NEDforNoisyText) A large-scale Named Entity Disambiguation dataset of text fragments from the web, which is significantly noisier and more challenging than existing news-based datasets.

<span id="nli">Natural Language Inference</span>
----
  - [SNLI](https://nlp.stanford.edu/projects/snli/) The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).
  - [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) The Multi-Genre Natural Language Inference (MultiNLI) corpus is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information. The corpus is modeled on the SNLI corpus, but differs in that covers **a range of genres** of spoken and written text, and supports a distinctive cross-genre generalization evaluation.
  - [Scitail](http://data.allenai.org/scitail/) The SciTail dataset is an entailment dataset created from multiple-choice science exams and web sentences. The domain makes this dataset different in nature from previous datasets, and it consists of more factual sentences rather than scene descriptions.
  - [PAWS](https://g.co/dataset/paws) A new dataset with 108,463 well-formed paraphrase and non-paraphrase pairs with high lexical overlap. *PAWS: Paraphrase Adversaries from Word Scrambling*

<span id="commonsense">Commonsense</span>
----
  - [ConceptNet](http://conceptnet.io) ConceptNet is a multilingual knowledge base, representing words and phrases that people use and the common-sense relationships between them.
    - [Commonsense Knowledge Representation](https://ttic.uchicago.edu/~kgimpel/commonsense.html) ConceptNet-related resources. Details in *Commonsense Knowledge Base Completion. Proc. of ACL, 2016*
  - [ATOMIC](https://homes.cs.washington.edu/~msap/atomic/), an atlas of everyday commonsense reasoning, organized through 877k textual descriptions of inferential knowledge. ATOMIC focuses on inferential knowledge organized as typed if-then relations with variables.
  - [SenticNet](http://sentic.net) SenticNet provides a set of semantics, sentics, and polarity associated with 100,000 natural language concepts. SenticNet consists of a set of tools and techniques for sentiment analysis combining commonsense reasoning, psychology, linguistics, and machine learning. 

<span id="other">Other</span>
----
  - [QA-SRL](https://dada.cs.washington.edu/qasrl/) This dataset use question-answer pairs to model verbal predicate-argument structure. The questions start with wh-words (Who, What, Where, What, etc.) and contains a verb predicate in the sentence; the answers are phrases in the sentence.
  - [QA-SRL 2.0](https://github.com/uwnlp/qasrl-bank) This repository is the reference point for QA-SRL Bank 2.0, the dataset described in the paper Large-Scale QA-SRL Parsing, ACL 2018. 
  - [NEWSROOM](https://summari.es) CORNELL NEWSROOM is a large dataset for training and evaluating summarization systems. It contains 1.3 million articles and summaries written by authors and editors in the newsrooms of 38 major publications.
  - [CoNLL 2010 Uncertainty Detection](http://rgai.inf.u-szeged.hu/conll2010st/tasks.html) The aim of this task is to identify sentences in texts which contain unreliable or uncertain information. Training Data contains biological abstracts and full articles from the **BioScope** (biomedical domain) corpus and paragraphs from **Wikipedia** possibly containing weasel information.
  - [COLING 2018 automatic identification of verbal MWE](https://gitlab.com/parseme/sharedtask-data/tree/master/1.1) Corpora were annotated by human annotators with occurrences of verbal multiword expressions (VMWEs) according to common annotation guidelines. For example, "He **picked** one **up**."
- Scientific NLP
  - [PubMed 200k RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) PubMed 200k RCT is new dataset based on PubMed for sequential sentence classification. The dataset consists of approximately 200,000 abstracts of randomized controlled trials, totaling 2.3 million sentences.
  - [Automatic Academic Paper Rating](https://github.com/lancopku/AAPR) A dataset for automatic academic paper rating (AAPR), which automatically determine whether to accept academic papers. The dataset consists of 19,218 academic papers by collecting data on academic pa- pers in the field of artificial intelligence from the arxiv.
  - [ACL Title and Abstract Dataset](https://github.com/EagleW/ACL_titles_abstracts_dataset) This dataset gathers 10,874 title and abstract pairs from the ACL Anthology Network (until 2016).
  - [SCIERC](http://nlp.cs.washington.edu/sciIE/) A dataset includes annotations for entities, relations, and coreference clusters in scientific articles.
  - [SciBERT](https://github.com/allenai/scibert) SciBERT is a BERT model trained on scientific text. A broad set of scientific nlp datasets under the data/ directory across ner, parsring, pico and text classification.
  - [5AbstractsGroup](https://github.com/qianliu0708/5AbstractsGroup) The dataset contains academic papers from five different domains collected from the Web of Science, namely business, artifical intelligence, sociology, transport and law.
  - [SciCite](https://github.com/allenai/scicite) A new large dataset of citation intent from *Structural Scaffolds for Citation Intent Classification in Scientific Publications*
  - [ACL-ARC](https://github.com/allenai/scicite) A dataset of citation intents in the computational linguistics domain (ACL-ARC) introduced by *Measuring the Evolution of a Scientific Field through Citation Frames*.
  - [GASP](https://github.com/ART-Group-it/GASP) The dataset consists of list of cited abstracts associated with the corresponding source abstract. The goal is to generete the abstract of a target paper given the abstracts of cited papers.

<span id="contribute">Contribute</span>
Contributions welcome!
