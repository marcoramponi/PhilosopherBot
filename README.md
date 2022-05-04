# PhilosopherBot
The Python code in this notebook can be used to implement a baseline Chatbot aimed at reproducing philosophical conversation. It consists of various NLP components: a topic extractor (SpaCy, nltk) via NMF algorithm followed by a summarization pipeline (Transformer). Then a GPT2 model for text generation fine-tuned on questions extracted from texts by F. Nietzsche.

More specifically, it is a blend of two main components:

### 1. Topic Extractor + Summarization
The underlying database is a list of paragraphs of text data.
Each paragraph has been assigned a topic label, via a non-negative matrix factorization algorithm.
Given an input text, the most 'relevant' topic is extracted by computing a cosine similarity function. 
A summarization is then extracted via a Hugging Face Transformers pipeline set for summarization task.
The details of this component are described here:

https://github.com/marcoramponi/Extract-Relevant-Text


### 2. Question Generator
The summarized output from step 1 is used as context to be passed to a question generator algorithm that has been trained on questions coming from the same dataset. The code for the training (fine-tuning) of this model is available here:

https://github.com/marcoramponi/QuestionGenerator-GPT2

In order to load this pre-trained model, a file 'pytorch_model.bin' must be in the current directory. Alternatively, one can use another Seq2Seq generator and experiment with other models.

### Dataset
In order to test this algorithm I prepared a dataset containing a list of paragraphs from F. Nietzsche.
The original dataset 'nietzsche.txt' is publicly availabe on Kaggle:

https://www.kaggle.com/datasets/christopherlemke/philosophical-texts

#### Note: It is recommended to run the notebook on Colab.
