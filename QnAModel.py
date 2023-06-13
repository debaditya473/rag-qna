# standard
import os
import numpy as np
import pandas as pd
import pickle

# DL
import torch
import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import sentence_transformers

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


class QnAModel():
    def __init__(self, model_corpus, retreiver_model = None, ST_retreiver: str = None, gen_model = None):
        self.corpus = model_corpus
        
        # Retreiver model
        if retreiver_model == None:
            retreiver_model = "deepset/roberta-base-squad2"
        self.retreiver_model = pipeline('question-answering', model=retreiver_model, tokenizer = retreiver_model)
        
        
        # Sentence tarnsformer model (unused)
        self.ST_retreiver = sentence_transformers.SentenceTransformer(ST_retreiver)
        
        #Generator Model
        if gen_model == None:
            gen_model = pickle.load(open('/kaggle/input/models/flan-t5-large-finetuned-finetuning_final_data-10_epochs.h5', 'rb'))
            self.gen_model = transformers.pipeline("text2text-generation", model = gen_model, tokenizer = 'google/flan-t5-large')
        else:
            self.gen_model = transformers.pipeline("text2text-generation", model = gen_model, tokenizer = gen_model)
        
        pass
    
    def tf_idf_retreival(self, query, k):
        # Retrieving relvant documents using tf-idf
        vectorizer = TfidfVectorizer()

        query_emb = vectorizer.fit_transform(self.corpus)
        doc_emb = vectorizer.transform([query])
        Z = cosine_similarity(doc_emb, query_emb)[0]
        top_ind = np.argsort(Z)[::-1][:k]
        top_docs = self.corpus[top_ind]
        return top_docs
        
    def DL_retreiver(self, query, top_documents, n):
        # Retrieving top documents using an end to end question answering model
        results = []
        for doc in top_documents:
            QA_input = QA_input = {
                'question': query,
                'context': doc
            }
            res = self.retreiver_model(QA_input)
            results.append((doc, res['score']))

        # get top n scores
        final = sorted(results, key=lambda x: x[1])[::-1][:n]
        return final
        
    def SentenceTransform_retreiver(self, query, top_documents, n):
        # retreiving top documents using a sentence transformer using vector embedding similarities 
        if self.ST_retreiver == None:
            raise Exception("Sentence Transformer not provided.")
        
        query_emb = self.ST_retreiver.encode([query])
        doc_emb = self.ST_retreiver.encode(top_documents)
        scores  = cosine_similarity(query_emb, doc_emb)[0]
        doc_score_pairs = list(zip(top_documents, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)[:n]
        return doc_score_pairs
    
    def generate_result(self, query, top_documents, max_length):
        # Generating results using a text to text generator model
        
        # Our top douments are the context for the model
        context = ' '.join(top_documents)
        
        # We prompt the task to the model
        prompt = f'Answer this question: {query} \n Given this is true: {context}'
        return self.gen_model(prompt, max_length=max_length)
    
    def answer_question(self, query, k = 50, n = 5, max_length = 75, use_sent_transformer = False):
        # tf idf
        top_documents = self.tf_idf_retreival(query, k)
        # retreive
        if use_sent_transformer:
            top_documents = self.SentenceTransform_retreiver(query, top_documents, n)
        else:
            top_documents = self.DL_retreiver(query, top_documents, n)
            #print(top_documents)
        
        
        # generate
        top_documents = [pair[0] for pair in top_documents]          
        
        output = self.generate_result(query, top_documents, max_length)
        
        return output