import streamlit as st

import tensorflow as tf
import tensorflow.keras.utils as ku

from transformers import TFBertModel, TFBertTokenizer
from transformers import TFBertForSequenceClassification

from sklearn.model_selection import train_test_split

from .DataConfig import DataConfig

import json
from typing import List, Tuple, Union
import random

class LoadBert:
    @classmethod
    @st.cache(allow_output_mutation=True)
    def load_bert(cls, trainable : bool = False):
        tokenizer = TFBertTokenizer.from_pretrained("bert-base-uncased")
        
        dataset = DataConfig.get_dataset_instance()
        
        
        if not trainable:
            minned_data = []
            for intent in dataset:
                for pattern in intent['patterns']:
                    minned_data.append((pattern, intent['tag']))
            
            tags = [data[1] for data in minned_data]
            intent_labels = list(set(tags))
            model = tf.keras.models.load_model('chatbot/models/bert')
            
            return model, tokenizer, intent_labels
        
        else:
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
            training_data = []
            for intent in train_data:
                for pattern in intent['patterns']:
                    training_data.append((pattern, intent['tag']))

            # Shuffle training data
            random.shuffle(training_data)

            # Tokenize and encode training data
            batch_size = 16
            epoch_size = 10
            patterns = [data[0] for data in training_data]
            tags = [data[1] for data in training_data]
            
            train_encodings = tokenizer(patterns, truncation=True)
            

            # Extract all unique intent tags from the dataset
            intent_labels = list(set(tags))

            train_labels = tf.constant([intent_labels.index(data[1]) for data in training_data])
            train_labels = ku.to_categorical(train_labels, num_classes=len(intent_labels))

            # Load or create BERT model
            try:
                model = TFBertForSequenceClassification.from_pretrained('bert-model')
            except:
                model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))

            
            # Define optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            model.fit(train_encodings, train_labels, epochs=epoch_size, batch_size=batch_size)
            

            # Save the trained model
            model.save('chatbot/models/bert')
            
            return model, tokenizer, intent_labels
                
        