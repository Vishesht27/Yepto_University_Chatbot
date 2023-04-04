import tensorflow as tf

from transformers import TFBertTokenizer

from typing import List, Tuple, Union

from ..utils.DataConfig import DataConfig

import random


def extract_intent(model: tf.keras.Model, tokenizer: TFBertTokenizer, intent_labels: List[str], user_input: str):
    """Extract intent from user input
    
    Args:
        model (tf.keras.Model): Trained model
        tokenizer (TFBertTokenizer): Tokenizer
        intent_labels (List[str]): List of all intent labels
        user_input (str): User input
    
    Returns:
        Tuple(str, float): Intent tag and confidence score
    """
    confidence_threshold = 0.5
    # Tokenize user input
    inputs = tokenizer([user_input], padding='longest', truncation=True)

    # Make a prediction
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class_idx = tf.argmax(logits, axis=1).numpy()[0]

    # Get the predicted intent tag and the confidence score
    predicted_tag = str(intent_labels[predicted_class_idx])
    confidence_score = tf.nn.softmax(logits, axis=1)[0][predicted_class_idx].numpy()

    if confidence_score < confidence_threshold:
        predicted_tag = "unknown"
    #st.write("Predicted Tag",predicted_tag)
    #st.write("Predicted Tag.lower",predicted_tag.lower())
    return predicted_tag.lower(), confidence_score


def get_response(intent_tag : str) -> str:
    dataset = DataConfig.get_dataset_instance()
    intent_data = next((item for item in dataset if item["tag"].lower() == intent_tag), None)
    if intent_data:
        responses = intent_data['responses']
        return random.choice(responses)
    else:
        return "I'm sorry, I didn't understand what you meant."