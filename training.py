import streamlit as st
import json
import numpy as np
from transformers import BertTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import datasets
from datasets import DatasetDict, Dataset
import tempfile
import os
import zipfile
import streamlit as st
from io import BytesIO
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# Define Streamlit app layout
st.title("NER Training App")
st.header("Training Configuration")

pre_train_model = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(pre_train_model)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Input fields
dataset_file = st.file_uploader("Upload dataset file", type=["json"])
epoch = st.number_input("Number of epochs", value=3)
output_model = st.text_input("Output model name", value="ner_model")
# Check if dataset is uploaded
if dataset_file is not None:
    # Your training script code
    def load_data(dataset_file):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(dataset_file.read())
            tmp_file.close()
            with open(tmp_file.name, "r") as f:
                return json.load(f)

    def main():
        conll2003 = load_data(dataset_file)
        
        dataset = {}
        for split_name, split_data in conll2003.items():
            tokenized_data = [tokenize_and_align_labels(example) for example in split_data]
            dataset[split_name] = tokenized_data
            
        training_dataset = convert_data(dataset['train'])
        validating_dataset = convert_data(dataset['validation'])
        id_length_train = len(training_dataset['id'])
        id_length_valid = len(validating_dataset['id'])
        train_dataset = Dataset.from_dict(training_dataset)
        train_dataset = train_dataset.select(list(range(id_length_train)))
        validation_dataset = Dataset.from_dict(validating_dataset)
        validation_dataset = validation_dataset.select(list(range(id_length_valid)))
        dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        
        if st.button('Train'):
            train_model(dataset_dict, num_epoch=epoch)
    
    def train_model(dataset_dict, num_epoch=3):
        args = TrainingArguments(
            "test-ner",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_epoch,
            weight_decay=0.01,
        )

        model = AutoModelForTokenClassification.from_pretrained(pre_train_model, num_labels=9, ignore_mismatched_sizes=True)

        trainer = Trainer(
            model,
            args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        st.text('Training started...')
        

        for epoch in range(num_epoch):
            epoch_text = ""
            epoch_text = f'Epoch {epoch + 1}/{num_epoch}'
            st.text(epoch_text)
            trainer.train()
            st.text('Epoch completed.')
        output_folder = 'model/'
        dynamic_output_path = os.path.join(output_folder, output_model)
        model.save_pretrained(dynamic_output_path)
        tokenizer.save_pretrained('model/tokenizer')
        
        # output_folder = 'model/'  # Define the folder containing the model files

        # Define the name for the zip file
        zip_file_name = 'model.zip'

        # Create a BytesIO object to hold the zip file in memory
        zip_buffer = BytesIO()

        # Create a ZipFile object in write mode
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Iterate through all the files and subdirectories in the output folder
            for foldername, subfolders, filenames in os.walk(output_folder):
                for filename in filenames:
                    # Construct the full path of the file
                    file_path = os.path.join(foldername, filename)
                    # Add the file to the zip archive
                    zipf.write(file_path, os.path.relpath(file_path, output_folder))

        # Set the pointer of the buffer back to the beginning
        zip_buffer.seek(0)

        # Offer the zip file for download in Streamlit
        st.download_button(label='Download Model Zip', data=zip_buffer, file_name=zip_file_name, mime='application/zip')
        st.text('Training finished.')

    def tokenize_and_align_labels(example, label_all_tokens=True):
        example['tokens'] = [example['tokens']]
        example['ner_tags'] = [example['ner_tags']]
        tokenized_input = tokenizer(example['tokens'], truncation=True, padding=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(example['ner_tags']):
            word_ids = tokenized_input.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_input['labels'] = labels
        tokenized_input['tokens'] = example['tokens']
        tokenized_input['ner_tags'] = example['ner_tags']
        tokenized_input['id'] = example['id']
        return tokenized_input

    def compute_metrics(eval_preds):
        pred_logits, labels = eval_preds
        pred_logits = np.argmax(pred_logits, axis=2)
        predictions = [
            [label_list[pred] for (pred, label) in zip(prediction, label) if label != -100] 
            for prediction, label in zip(pred_logits, labels)
        ]
        true_labels = [
            [label_list[label] for (pred, label) in zip(prediction, label) if label != -100] 
            for prediction, label in zip(pred_logits, labels)
        ]
        results = datasets.load_metric('seqeval').compute(predictions=predictions, references=true_labels)
        return {
            "precision": results['overall_precision'],
            "recall": results['overall_recall'],
            "f1": results['overall_f1'],
            "accuracy": results['overall_accuracy'],
        }    
    
    def convert_data(input_data):
        output_data = {
            'id': [],
            'tokens': [],
            'ner_tags': [],
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for item in input_data:
            try:
                required_keys = ['tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels']
                if not all(key in item for key in required_keys):
                    print("Missing required keys in item:", item)
                output_data['id'].append(item['id'][0])
                output_data['tokens'].append(item['tokens'][0])
                output_data['ner_tags'].append(item['ner_tags'][0])
                output_data['input_ids'].append(item['input_ids'][0])
                output_data['token_type_ids'].append(item['token_type_ids'][0])
                output_data['attention_mask'].append(item['attention_mask'][0])
                output_data['labels'].append(item['labels'][0])
            except Exception as e:
                print("Error processing item:", e)
                continue  
        return output_data

    main()
else:
    st.write("Please upload a JSON file.")