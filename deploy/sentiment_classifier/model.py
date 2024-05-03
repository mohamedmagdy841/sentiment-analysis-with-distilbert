import json

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config["DISTILBERT_MODEL"])

        classifier = DistilBertForSequenceClassification.from_pretrained(config["DISTILBERT_MODEL"],
                                                                              num_labels=3)
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)


    def predict(self, text):
        encoded_text = self.tokenizer(text,
                                      truncation=True, padding="max_length",
                                      max_length=config["MAX_SEQUENCE_LEN"], return_tensors='pt').to(self.device)


        with torch.inference_mode():
            outputs = self.classifier(**encoded_text)
            preds_prob = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(preds_prob, dim=-1)

            predicted_class = preds.cpu().item()
            score = preds_prob.max()
        return (
            config["CLASS_NAMES"][predicted_class],
            score
        )

model = Model()


def get_model():
    return model
