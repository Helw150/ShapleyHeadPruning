import os
import random
import sys
import signal
import time
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
from datasets import load_dataset, load_metric, concatenate_datasets
from learning_utils import get_free_gpu

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import torch
from torch.utils.data import DataLoader

from collections import defaultdict
import json
import scipy.stats as st

fake_model = None


def signal_handler(signal, frame):
    print("You pressed Ctrl+C!")
    print(signal)  # Value is 2 for CTRL + C
    print(frame)  # Where your execution of program is at moment - the Line Number
    fake_model.finish()
    sys.exit(0)


# Assign Handler Function
signal.signal(signal.SIGINT, signal_handler)


def bernstein(sample):
    if len(sample) < 2:
        return -1, 1
    mean = np.mean(sample)
    variance = np.std(sample)
    delta = 0.1
    R = 1
    bern_bound = (variance * np.sqrt((2 * np.log(3 / delta))) / len(sample)) + (
        (3 * R * np.log(3 / delta)) / len(sample)
    )
    return mean - bern_bound, mean + bern_bound


class MaskModel(torch.nn.Module):
    def __init__(self, real_model, head_mask, lang):
        super(MaskModel, self).__init__()
        self.contribs = defaultdict(self.construct_array)
        self.counter = 0
        self.prev = 1.0
        self.lang = lang
        self.real_model = real_model
        self.head_mask = head_mask
        self.true_prev = True
        self.prev_mask = torch.ones_like(head_mask).flatten()
        self.u = torch.zeros_like(head_mask).flatten()
        self.tracker = open(lang + "_tracker.txt", "a")
        self.sample_limit = 5000

    def construct_array(self):
        return []

    def track(self, head, acc):
        if head != None:
            self.contribs[head].append(self.prev - acc)
        else:
            self.baseline = acc
        self.prev = acc
        if self.counter % 100 == 0:
            self.tracker.write(str(self.u.sum()) + "-" + str(self.counter) + "\n")
            self.tracker.flush()
        self.counter += 1

    def finish(self):
        self.tracker.write("Contribution Arrays")
        self.tracker.write(json.dumps(self.contribs))
        self.tracker.close()

    def set_mask(self, mask):
        mask = mask.reshape(12, 12)
        self.head_mask = mask

    def get_head(self, mask):
        head = (mask.reshape(-1) != self.prev_mask.reshape(-1)).nonzero(as_tuple=True)[
            0
        ]
        head = head.detach().cpu().tolist()[0]
        self.prev_mask = mask
        return head

    def active(self, head):
        def active_memo(head):
            contribs = np.array(self.contribs[head])
            lower, upper = bernstein(contribs)
            if lower > -0.01:
                return False
            elif len(contribs) > self.sample_limit:
                return False
            return True

        stored = self.u[head]
        if head == None:
            return True
        elif stored == 1:
            return False
        else:
            is_active = active_memo(head)
            if is_active:
                return True
            else:
                self.u[head] = 1
                return False

    def reset(self):
        print("RESET")
        self.true_prev = True
        self.prev_mask = torch.ones_like(self.prev_mask).flatten()
        self.head_mask = torch.ones_like(self.head_mask)
        self.prev = self.baseline

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.real_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_mask=self.head_mask,
        )


# Invert Mask Order and Accuracy to do top down marginal contribution rather than bottom up
def attribute_factory(model):
    def attribute(mask):
        mask = mask.flatten()
        if mask.sum() == 1:
            model.reset()
        mask = mask == 0  # invert mask order
        if not mask.sum() == 144:
            head = model.get_head(mask)
        else:
            head = None
        if not model.active(head) or mask.sum() <= 72:
            acc = model.prev
            model.true_prev = False
        else:
            if not model.true_prev:
                mask_copy = mask.clone()
                mask_copy[head] = 1
                model.set_mask(mask_copy)
                model.prev = trainer.evaluate()["eval_accuracy"]
            model.set_mask(mask)
            acc = trainer.evaluate()["eval_accuracy"]
            model.track(head, acc)
            model.true_prev = True
        acc = -1 * acc
        return acc

    return attribute


os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())
reserve = torch.tensor(1)
reserve.to("cuda:0")

# The paper uses a custom finetuned version of XLM-R Base, but for convenience
# I shifted to an available HuggingFace XNLI model
model = AutoModelForSequenceClassification.from_pretrained(
    "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
)
tokenizer = AutoTokenizer.from_pretrained(
    "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
)


def preprocess_function(examples):
    # Tokenize the texts
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )


metric = load_metric("xnli")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)


from captum.attr import ShapleyValueSampling
from transformers.utils import logging
import json

logging.set_verbosity_error()
langs = sys.argv[1].split(",")
for eval_lang in langs:
    mask = torch.ones((1, 144)).to("cuda:0")
    fake_model = MaskModel(model, mask, eval_lang)
    args = transformers.TrainingArguments(
        "/data/wheld3/tmp", per_device_eval_batch_size=1024
    )
    eval_dataset = load_dataset(
        "xnli",
        eval_lang,
        split="validation",
    )
    label_list = eval_dataset.features["label"].names
    eval_dataset = eval_dataset.shuffle().select(range(1024))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on validation dataset",
    )
    trainer = Trainer(
        model=fake_model,
        args=args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    attribute = attribute_factory(fake_model, 0.50)

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            torch.ones((1, 144)).to("cuda:0"), n_samples=3000, show_progress=True
        )
    fake_model.finish()
