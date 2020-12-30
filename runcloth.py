'''
nohup python -u finetune.py 
--task_name cloth 
--model_name_or_path google/electra-large-discriminator 
--do_train --do_eval --learning_rate 1e-5 --num_train_epochs 3 
--max_seq_length 512 --output_dir ../model/electra153/ 
--per_gpu_eval_batch_size=16 --per_device_train_batch_size=16 
--gradient_accumulation_steps 2 --overwrite_output > e153run.log 2>&1
'''
import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputExample:
    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]

@dataclass(frozen=True)
class InputFeatures:
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        features: List[InputFeatures]
        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()# 获取处理器

            #文件名拼接
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )

            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir+ 'dev/')
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir+ 'test/')
                    else:
                        examples = processor.get_train_examples(data_dir+ 'train/')
                    
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)
        
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

class DataProcessor:
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class ClothProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        return ["A", "B", "C", "D"]
    
    def _read_json(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['article'], data['options'], data['answers'],data['maskPredict'],data['summary']
    
    def _get_feature(self, i, article, maskPredict):
        count = 0
        copy = article
        while re.search('_',copy):
            if count==i: # 当前空
                copy = copy.replace('_', '['+maskPredict[count]+']',1)
            else:
                copy = copy.replace('_', maskPredict[count],1)
            count+=1
        
        #找到空并替换回_
        front = re.search(r'\[.*?]',copy).span()[0]
        end = re.search(r'\[.*?]',copy).span()[1]
        copy = copy[0:front]+'_'+copy[end:]

        #向前找两句
        pos = front
        i = 0
        while i<3 and pos>=0:
            if copy[pos]=='.':
                i+=1
            pos-=1
        start = pos

        #向后找两句
        pos = front+1
        i = 0
        while i<3 and pos<len(copy):
            if copy[pos]=='.':
                i+=1
            pos+=1
        back = pos

        # print(copy[start+2:back])
        # print(copy)
        return copy[start+2:back]

    # 文件夹 文件列表 训练类型
    def _create_examples(self, dir, set_type):
        examples = []
        filelist = os.listdir(dir)
        for data in filelist:
            article, options, answers, maskPredict,summary = self._read_json(dir+data)

            for i in range(len(options)):
                context = self._get_feature(i, article, maskPredict)
                examples.append(
                    InputExample(
                        example_id=data.split('.')[0]+'_'+str(i),
                        question=summary,
                        contexts=[context,context,context,context],
                        endings=[options[i][0],options[i][1],options[i][2],options[i][3]],
                        label=answers[i]
                    )
                )
        return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    return features

processors = {"cloth": ClothProcessor}