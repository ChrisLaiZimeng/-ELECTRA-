import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from runcloth import MultipleChoiceDataset, Split, processors

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    print('preds:')
    print(preds)
    print('labels:')
    print(labels)
    return (preds == labels).mean()

@dataclass
class ModelArguments:
    # 模型
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataTrainingArguments:
    # task_name swag
    task_name: str = field(
      default = 'cloth',
      metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())}
    )
    # data_dir swag/
    data_dir: str = field(
        default='../midData/',
        metadata={"help": "Should contain the data files for the task."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def main():
    '''
        HfArgumentParser(prog='run_multiple_choice.py', usage=None, description=None, 
        formatter_class=<class 'argparse.HelpFormatter'>, conflict_handler='error', add_help=True) 

        ModelArguments(model_name_or_path='google/electra-base-discriminator', config_name=None, 
        tokenizer_name=None, cache_dir=None) 

        DataTrainingArguments(task_name='swag', data_dir='/home/laizimeng/dataset/swag/', max_seq_length=80, 
        overwrite_cache=False) 

        TrainingArguments(output_dir='/home/laizimeng/output/test/', overwrite_output_dir=True, 
        do_train=True, do_eval=True, do_predict=False, evaluate_during_training=False, 
        evaluation_strategy=<EvaluationStrategy.NO: 'no'>, prediction_loss_only=False, 
        per_device_train_batch_size=16, per_device_eval_batch_size=8, per_gpu_train_batch_size=None, 
        per_gpu_eval_batch_size=16, gradient_accumulation_steps=2, eval_accumulation_steps=None, l
        earning_rate=5e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, 
        max_grad_norm=1.0, num_train_epochs=3.0, max_steps=-1, warmup_steps=0, 
        logging_dir='runs/Dec04_23-48-18_gpu03', logging_first_step=False, logging_steps=500, save_steps=500, 
        save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level='O1', local_rank=-1, 
        tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, 
        dataloader_num_workers=0, past_index=-1, run_name='/home/laizimeng/output/test/', disable_tqdm=False, 
        remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, 
        greater_is_better=None)
    '''
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 输出文件夹存在抛出错误
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]() #数据处理器
        label_list = processor.get_labels() # label ["A", "B", "C", "D"]
        num_labels = len(label_list) #需要这个参数config
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    
    # 参数
    # config_name=None cache_dir=None
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        print(result)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    return results

if __name__ == "__main__":
    main()