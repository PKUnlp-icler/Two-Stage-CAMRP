from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
from relation_data_collator import DualCollatorForSentenceAndRelation
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.cuda.amp import autocast
from transformers.trainer_pt_utils import (
    nested_detach
)

import sys
model_path = sys.argv[1]

from transformers.models.bert.modeling_bert import BertForRelationSequenceClassification

# Label: BxNxN , -100 denotes pad
# Output: BxNxN , 0 
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
#model = BertForRelationSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=150)
model = BertForRelationSequenceClassification.from_pretrained(model_path, num_labels=150)


# for name,param in model.named_parameters():
#     print(name,param)

# print(model)

def load_dataset(node_path,relation_path):
    tokens = open(node_path,"r").readlines()
    tags = open(relation_path,"r").readlines()
    result = {"nodes":[],"relation":[]}

    for tok,tag in zip(tokens,tags):
        cur_nodes = [i[1] for i in eval(tok)]
        cur_relation = eval(tag)

        assert len(cur_nodes),len(cur_nodes) == cur_relation.shape 
    
        result["nodes"].append(cur_nodes)
        result["relation"].append(cur_relation)
        
    return Dataset.from_dict(result)

def load_inference_dataset(node_path):
    tokens = open(node_path,"r").readlines()
    result = {"nodes":[],"relation":[]}
    num_of_node_list = []

    for tok in tokens:
        cur_nodes = [i[1] if i[1]!='' else 'null' for i in eval(tok.strip()) ]
        temp_relations = [[0.0]*len(cur_nodes)]*len(cur_nodes)
        num_of_node_list.append(len(cur_nodes))
        result["nodes"].append(cur_nodes)
        result["relation"].append(temp_relations)
    return Dataset.from_dict(result), num_of_node_list

def restore_2drelation_from_1d_labels(nodes_num_list,lineried_labels):
    num_of_relation=[i*i for i in nodes_num_list]

    pred_each_amr = []
    #assert sum(num_of_relation)==lineried_labels.shape[0]
    print(sum(num_of_relation),lineried_labels.shape[0])


    start_index = 0
    for i,j in zip(num_of_relation,nodes_num_list):
        pred_each_amr.append(lineried_labels[start_index:start_index+i].reshape(j,j))
        start_index+=i
    
    return pred_each_amr

def labelsfile_to_relation(nodes_path,relation_path,relation_dict_path,out_path):
    relation_index={}
    with open(relation_dict_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            relation_index[i+1]=j.strip().split()[0]
    
    with open(nodes_path,"r") as f:
        nodes = f.readlines()
    
    with open(relation_path,"r") as f:
        relations = f.readlines()
    
    literal_relations_list=[]
    
    for nod,rel in zip(nodes,relations):
        current_literal_relation=[]
        cur_nodes = [i[0] for i in eval(nod)]
        cur_relation = eval(rel)

        for i in range(len(cur_relation)):
            for j in range(len(cur_relation)):
                if cur_relation[i][j] != 0:
                    current_literal_relation.append([cur_nodes[i],relation_index[cur_relation[i][j]],cur_nodes[j]])

        literal_relations_list.append(cur_relation)
    
    with open(out_path,"w") as f:
        for i in literal_relations_list:
            f.write(str(i)+"\n")





# temp_input = torch.randint(low=0,high=100,size=[10,256])
# print(temp_input)
# data_all = model(input_ids=temp_input)

def build_train_dev_dataset(train,dev):
    dev_dataset = load_dataset(dev+".relations_nodes",dev+".reverse_of.relations")
    train_dataset = load_dataset(train+".relations_nodes",train+".reverse_of.relations")
    return DatasetDict({"train":train_dataset,"dev":dev_dataset})

def build_train_dev_dataset_4level(train,dev):
    dev_dataset = load_dataset(dev+".4level.relations_nodes",dev+".4level.relations")
    train_dataset = load_dataset(train+".4level.relations_nodes",train+".4level.relations")
    return DatasetDict({"train":train_dataset,"dev":dev_dataset})

#dataset = build_train_dev_dataset("/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/dev","/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/dev")

#dataset = build_train_dev_dataset_4level("/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/train","/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/dev")

#inference_dataset, node_num_list =  load_inference_dataset("/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/output_for_relation_classification/output_nodes_align_0805.align_extra_nodes")

import sys

NODES_PATH=sys.argv[2]

inference_dataset, node_num_list =  load_inference_dataset(NODES_PATH)


LABELS_TEMP = 0

def preprocess(examples):
    # from text/relation matrix to tensors , no need to pad here. do padding in data collator
    tokenized_inputs = tokenizer(examples["nodes"], truncation=False, is_split_into_words=True)
    
    labels = []
    is_node_start = []
    for i, node in enumerate(examples[f"nodes"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(1)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        is_node_start.append(label_ids)
    
    if "relation" in examples.keys():
        tokenized_inputs["labels"] = examples["relation"]
    tokenized_inputs["is_node_start"] = is_node_start
    return tokenized_inputs

# tokenized_dataset = dataset.map(preprocess,batched=True)
inference_dataset = inference_dataset.map(preprocess,batched=True)

sentence_data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True)
full_collator = DualCollatorForSentenceAndRelation(sentence_data_collator)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro',labels=list(range(1,151)))
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class CAMRRelationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        relation = inputs.get("relation") # BxNxN
        node_start = inputs.get("is_node_start")
        # forward pass
        outputs = model(**inputs) # BxNxN
        logits = outputs.get("logits")
        node_for_relation = [[ j for j,value in enumerate(i) if value==1] for i in node_start['is_node_start']]
        useful_logits = torch.cat([l[n,:,:][:,n,:].view(-1,model.config.num_labels)  for l,n in zip(logits,node_for_relation)],dim=0)
        useful_relation = torch.cat([l[:len(n),:][:,:len(n)].reshape(-1)  for l,n in zip(relation,node_for_relation)],dim=0)
        
        weighted = False
        focal_loss = False
        label_smoothing_factor = 0.1
        if weighted:
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.1]+[1.]*(model.config.num_labels-1)).cuda())
        elif focal_loss:
            pass
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        loss = loss_fct(useful_logits,useful_relation)
        return (loss, outputs) if return_outputs else loss

    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]


        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                if self.use_amp:
                    with autocast():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                else:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                        outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            #HACK reshape logits and labels
            relation = inputs.get("relation") # BxNxN
            node_start = inputs.get("is_node_start")
            node_for_relation = [[ j for j,value in enumerate(i) if value==1] for i in node_start['is_node_start']]
            logits = torch.cat([l[n,:,:][:,n,:].view(-1,model.config.num_labels)  for l,n in zip(logits,node_for_relation)],dim=0)
            labels = torch.cat([l[:len(n),:][:,:len(n)].reshape(-1)  for l,n in zip(relation,node_for_relation)],dim=0)

            if labels.shape[0]==sum([len(i)*len(i) for i in node_for_relation]):
                pass
            else:
                raise("no")

        return (loss, logits, labels)


OUT_DIR="./models/0811/7e-5_batch50_wd0.01_3level_relation_reverse-of_lstm_d0.2_no_lastlayer_dropout"

training_args = TrainingArguments(
    output_dir=OUT_DIR,    
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    learning_rate=7e-5,
    warmup_ratio=0.01,
    lr_scheduler_type="linear",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=5,
    num_train_epochs=200,
    weight_decay=0.01,
    logging_steps=100,
    label_names=["relation"],
    fp16=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    disable_tqdm=False,
    save_total_limit=3,
    seed=42,
    load_best_model_at_end=True
)

print(training_args)

trainer = CAMRRelationTrainer(
    model=model,
    args=training_args,
    train_dataset=inference_dataset,
    eval_dataset=inference_dataset,
    tokenizer=tokenizer,
    data_collator=full_collator,
    compute_metrics=compute_metrics

)

# trainer.train()
logtis,labels,scores = trainer.predict(inference_dataset)
pred_labels = logtis.argmax(-1)
result = restore_2drelation_from_1d_labels(node_num_list,pred_labels)

# with open(OUT_DIR+"/relation_relation_with_v2_wsd_7classes.txt","w") as f:
#     for i in result:
#         f.write(str(i.tolist())+"\n")

with open(NODES_PATH+".relation","w") as f:
    for i in result:
        f.write(str(i.tolist())+"\n")
