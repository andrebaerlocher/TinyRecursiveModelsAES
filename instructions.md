# Instruction
You are to create a Tiny Recursive Model using the ASAPPP dataset. This code is a fork of the original repo for TinyRecursiveModels. You need to adapt it so that it can be used for training on a MacBook Pro M1 with 16GB RAM.

# Requirements
- Training on a MacBook Pro M1 with 16GB RAM

# Training Data
The training data will be used from Huggingface. For each prompt/essay set, set split to train and test.

## Prompts 1 - 2
```
from datasets import load_dataset

ds = load_dataset("llm-aes/asappp-1-2-original")
```

### Column titles, data types, min, max
essay_set
int64
1
2

essay
stringlengths
47
6.1k

rater1_domain1
int64
1
6

rater2_domain1
int64
1
6

domain1_score
int64
1
12

rubrics
stringclasses
2 values

prompt
stringclasses
2 values

content
int64
1
6

organization
int64
1
6

word_choice
int64
1
6

sentence_fluency
int64
1
6

conventions
int64
1
6

__index_level_0__
int64
0
3.58k


## Prompts 3 - 6
```
from datasets import load_dataset

ds = load_dataset("llm-aes/asappp-3-6-original")
```
### Column titles, data types, min, max
Essay_ID
int64
5.98k
16.6k

essay_set
int64
3
6

essay
stringlengths
8
2.72k

rater1_domain1
int64
0
4

rater2_domain1
int64
0
4

domain1_score
int64
0
4

rubrics
stringclasses
2 values

prompt
stringclasses
4 values

Content
int64
0
4

Prompt_Adherence
int64
0
4

Language
int64
0
4

Narrativity
int64
0
4

## Prompt 7
```
from datasets import load_dataset

ds = load_dataset("llm-aes/asap-7-original")
```

### Column titles, data types, min, max
essay_id
int64
17.8k
19.6k

essay_set
int64
7
7

essay
stringlengths
23
3.26k

rater1_domain1
int64
0
12

rater2_domain1
int64
0
12

domain1_score
int64
2
24

rater1_trait1
float64
0
3

rater1_trait2
float64
0
3

rater1_trait3
float64
0
3

rater1_trait4
float64
0
3

rater2_trait1
float64
0
3

rater2_trait2
float64
0
3

rater2_trait3
float64
0
3

rater2_trait4
float64
0
3

rubrics
stringclasses
1 value

prompt
stringclasses
1 value

__index_level_0__
int64
10.7k
12.3k


## Infos about using 🤗 Datasets
```
# Using 🤗 Datasets

Once you've found an interesting dataset on the Hugging Face Hub, you can load the dataset using 🤗 Datasets. You can click on the [**Use this dataset** button](https://huggingface.co/datasets/nyu-mll/glue?library=datasets) to copy the code to load a dataset.

First you need to [Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login), for example using:

```
hf auth login
```

And then you can load a dataset from the Hugging Face Hub using

```python
from datasets import load_dataset

dataset = load_dataset("username/my_dataset")

# or load the separate splits if the dataset has train/validation/test splits
train_dataset = load_dataset("username/my_dataset", split="train")
valid_dataset = load_dataset("username/my_dataset", split="validation")
test_dataset  = load_dataset("username/my_dataset", split="test")
```

You can also upload datasets to the Hugging Face Hub:

```python
my_new_dataset.push_to_hub("username/my_new_dataset")
```

This creates a dataset repository `username/my_new_dataset` containing your Dataset in Parquet format, that you can reload later.

For more information about using 🤗 Datasets, check out the [tutorials](/docs/datasets/tutorial) and [how-to guides](/docs/datasets/how_to) available in the 🤗 Datasets documentation.


<EditOnGithub source="https://github.com/huggingface/hub-docs/blob/main/docs/hub/datasets-usage.md" />
```
