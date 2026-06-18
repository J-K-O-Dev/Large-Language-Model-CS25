### Week 4 Report: Fine-Tuning Pretrained Models

After spending Week 3 building a decoder-only Transformer, this week felt like stepping into the world where these architectures are actually used in practice. Instead of writing every layer by hand, I focused on fine-tuning pretrained models with the Hugging Face ecosystem. The main idea was simple but powerful: take a model that already knows a lot from large-scale pretraining, then adapt it to a more specific task.

This week had two tracks. First, I fine-tuned BERT for sentiment classification on the SST-2 dataset. Then I worked with GPT-2 as a causal language model and fine-tuned it on programming problems from MBPP. It was a nice shift from pure architecture-building to learning the real training workflow used around modern language models.

#### The Theory: Transfer Learning and Fine-Tuning

The biggest concept this week was **transfer learning**. In the earlier weeks, the models started from random weights and had to learn everything from scratch. With pretrained models, the base model already has useful language representations, so fine-tuning becomes the process of nudging those representations toward a particular task.

- **Pretraining vs Fine-Tuning:** Pretraining teaches a model broad language patterns using massive data. Fine-tuning is much more focused. We use a smaller, task-specific dataset and update the model so it becomes better at one kind of problem, like sentiment classification or code-style generation.

- **BERT for Classification:** BERT is an encoder-based model, which makes it strong for understanding tasks. For sentiment analysis, I used `bert-base-uncased` with a sequence classification head. Since SST-2 is a binary classification dataset, the model only needed two output labels: positive or negative.

- **GPT-2 for Causal Language Modeling:** GPT-2 is decoder-only, so it fits naturally with next-token prediction. For the coding experiment, I formatted each MBPP example as a problem statement followed by its solution. This turns the task into language modeling: given the problem prompt and previous code tokens, the model learns to predict the next token in the solution.

- **Tokenization and Padding:** A big practical lesson was that the model is only as happy as the data pipeline feeding it. BERT used its own tokenizer and dynamic padding through `DataCollatorWithPadding`, while GPT-2 needed its padding token set to the end-of-sequence token and used `DataCollatorForLanguageModeling` with `mlm=False`.

#### Implementation: Hugging Face in Practice

This week, most of the work happened inside two notebooks, and the Hugging Face libraries made the training pipeline much cleaner than writing everything manually.

- **Fine-Tuning BERT:** In `Copy of Fine_tune.ipynb`, I loaded `bert-base-uncased` using `AutoModelForSequenceClassification` with `num_labels=2`. The dataset came from `stanfordnlp/sst2`, which includes train, validation, and test splits. After tokenizing the sentences, I used `TrainingArguments`, `Trainer`, and `DataCollatorWithPadding` to handle the training loop. Finally, the trained model was saved as `./final_bert_sentiment_model`.

- **Fine-Tuning GPT-2:** In `Copy of GPT2.ipynb`, I loaded GPT-2 with `AutoModelForCausalLM` and used the GPT-2 tokenizer. The dataset was `google-research-datasets/mbpp`, and I wrote a formatting function that converted each example into a clean prompt:
  - a problem description inside a comment block
  - a solution section containing the reference code

- **Preparing the Code Dataset:** The MBPP examples were mapped into a single text field before tokenization. This made the dataset compatible with causal language modeling, where the model learns from one continuous sequence instead of separate input and label columns.

- **Trainer Workflow:** The `Trainer` API handled the repeated forward pass, loss calculation, backpropagation, and optimizer steps. Compared to the earlier weeks, where I had to manually think through every gradient and update, this felt much more like using a professional training pipeline. The GPT-2 coding model was saved as `./final_gpt2_coding_model`.

#### Key Takeaways

This week showed me a different side of LLM work. In Weeks 1 to 3, the focus was on understanding what is inside the model. Week 4 was about learning how to actually adapt powerful pretrained models to useful downstream tasks.

The BERT notebook made the fine-tuning workflow feel concrete: load a pretrained checkpoint, attach or initialize the right task head, tokenize the dataset properly, train, and save the result. The GPT-2 notebook was especially interesting because it connected directly to code generation. Formatting the MBPP data as "problem plus solution" made it clear how much prompt structure matters when training language models.

The biggest realization was that using pretrained models does not mean the hard parts disappear. The hard parts just move. Instead of deriving every gradient by hand, I had to think carefully about datasets, tokenizers, padding, labels, collators, training arguments, and model saving. It felt like moving one level up the stack, from building the engine to learning how to tune and drive it properly.

After this week, I feel much more comfortable with the practical side of modern NLP. Building a Transformer from scratch gave me the intuition for what these models are doing internally, and fine-tuning BERT and GPT-2 showed me how that knowledge plugs into real tools and real tasks. It was a satisfying bridge between theory and application.
