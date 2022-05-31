export OPENAI_API_KEY=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
openai api fine_tunes.create -t gpt3_train.jsonl -v gpt3_test.jsonl -m ada --n_epochs=1 --batch_size=256 --suffix=hint
openai wandb sync
