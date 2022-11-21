text_roberta = dict(
    tokenizer = "roberta-base",
    vocab_size = 50265,
    input_text_embed_size = 768,
)

text_roberta_large = dict(
    tokenizer = "roberta-large",
    vocab_size = 50265,
    input_text_embed_size = 1024,
)

text_bert = dict(
    tokenizer = "bert-base-uncased",
    vocab_size = 30522,
    input_text_embed_size = 768,
)

text_dict = {"text_bert": text_bert,
             "text_roberta": text_roberta,
             "text_roberta_large": text_roberta_large,}
