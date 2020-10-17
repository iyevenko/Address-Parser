from address_parser.model import model_fn, train_model
from address_parser.dataset import input_fn, get_saved_tokenizer


tokenizer = get_saved_tokenizer()
model = model_fn(embedding_dim=256, GRU_units=128,  vocab_size=len(tokenizer.word_index)+1, show_summary=True)
dataset = input_fn(batch_size=128, dataset_size=1e6)

model_filename = 'model1.ckpt'
train_model(model, dataset, epochs=10, model_filename=model_filename)
