import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


#  PART ONE
# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('../models/tokenization_bert/bert-base-uncased-vocab.txt')
# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer is a very good person yeah [SEP]"
# print('text = ', text)
tokenized_text = tokenizer.tokenize(text)
print('tokenized_text_1 = ', tokenized_text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'  # 'about'
# print('tokenized_text_2 = ', tokenized_text)
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', 'is', 'a', 'very', 'good', 'person', 'yeah', '[SEP]']


# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


#  PART TWO
# Load pre-trained model (weights)
# model = BertModel.from_pretrained('../models/modeling_bert/bert-base-uncased-pytorch_model.bin')
# model = BertModel.from_pretrained('../models/modeling_bert/bert-base-uncased-config.json')
modelConfig = BertConfig.from_pretrained('../models/modeling_bert/bert-base-uncased-config.json')
model = BertModel.from_pretrained('../models/modeling_bert/bert-base-uncased-pytorch_model.bin', config=modelConfig)

# Set the model in evaluation mode to desactivate the DropOut modules
# This is IMPORTANT to have reproductible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=input_masks)
    # PyTorch-Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0][:,0,:]
    print('encoded_layers.shape = ', encoded_layers.shape)
# # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
# assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)


# # PART THREE
# # Load pre-trained model (weights)
# # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# modelConfig = BertConfig.from_pretrained('../models/modeling_bert/bert-base-uncased-config.json')
# model = BertForMaskedLM.from_pretrained('../models/modeling_bert/bert-base-uncased-pytorch_model.bin', config=modelConfig)
# model.eval()

# # If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

# # Predict all tokens
# with torch.no_grad():
#     outputs = model(tokens_tensor, token_type_ids=segments_tensors)
#     predictions = outputs[0]

# # confirm we were able to predict 'henson'
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# # assert predicted_token == 'henson'
# print('predicted_token = ', predicted_token)
# assert predicted_token == 'henson'