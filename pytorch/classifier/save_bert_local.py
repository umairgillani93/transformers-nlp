from transformers import BertModel, BertTokenizerFast

bert = BertModel.from_pretrained('bert-base-uncase')
tokenizer = BertTokenizerFast('bert-base-uncase')

bert.save_pretrained('/home/tahirshah/bert_pretrained_weights')
tokenizer.save_pretrained('/home/tahirshah/bert_pretrained_weights')

print('done saving, file path -> /home/tahirshah/bert_pretrained_weights')
