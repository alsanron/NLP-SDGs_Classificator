from bert_serving.client import BertClient

# make a connection with the BERT server using it's ip address; do not give any ip if same computer
bc = BertClient()
# get the embedding
embedding = bc.encode(["I love data science and analytics vidhya."])
# check the shape of embedding, it should be 1x768
print(embedding.shape)