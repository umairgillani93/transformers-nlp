from transformers import pipeline

classifier = pipeline('sentiment-analysis')

print(classifier('this is  a nice lecture, but I am still boring'))
