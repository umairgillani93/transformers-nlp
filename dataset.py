from datasets import load_dataset

rd = load_dataset('glue', 'mrpc')

#print(raw_dataset)

rtd = rd['train'][0]
print(rtd)
print([k for k in rtd.keys()])

