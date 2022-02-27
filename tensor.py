import numpy as np 
import random

def tensor(txt_arrays):
  ids = np.zeros(shape = (len(txt_arrays), len(max(txt_arrays))))
  for row in range(len(ids)):
    for col in range(len(ids[0])):
      ids[row][col] = random.randint(1, 100)
  print('ids: {}'.format(ids))

arrays = [
    ['this', 'is', 'first', 'array'],
    ['smaller', 'one']
    ]
print(tensor(arrays))

