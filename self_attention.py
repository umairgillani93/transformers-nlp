import numpy as np 

def _softmax(x):
  '''
  implements the softmax function
  '''
  return np.exp(x) / sum(np.exp(x))
def mask_func(vector):
  return vector

def self_attention(q,k,v,scaler):
  ''' returns the self-attention score 
  of query, key and value vectors '''
  
  query_key_vector = np.matmul(q, k)

  # multiply with scaler
  scaled_output = query_key_vector * scaler

  # wrap mask function
  result = mask_func(scaled_output)

  # apply softmax distribution
  result = _softmax(result)

  # return final result
  final_result = np.matmul(result, v)

  return final_result

if __name__ == '__main__':
  q = np.random.rand(1,3)
  k = np.random.rand(1,3).T
  v = np.random.rand(1,3)

  print(self_attention(q,k,v, 1/5))
  
