# Returns iterators on the list of combinations tuples
def bitflip(nb_dim, patterns):
  def rec_bitflip(nb_dim):
    if nb_dim == 0:
      return []
    if nb_dim == 1:
      return [[0],[1]]
    else:
      next_bitflip = rec_bitflip(nb_dim-1)
      return [[0] + x for x in next_bitflip] \
             + [[1] + x for x in next_bitflip]

  def inverse(bit):
    if bit == 0:
      return 1
    else:
      return 0


  training = rec_bitflip(nb_dim)
  training = [[training[i] for i in pattern] for pattern in patterns]
  labels = list(map(lambda x: [inverse(y) for y in x],training))
  return (training,labels)
