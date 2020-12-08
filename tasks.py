import itertools

# Returns iterators on the list of combinations tuples
def bitflip(nb_dim, size_dataset):
  def rec_bitflip(nb_dim):
    if nb_dim == 0:
      return []
    if nb_dim == 1:
      return [[0],[1]]
    else:
      next_bitflip = rec_bitflip(nb_dim-1)
      return [[0] + x for x in next_bitflip] \
             + [[1] + x for x in next_bitflip]

  training = rec_bitflip(nb_dim)
  labels = training[::-1]
  training = itertools.combinations(training,size_dataset)
  labels = itertools.combinations(labels,size_dataset)
  return (training,labels)
