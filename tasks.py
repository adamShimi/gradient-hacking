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

  def flip(x):
    if x == 0:
      return 1
    else:
      return 0

  training = rec_bitflip(nb_dim)
  labels = training[::-1]
  # To clean for bigger datasets.
  training = [[l] for l in training]
  labels = [[l] for l in labels]
  return (training,labels)
