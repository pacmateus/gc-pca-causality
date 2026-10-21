import numpy as np

def parity(dt):
    r = []

    for i in range(0,len(dt)):
        if(np.random.randint(1,4) == 2):
            r.append((dt[i]+1)%2)
        else:
            r.append(dt[i]%2)

    return r


def create_mapping(k_len, y_len):
    """
    Create a mapping between keys and categories.

    Parameters:
        k_len (int): Length of the keys.
        y_len (int): Length of the categories.

    Returns:
        numpy.ndarray: Array representing the mapping.
    """
    mapping = np.array([]) 
    
    #domain is smaller than codomain
    if k_len < y_len:
        mapping = np.random.choice(np.arange(y_len), size=k_len, replace=False)
    else: #domain is equal or bigger than codomain
        new = np.random.choice(np.arange(y_len), size=y_len, replace=False)
        mapping = mapping.astype(new.dtype)
        mapping = np.concatenate((mapping, new))
        k_len -= y_len
        if k_len > 0: # if domain is still smaller than codomain after already selecting y_len values
            new = np.random.choice(np.arange(y_len), size=k_len, replace=True)
            mapping = mapping.astype(new.dtype)
            mapping = np.concatenate((mapping, new))
    
    return mapping


def map_dataset_independent(dataset, mapping, y_len, noise):
    """
    Map the dataset based on the provided mapping.

    Parameters:
        dataset (list): List of data points to be mapped.
        mapping (numpy.ndarray): Mapping between keys and categories.
        y (int): Length of the categories.
        noise (int): Level of noise to introduce (bigger value mean error is less likely to occur).

    Returns:
        list: Mapped dataset.
    """
    mapped_dataset = []
    for i in range(len(dataset)):
        #if there is noise
        if noise:
            if(np.random.randint(noise) == 1):
                error = np.random.randint(y_len)
                while error == mapping[dataset[i]]:
                    error = np.random.randint(y_len)
                mapped_dataset.append(error)
            else:
                mapped_dataset.append(mapping[dataset[i]])
        else: #noiseless case
            mapped_dataset.append(mapping[dataset[i]])

    return mapped_dataset


# Define a function to generate random probabilities for each value
def generate_random_probabilities(num_values):
    probabilities = np.random.rand(num_values)
    probabilities /= np.sum(probabilities)  # Normalize to ensure they sum up to 1
    return probabilities

def generate_noise_probabilities(x_values, y_values, mapping):
    # Generate random probabilities for each value, excluding the original value itself
    probabilities = {}
    for value in range(x_values):
        other_values = [v for v in range(y_values) if v != mapping[value]]
        prob_values = generate_random_probabilities(len(other_values))
        probabilities[value] = {'probabilities': prob_values, 'corresponding_values': other_values}
    return probabilities

# Define a function to select a new value y given the probabilities of the value x
def select_new_value(probabilities):
    corresponding_values = probabilities['corresponding_values']
    prob_values = probabilities['probabilities']
    selected_value = np.random.choice(corresponding_values, p=prob_values)
    return selected_value


def map_dataset_dependent(dataset, mapping, x_len,  y_len , noise):
    mapped_dataset = []
    probabilities = generate_noise_probabilities(x_len, y_len, mapping)

    # for value, data in probabilities.items():
    #     print("Value:", value)
    #     print("Probabilities:", data['probabilities'])
    #     print("Corresponding Values:", data['corresponding_values'])
    #     print()

    for i in range(len(dataset)):
        #if there is noise
        if noise:
            if(np.random.randint(noise) == 1):
                mapped_dataset.append(select_new_value(probabilities[dataset[i]]))
            else:
                mapped_dataset.append(mapping[dataset[i]])
        else: #noiseless case
            mapped_dataset.append(mapping[dataset[i]])

    return mapped_dataset



def createSynthetic(k_len, y_len, dataset, noise=0, noise_dependency='independent'):
    """
    Create synthetic dataset with mapping and optional noise.

    Parameters:
        k_len (int): Length of the keys.
        y_len (int): Length of the categories.
        dataset (list): List of data points.
        noise (int): Level of noise to introduce (default is 0).Bigger value mean error is less likely to occur.

    Returns:
        list: Synthetic dataset.
    """
    if noise_dependency == 'independent':
        mapping = create_mapping(k_len, y_len)  # Generate mapping between keys and categories
        mapped_dataset = map_dataset_independent(dataset, mapping, y_len, noise)  # Map the dataset based on the mapping
        
        return mapped_dataset
    
    elif noise_dependency == 'dependent':
        mapping = create_mapping(k_len, y_len)  # Generate mapping between keys and categories
        mapped_dataset = map_dataset_dependent(dataset, mapping, k_len, y_len, noise)  # Map the dataset based on the mapping
        return mapped_dataset
