
def normalization_range(data):
    mean = data.mean()
    min = data.min()
    max = data.max()
    normalized_data = (data - mean) / (max - min)
    return normalized_data

def normalization_zscore(data):
    mean = data.mean()
    std = data.std()
    z_scores = (data - mean) / std
    return z_scores

