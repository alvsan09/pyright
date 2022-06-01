# Resolve a frequency dictionary, filtering out special words and sorted
def resolve_freq_dict(model, context: tuple[str]) -> dict:
    if len(context) == 0:
        raise Exception('Invalid Empty context')
    
    special_words = ('<s>', '</s>', 'BEGINNINGOFFILE', 'ENDOFFILE')
    frequency = dict()
    
    # Fetch a FreqDist object for tokens following the given context 
    freq_dist = model.context_counts(context)
    for key, value in freq_dist.items():
        if (key not in special_words):
            frequency[key] = value

    # Ordered by frequency value descending
    return {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}

# Obtaining the scores not yet sorted
def resolve_scores(model, context: tuple[str]):
    scores = dict()
    freq_dict = resolve_freq_dict(model, context)
    for key, _ in freq_dict.items():
        scores[key] = model.logscore(key, list(context))
    return scores

# Function to rank tokens that follow a given context
# context is represented by a tuple of tokens e.g. ('math', '.')
def rank(model, context: tuple[str], n = -1):
    scores = resolve_scores(model, context)

    chunk_size = n if n > -1 else len(scores)

    # Reordering by score, descending
    return {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:chunk_size]}

# Provide the top 'n' tokens that follow the context
# ordered by score (descending)
def top(model, context: tuple[str], n = -1) -> list:
    return list(rank(model, context, n).keys())
