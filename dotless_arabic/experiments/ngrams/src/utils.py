import psutil


def estimate_memory_to_use_by_lm_modeler(verbose=True, margin=5):
    """
    this funciton will estimate the memory to use by LM
    it calculates the current usage of memory adding some margin
    then return the remaining percentage of memory to be used
    """
    # get the current memory used
    current_memory_usage = psutil.virtual_memory().percent
    # add some margin
    current_memory_usage += margin
    memory_to_use = int(100 - current_memory_usage)
    if verbose:
        print()
        print("#" * 80)
        print("Estimating the LM model using:", f"{memory_to_use}% of memory.")
        print("#" * 80)
        print()
    return memory_to_use


def extract_ngram_counts(arpa_path, max_ngram_only=True):
    ngram_counts = {}
    for line in open(arpa_path):
        if not line.strip():
            break
        if "=" not in line:
            continue
        ngram, counts = line.split("=")
        ngram = int("".join(c for c in ngram if c.isdigit()).strip())
        ngram_counts[ngram] = int(counts.strip())
    if max_ngram_only:
        max_ngram = max(ngram_counts.items(), key=lambda item: item[0])
        return max_ngram[1]  # return only the value as the key is the max ngram
    return ngram_counts
