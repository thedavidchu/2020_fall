import numpy as np

# N.B. The code to run this is in PS1_10a_d.py.
# Change the variable 'part_e' (at the top) to True.
# It will call this function.
# The file just extracts the words and creates the f_term(t, d) matrix.
# The rest of the code is identical to the first.


def generate_frequency_vector():
    """
    Read the text document and
    :return:
    """
    # Read wordVecArticles.txt
    with open('wordVecArticles.txt') as f:
        raw_text = f.read()

    # Create set of words
    all_words = raw_text.replace('\n', ' ').split(' ')
    W = set()
    W.update(all_words)
    W = sorted(W)

    # Populate article vectors with keys
    articles = raw_text.split('\n')
    f = [{word: 0 for word in W} for i in range(len(articles))]

    # Populate article vector with values
    for i, article in enumerate(articles):
        words = article.split(' ')
        for word in words:
            f[i][word] += 1

    # Convert to np.array
    f = np.array([list(dic.values()) for dic in f])

    return f.T