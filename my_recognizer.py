import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

    :param models: dict of trained models
     {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list)  as probabilities, guesses
     both lists are ordered by the test set word_id
     probabilities is a list of dictionaries where each key a word and value is Log Liklihood
         [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
          {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
          ]
     guesses is a list of the best guess words ordered by the test set word_id
         ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # DONE implement the recognizer
    for _, (X, lengths) in test_set.get_all_Xlengths().items():
        probs = {}
        for word, model in models.items():
            try:
                probs[word] = model.score(X, lengths)
            except ValueError:
                probs[word] = float("-inf")

        guess = max(probs, key=probs.get)
        guesses.append(guess)
        probabilities.append(probs)

    # return probabilities, guesses
    return probabilities, guesses
