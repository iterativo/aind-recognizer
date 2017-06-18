import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("-inf"), None

        # DONE implement model selection based on BIC scores
        for n in range(self.min_n_components, self.max_n_components + 1):
            # Guard for exception thrown by hmmlearn bug as explained here:
            # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/4
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/3
                # Initial state occupation probabilities = numStates
                # Transition probabilities = numStates*(numStates - 1)
                # Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
                # Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
                p = n + (n * (n - 1)) + (n * self.X.shape[1] * 2)

                logN = np.log(self.X.shape[0])
                bic = -2 * logL + p * logN

                if bic > best_score:
                    best_score = bic
                    best_model = model

            except ValueError:
                continue

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    More on the meaning of this formula here:
    https://discussions.udacity.com/t/dic-score-calculation/238907
    - log(P(X(i)) is simply the log likelyhood (score) that is returned from
      the model by calling model.score.
    - log(P(X(j)); where j != i is just the model score when evaluating the
      model on all words other than the word for which we are training this
      particular model. 1/(M-1)SUM(log(P(X(all but i)) is simply the average
      of the model scores for all other words.
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on DIC scores
        best_score, best_model = float("-inf"), None

        for n in range(self.min_n_components, self.max_n_components + 1):
            # Guard for exception thrown by hmmlearn bug as explained here:
            # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/4
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                other_logL_lst = []

                for other_word in (w for w in self.hwords if w != self.this_word):
                    x, lengths = self.hwords[other_word]
                    other_logL_lst.append(model.score(x, lengths))

                other_logL = np.average(other_logL_lst)
                score = logL - other_logL

                if score > best_score:
                    best_score = score
                    best_model = model

            except ValueError:
                continue

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection using CV
        best_score, best_model = float("-inf"), None
        n_splits = 2  # could be injected

        # evaluate model for each n_components between min and max
        for n in range(self.min_n_components, self.max_n_components + 1):
            kfold = KFold(random_state=self.random_state, n_splits=n_splits)

            # calculate average score for all splits and update best model
            logL_lst = []
            model = None
            for train_index, test_index in kfold.split(self.sequences):
                # Guard for exception thrown by hmmlearn bug as explained here:
                # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/4
                try:
                    x_train, len_train = combine_sequences(train_index, self.sequences)
                    x_test, len_test = combine_sequences(test_index, self.sequences)

                    model = GaussianHMM(n_components=n, n_iter=1000).fit(x_train, len_train)
                    logL_lst.append(model.score(x_test, len_test))
                except ValueError:
                    break

            avg = np.average(logL_lst) if len(logL_lst) > 0 else float("-inf")

            if avg > best_score:
                best_score = avg
                best_model = model

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model
