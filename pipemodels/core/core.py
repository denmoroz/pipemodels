class BaseFeaturesExtractor(object):

    def extract(self, object_ids, *args, **kwargs):
        raise NotImplementedError()


class BaseEstimator(object):

    def fit(self, train_data, *args, **kwargs):
        return self

    def predict_one(self, features):
        raise NotImplementedError()

    def predict_all(self, features_iter):
        return [self.predict_one(features) for features in features_iter]


class ModelEvaluator(object):

    def __init__(self, fitted_model, metrics):
        self.fitted_model = fitted_model
        self.metrics = metrics

    def evaluate(self, test_sample):
        evaluation_result = []

        test_features, test_labels = [], []
        for observation in test_sample:
            test_features.append(observation['features'])
            test_labels.append(observation['label'])

        predicted_labels = self.fitted_model.predict_all(test_features)

        for metric in self.metrics:
            metric_value = metric(test_labels, predicted_labels)
            evaluation_result.append(metric_value)

        return evaluation_result
