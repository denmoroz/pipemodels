def wrap_univariate_metric(univariate_metric, label_processor):

    def wrapper(test_labels, predicted_labels):
        test_values = list(map(label_processor, test_labels))
        predicted_values = list(map(label_processor, predicted_labels))

        return univariate_metric(test_values, predicted_values)

    return wrapper
