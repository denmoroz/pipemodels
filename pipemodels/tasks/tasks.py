import os
import pickle
import yaml

import luigi

from .. core import ModelEvaluator, BaseEstimator, BaseFeaturesExtractor

from .. utils.common import import_object, split_object_path
from .. utils.common import file_checksum, object_checksum


class ExtractFeatures(luigi.Task):
    base_path = luigi.Parameter(default='')

    input_file = luigi.Parameter()

    extractor_class = luigi.Parameter()
    extractor_params = luigi.DictParameter()

    def _read_lines(self):
        return [line.rstrip('\n') for line in open(self.input_file)]

    def run(self):
        objects_ids = self._read_lines()

        extractor_class = import_object(self.extractor_class)

        if not issubclass(extractor_class, BaseFeaturesExtractor):
            raise ValueError(
                '%s is not a subclass of BaseFeaturesExtractor' %
                extractor_class
            )

        extractor = extractor_class(**self.extractor_params)
        extracted_features = extractor.extract(objects_ids)

        with self.output().open('wb') as output_file:
            pickle.dump(extracted_features, file=output_file)

    def output(self):
        _, extractor_class_name = split_object_path(self.extractor_class)

        input_checksum = file_checksum(self.input_file)
        extractor_checksum = object_checksum(self.extractor_params)

        features_id = object_checksum(
            ''.join([extractor_checksum, input_checksum])
        )

        file_name = 'features-%s-%s.pickle' % (
            extractor_class_name, features_id
        )

        return luigi.LocalTarget(
            os.path.join(self.base_path, 'features', file_name),
            format=luigi.format.Gzip
        )


class TrainModel(luigi.Task):
    base_path = luigi.Parameter(default='')

    train_file = luigi.Parameter()

    extractor_class = luigi.Parameter()
    extractor_params = luigi.DictParameter()

    estimator_class = luigi.Parameter()
    estimator_params = luigi.DictParameter()

    def run(self):
        train_set_target = yield ExtractFeatures(
            base_path=self.base_path,
            input_file=self.train_file,
            extractor_class=self.extractor_class,
            extractor_params=self.extractor_params
        )

        model_class = import_object(self.estimator_class)

        if not issubclass(model_class, BaseEstimator):
            raise ValueError(
                '%s is not a subclass of BaseEstimator' % model_class
            )

        model = model_class(**self.estimator_params)

        with train_set_target.open('rb') as input_file:
            train_set = pickle.load(input_file)

        print('Training model on %d samples' % len(train_set))

        model.fit(train_set)

        with self.output().open('wb') as output_file:
            pickle.dump(model, file=output_file)

    def output(self):
        _, extractor_class_name = split_object_path(self.extractor_class)
        _, estimator_class_name = split_object_path(self.estimator_class)

        train_checksum = file_checksum(self.train_file)

        extractor_checksum = object_checksum(self.extractor_params)
        estimator_checksum = object_checksum(self.estimator_params)

        model_id = object_checksum(
            ''.join([
                extractor_checksum, estimator_checksum,
                train_checksum
            ])
        )

        file_name = 'model-%s-%s-%s.pickle' % (
            extractor_class_name, estimator_class_name, model_id
        )

        return luigi.LocalTarget(
            os.path.join(self.base_path, 'models', file_name),
            format=luigi.format.Gzip
        )


class EvaluateModel(luigi.Task):
    base_path = luigi.Parameter(default='')

    train_file = luigi.Parameter()
    test_file = luigi.Parameter()
    metrics = luigi.ListParameter()

    extractor_class = luigi.Parameter()
    extractor_params = luigi.DictParameter()

    estimator_class = luigi.Parameter()
    estimator_params = luigi.DictParameter()

    def run(self):
        model_target = yield TrainModel(
            base_path=self.base_path,
            train_file=self.train_file,
            extractor_class=self.extractor_class,
            extractor_params=self.extractor_params,
            estimator_class=self.estimator_class,
            estimator_params=self.estimator_params
        )

        test_set_target = yield ExtractFeatures(
            base_path=self.base_path,
            input_file=self.test_file,
            extractor_class=self.extractor_class,
            extractor_params=self.extractor_params
        )

        with model_target.open('rb') as model_file:
            fitted_model = pickle.load(model_file)

        with test_set_target.open('rb') as test_set_file:
            test_set = pickle.load(test_set_file)

        print('Testing model on %d samples' % len(test_set))

        metrics_objects = [import_object(metric) for metric in self.metrics]

        model_evaluator = ModelEvaluator(fitted_model, metrics_objects)
        evaluation_result = model_evaluator.evaluate(test_set)

        evaluation_result = dict(zip(self.metrics, evaluation_result))
        print('Testing result: %s' % str(evaluation_result))

        with self.output().open('wb') as result_file:
            pickle.dump(evaluation_result, result_file)

    def output(self):
        _, extractor_class_name = split_object_path(self.extractor_class)
        _, estimator_class_name = split_object_path(self.estimator_class)

        train_checksum = file_checksum(self.train_file)
        test_checksum = file_checksum(self.test_file)

        extractor_checksum = object_checksum(self.extractor_params)
        estimator_checksum = object_checksum(self.estimator_params)

        evaluation_id = object_checksum(
            ''.join([
                extractor_checksum, estimator_checksum,
                train_checksum, test_checksum
            ])
        )

        file_name = 'evaluation-%s-%s-%s.pickle' % (
            extractor_class_name, estimator_class_name, evaluation_id
        )

        return luigi.LocalTarget(
            os.path.join(self.base_path, 'evaluation', file_name),
            format=luigi.format.Gzip
        )


class CompareModels(luigi.Task):
    specification = luigi.Parameter()

    def _load_specification(self):
        with open(self.specification, 'r') as spec_file:
            return yaml.load(spec_file)

    def run(self):
        dependencies = []
        evaluation_results = []

        specificaton = self._load_specification()

        for extractor_name, estimator_name in specificaton['compare']:
            extractor_section = specificaton['extractors'][extractor_name]
            estimator_section = specificaton['estimators'][estimator_name]

            evaluation_task = EvaluateModel(
                base_path=specificaton.get('base_path', ''),
                train_file=specificaton['train_file'],
                test_file=specificaton['test_file'],
                metrics=specificaton['metrics'],
                extractor_class=extractor_section['class'],
                extractor_params=extractor_section['params'],
                estimator_class=estimator_section['class'],
                estimator_params=estimator_section['params']
            )

            dependencies.append(evaluation_task)

        yield dependencies

        for dependency in dependencies:
            with dependency.output().open('rb') as dependency_file:
                metrics_values = pickle.load(dependency_file)

                evaluation_results.append(metrics_values)
                print(metrics_values)
