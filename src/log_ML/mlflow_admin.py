import json

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.store.entities import PagedList
from loguru import logger

from src.log_ML.mlflow_log import get_metamodel_name_from_log_model


def mlflow_update_best_model(project_name: str,
                             stage: str = 'Staging',
                             best_metric_name: str = 'Dice'):

    # Get best run from all the runs so far
    best_run = get_best_run(project_name,
                            best_metric_name=best_metric_name)


    # Check the best metric(s) from the registered models
    register_best_run_as_best_registered_model = (
        get_best_registered_model(project_name=project_name,
                                  best_run=best_run,
                                  best_metric_name=best_metric_name))

    # Register the model from the best run from MLflow experiments,
    # if the best run is better than the best registered model
    if register_best_run_as_best_registered_model:
        logger.info('Register the best run as the best registered model')
        register_model_from_run(run=best_run,
                                stage=stage,
                                project_name=project_name)
    else:
        logger.info('Keeping the best registered model as the best model')


def get_best_registered_model(project_name: str,
                              best_run,
                              best_metric_name: str = 'Dice'):

    prev_best = best_run.data.metrics[best_metric_name]
    register_best_run_as_best_registered_model = False

    client = MlflowClient()
    rmodels = client.search_registered_models()
    for rmodel in rmodels:
        # TODO! you need to loop the prev_best to match this
        if project_name in rmodel.name:
            logger.info('registered model name "{}" '
                        '(nr of versions = {})'.format(rmodel.name, len(rmodel.latest_versions)))
            latest_ver = rmodel.latest_versions[0]
            run_id = latest_ver.run_id
            best_value = get_best_metric_of_run(run_id=run_id,
                                                best_metric_name=best_metric_name)

            if best_value is None or best_value < prev_best:
                logger.info('Best run is better than the best registered model')
                register_best_run_as_best_registered_model = True
            else:
                logger.info('Best registered model does not need to be updated')
                register_best_run_as_best_registered_model = False

    return register_best_run_as_best_registered_model


def get_best_metric_of_run(run_id: str,
                           best_metric_name: str = 'Dice'):

    run = mlflow.get_run(run_id)
    if best_metric_name in run.data.metrics:
        best_value = run.data.metrics[best_metric_name]
    else:
        logger.warning('It seems that you have never registered '
                       'a model with the metric = "{}" even computed'.format(best_metric_name))
        best_value = None

    return best_value


def register_model_from_run(run, stage: str = 'Staging',
                            project_name: str = 'segmentation-minivess'):

    # https://mlflow.org/docs/1.8.0/model-registry.html#mlflow-model-registry
    client = MlflowClient()

    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    log_model_dict = get_log_model_dict_from_run(run=run)
    metamodel_name = get_metamodel_name_from_log_model(log_model_dict)  # artifact_path

    # Registered model names you don't nececcsarily want to be as cryptic as the model log name
    # which comes from the hyperparameter sweep. In the end, you might want to have the best segmentor
    # model (or in general you want these names to be a lot more human-readable)
    reg_model_name = project_name

    logger.info('Register best model with the name = {}'.format(reg_model_name))
    model_uri = f"runs:/{run.info.run_id}/{reg_model_name}"
    reg = mlflow.register_model(model_uri, reg_model_name)

    # Set model version tag
    # mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST
    # client.set_model_version_tag(name=reg_model_name,
    #                              version=reg.version,
    #                              key="key",
    #                              value="value")

    # Set and delete aliases on models
    client.set_registered_model_alias(name=reg_model_name,
                                      alias="autoreregistered",
                                      version=reg.version)

    # Auto-stage
    transition_model_stage(name=reg_model_name,
                           version=reg.version,
                           stage=stage)

def transition_model_stage(name: str,
                           version: str,
                           stage: str = 'Staging'):
    # https://mlflow.org/docs/1.8.0/model-registry.html#transitioning-an-mlflow-models-stage
    logger.info('Transition model "{}" (v. {}) stage to {}'.format(name, version, stage))
    client = MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage
    )


def get_log_model_dict_from_run(run):

    run_id = run.info.run_id
    tags = run.data.tags
    log_model_histories_string: str = tags['mlflow.log-model.history']
    log_model = json.loads(log_model_histories_string)

    if len(log_model) == 1:
        log_model_dict: dict = log_model[0]
    else:
        raise NotImplementedError('Check why there are more entries or none?')

    return log_model_dict


def get_mlflow_ordering_direction(best_metric_name):

    if best_metric_name == 'Dice':
        metric_best = f'{best_metric_name} DESC'
    elif best_metric_name == 'Hausdorff':
        metric_best = f'{best_metric_name} ASC'
    else:
        raise NotImplementedError('Check the metric name!, best_metric_name = {}'.format(best_metric_name))

    return metric_best


def get_best_run(project_name: str,
                 best_metric_name: str = 'Dice'):

    metric_best = get_mlflow_ordering_direction(best_metric_name)
    try:
        runs = get_runs_of_experiment(project_name=project_name,
                                      metric_best=metric_best,
                                      max_results=1)
    except Exception as e:
        logger.error('Failed to get the runs from experiment = {}! e = {}'.format(project_name, e))
        raise IOError('Failed to get the runs from experiment = {}! e = {}'.format(project_name, e))

    if len(runs) > 0:
        best_run = runs[0]
        metric_name = metric_best.split(' ')[0]
        logger.info('Best run from MLflow Tracking experiments, '
                    '{} = {:.3f}'.format(metric_name, best_run.data.metrics[metric_name]))
    else:
        logger.warning('No runs returned!')
        best_run = None

    return best_run


def get_runs_of_experiment(project_name: str,
                           max_results: int = 3,
                           metric_best: str = 'Dice DESC') -> PagedList:

    # https://mlflow.org/docs/latest/search-runs.html#python
    order_by = ["metrics.{}".format(metric_best)]
    filter_string=""
    logger.info('Get a list of the best {} runs from experiment = {}'.format(max_results, project_name))
    logger.info('Ordering of model performance based on {}'.format(order_by))
    logger.info('Filtering string {}'.format(filter_string))
    logger.info('Returning only ACTIVE runs')
    # ADD info.status == 'FINISHED'
    runs = MlflowClient().search_runs(
        experiment_ids=get_current_id(project_name=project_name),
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=order_by,
    )

    return runs


def get_current_id(project_name: str):
    current_experiment = dict(mlflow.get_experiment_by_name(project_name))
    return current_experiment['experiment_id']