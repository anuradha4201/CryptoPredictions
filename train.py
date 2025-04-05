import logging
import os

import hydra
from omegaconf import DictConfig
from models import MODELS
from data_loader import get_dataset
from factory.trainer import Trainer
from factory.evaluator import Evaluator
from factory.profit_calculator import ProfitCalculator
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH

from utils.reporter import Reporter
from data_loader.creator import create_dataset, preprocess


logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    elif cfg.load_path is not None:
        dataset_ = pd.read_csv(cfg.load_path)
        if 'Date' not in dataset_.keys():
            dataset_.rename(columns={'timestamp': 'Date'}, inplace=True)
        if 'High' not in dataset_.keys():
            dataset_.rename(columns={'high': 'High'}, inplace=True)
        if 'Low' not in dataset_.keys():
            dataset_.rename(columns={'low': 'Low'}, inplace=True)

        dataset, profit_calculator = preprocess(dataset_, cfg, logger)

    elif cfg.model is not None:
        dataset, profit_calculator = get_dataset(cfg.dataset_loader.name, cfg.dataset_loader.train_start_date,
                              cfg.dataset_loader.valid_end_date, cfg)

    cfg.save_dir = os.getcwd()
    reporter = Reporter(cfg)
    reporter.setup_saving_dirs(cfg.save_dir)
    model = MODELS[cfg.model.type](cfg.model)

    dataset_for_profit = dataset.copy()
    dataset_for_profit.drop(['prediction'], axis=1, inplace=True)
    dataset.drop(['predicted_high', 'predicted_low'], axis=1, inplace=True)
    if cfg.validation_method == 'simple':
        train_dataset = dataset[
            (dataset['Date'] > cfg.dataset_loader.train_start_date) & (
                        dataset['Date'] < cfg.dataset_loader.train_end_date)]
        print("Train dataset:", len(train_dataset))
        train_dataset = create_feature_dataset(train_dataset)
        X, y = create_lstm_dataset(train_dataset)
        train_dataset.to_csv("/content/train_dataset.csv", index=False)
        valid_dataset = dataset[
            (dataset['Date'] > cfg.dataset_loader.valid_start_date) & (
                        dataset['Date'] < cfg.dataset_loader.valid_end_date)]
        Trainer(cfg, X, y, None, model).train()
        mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

    elif cfg.validation_method == 'cross_validation':
        n_split = 3
        tscv = TimeSeriesSplit(n_splits=n_split)

        for train_index, test_index in tscv.split(dataset):
            train_dataset, valid_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
            Trainer(cfg, train_dataset, None, model).train()
            mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

        reporter.add_average()

    ProfitCalculator(cfg, dataset_for_profit, profit_calculator, mean_prediction, reporter).profit_calculator()

    reporter.print_pretty_metrics(logger)
    reporter.save_metrics()

def create_feature_dataset(df):
    desired_columns = ['Date', 'open_day0', 'High_day0', 'Low_day0', 'close_day0', 'volume_day0',
                    'rsi_day0', 'macd_day0', 'mean_day0']
    feature_types = ['open', 'High', 'Low', 'close', 'volume', 'rsi', 'macd', 'mean']
    
    filtered_df = df[desired_columns].copy()
    last_row = df.iloc[-1]
    for i in range(1,4):
        last_date = pd.to_datetime(filtered_df['Date'].iloc[-1])
        next_date = last_date + timedelta(days=1)

        day_values = [last_row[f"{ft}_day{i}"] for ft in feature_types]
        day_df = pd.DataFrame([day_values], columns=[f"{ft}_day0" for ft in feature_types])
        day_df.insert(0, 'Date', next_date.strftime('%Y-%m-%d')) 
        filtered_df = pd.concat([filtered_df, day_df], ignore_index=True)
    return filtered_df

def create_lstm_dataset(df, sequence_length=20, target_column='mean_day0'):
    feature_columns = df.columns.drop(['Date'])
    X, y = [], []

    for i in range(len(df) - sequence_length):
        seq_x = df.iloc[i:i+sequence_length][feature_columns].values
        seq_y = df.iloc[i + sequence_length][target_column]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


if __name__ == '__main__':
    train()
