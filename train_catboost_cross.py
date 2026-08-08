"""Train CatBoost on VitalDB, test on MUST (paper OOD).

  python train_catboost_cross.py
"""

import argparse

import numpy as np

import data_loader
import evaluation
import utils
from config import DEFAULT_SEEDS, DEFAULT_VITALDB_PPG_SECONDS
from train_catboost import build_model, load_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dataset', default='vitaldb')
    p.add_argument('--test-dataset', default='MUST')
    args = p.parse_args()

    # VitalDB defaults to 60 s crop; MUST uses full signal.
    train_ppg = DEFAULT_VITALDB_PPG_SECONDS if args.train_dataset == 'vitaldb' else None
    test_ppg = DEFAULT_VITALDB_PPG_SECONDS if args.test_dataset == 'vitaldb' else None

    X_train, y_train, _ = data_loader.split(
        data_loader.load_dataset(args.train_dataset, ppg_seconds=train_ppg)
    )
    X_test, y_test, _ = data_loader.split(
        data_loader.load_dataset(args.test_dataset, ppg_seconds=test_ppg)
    )

    # Same columns on both sides.
    cols = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[cols], X_test[cols]
    X_train, X_test, _ = utils._impute_features(X_train, X_test)

    params = load_params(args.train_dataset)
    preds = []
    for seed in DEFAULT_SEEDS:
        model = build_model(seed, params)
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    y_pred = np.mean(np.column_stack(preds), axis=1)

    def report(name, pred):
        m = evaluation.evaluate_regression(y_test, pred)
        print(f'\n{name}')
        print(f"  MAE:      {m['mae']:.4f} mmol/L")
        print(f"  RMSE:     {m['rmse']:.4f} mmol/L")
        print(f"  R2:       {m['r2']:.4f}")
        print(f"  ACC:      {100 * m['acc']:.2f}%")
        print(f"  Zone A:   {100 * m['zone_a']:.2f}%")
        print(f"  Zone B:   {100 * m['zone_b']:.2f}%")
        print(f"  Zone A+B: {100 * (m['zone_a'] + m['zone_b']):.2f}%")
        print(f"  MARD:     {m['mard']:.2f}%")
        print(f"  Pearson:  {m['pearson_r']:.4f}")

    report(f'CatBoost ({args.train_dataset} → {args.test_dataset})', y_pred)
    # Train-mean baseline: R2 is 0 only if this equals mean(y_test).
    report('Mean baseline (mean train y)', np.full(len(y_test), float(y_train.mean())))


if __name__ == '__main__':
    main()
