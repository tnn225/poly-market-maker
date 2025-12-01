from lightgbm import LGBMClassifier
from poly_market_maker.dataset import Dataset
from poly_market_maker.models import Model

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df
    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

    lightgbm_params = {
        'n_estimators': 1000,
        'max_depth': -1,
        'learning_rate': 0.001,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_jobs': -1
    }
    
    model_name = f"LGBMClassifier_features_{len(feature_cols)}_{lightgbm_params['n_estimators']}_{lightgbm_params['max_depth']}_{lightgbm_params['learning_rate']}_{lightgbm_params['num_leaves']}_{lightgbm_params['subsample']}_{lightgbm_params['colsample_bytree']}_{lightgbm_params['n_jobs']}"
    model = Model(model_name, LGBMClassifier(**lightgbm_params), feature_cols=feature_cols, dataset=dataset)

    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)

    print(model.get_probability(87684.42122177457, 87498.58994751809, 60, 0.53, 0.55))
    print(model.get_probability(87398.58994751809, 87584.42122177457, 60, 0.45, 0.47))

if __name__ == "__main__":
    main()
