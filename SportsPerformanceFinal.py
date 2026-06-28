import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report


data=pd.read_csv('/Users/tommygillan/Documents/Unstructured/final_gl_df.csv')
bio_columns = ['pitcher_age', 'weight', 'height', 'birthYear']
data = data.dropna(subset=bio_columns).reset_index(drop=True)

data['game_date'] = pd.to_datetime(data['game_date'])
data = data.sort_values(['mlbam_id', 'game_date']).reset_index(drop=True)

data['cumulative_pitches_season'] = (
    data.sort_values(['mlbam_id', 'season', 'game_date'])
        .groupby(['mlbam_id', 'season'])['pitch_count_game']
        .cumsum()
        .shift(1)  
        .fillna(0)
)

data['velo_drop_x_high_workload'] = (
    (data['velo_15d_avg'] < data['velo_60d_avg']).astype(int) *
    (data['pitch_count_28d'] > data['pitch_count_28d'].median()).astype(int)
)
data['age_x_workload'] = data['pitcher_age'] * data['pitch_count_28d']
data['velo_drop_pct'] = (data['velo_60d_avg'] - data['velo_15d_avg']) / (data['velo_60d_avg'] + 1e-6)

## first outing of year flag
data['is_first_appearance_season'] = np.where((data['days_rest'] == 99) | (data['days_rest'].isna()), 1, 0)
# anything over 10 days including the offseason and injuries is treated as fully rested
max_rest_days = 10
# replace the 99s with the MAX_REST_DAYS, cap max days rest everywhere at 10
data['days_rest'] = np.where(data['days_rest'] == 99, max_rest_days, data['days_rest'])
data['days_rest'] = data['days_rest'].clip(upper=max_rest_days)

data['pitch_count_prior_outing'] = data.groupby('mlbam_id')['pitch_count_game'].shift(1)
data['max_inning_prior_3game_avg'] = (
    data.groupby('mlbam_id')['max_inning_game']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
)

data['pitch_count_prior_outing'] = (
    data.groupby('mlbam_id')['pitch_count_game'].shift(1)
)
def lagged_pc_sum(group, window_days):
    g = group.set_index('game_date').sort_index()
    inclusive = g['pitch_count_game'].rolling(f'{window_days}D').sum()
    exclusive = inclusive - g['pitch_count_game']
    # Restore the row index that 'data' uses
    return pd.Series(exclusive.values, index=group.index)

for wd in [7, 28, 60]:
    col = f'pitch_count_{wd}d'
    data[col] = (
        data.groupby('mlbam_id', group_keys=False)
            .apply(lambda g: lagged_pc_sum(g, wd))
    )


##ACWR for a reliver much different than a pitcher, need model to treat them differently as well
data['acwr_pitch_count_x_starter'] = data['acwr_pitch_count'] * data['is_starter_role']
data['acwr_velo_x_starter']        = data['acwr_velo']        * data['is_starter_role']
data['acwr_spin_x_starter']        = data['acwr_spin']        * data['is_starter_role']

data['acwr_pitch_count_x_reliever'] = data['acwr_pitch_count'] * (1 - data['is_starter_role'])
data['acwr_velo_x_reliever'] = data['acwr_velo'] * (1 - data['is_starter_role'])
data['acwr_spin_x_reliever'] = data['acwr_spin'] * (1 - data['is_starter_role'])

### trim extreme outliers frim acwr features
for col in ['acwr_spin_x_starter', 'acwr_spin_x_reliever',
            'acwr_velo_x_starter', 'acwr_velo_x_reliever',
            'acwr_pitch_count_x_starter', 'acwr_pitch_count_x_reliever']:
    lower = data[col].quantile(0.01)
    upper = data[col].quantile(0.99)
    data[col] = data[col].clip(lower=lower, upper=upper)

data['game_date'] = pd.to_datetime(data['game_date'])
data = data.sort_values(by='game_date').reset_index(drop=True)

data.to_csv('SportPerformanceDataset.csv')


train_df = data[data['season'].isin([2021, 2022, 2023])]
val_df = data[data['season'] == 2024]
test_df = data[data['season'] == 2025]


drop_cols = ['target_inj_14d', 'game_date', 'mlbam_id', 'game_pk', 'il_start_date',
             'season', 'description', 'days_to_injury','birthYear','acwr_pitch_count','acwr_velo',
             'acwr_spin', 'min_inning_game', 'max_inning_game', 'pitch_count_game'
             ]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target_inj_14d']

X_val = val_df.drop(columns=drop_cols)
y_val = val_df['target_inj_14d']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target_inj_14d']

from sklearn.impute import SimpleImputer

X_train_enc = pd.get_dummies(X_train, columns=['pitch_hand'], drop_first=True)
X_val_enc   = pd.get_dummies(X_val, columns=['pitch_hand'], drop_first=True)
X_test_enc  = pd.get_dummies(X_test, columns=['pitch_hand'], drop_first=True)


X_val_enc  = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
##might want to remove indicator
imputer = SimpleImputer(strategy='median', add_indicator=True)

X_train_imp = imputer.fit_transform(X_train_enc)
X_val_imp   = imputer.transform(X_val_enc)
X_test_imp  = imputer.transform(X_test_enc)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled   = scaler.transform(X_val_imp)
X_test_scaled  = scaler.transform(X_test_imp)

logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg_model.fit(X_train_scaled, y_train)



# 2024 Validation Results
y_val_pred  = logreg_model.predict(X_val_scaled)
y_val_probs = logreg_model.predict_proba(X_val_scaled)[:, 1]
pr_auc_val  = average_precision_score(y_val, y_val_probs)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
beta = 0.5
f_beta_scores = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-9)
best_idx = np.argmax(f_beta_scores)
best_threshold = thresholds[best_idx]


print(f"Best threshold: {best_threshold:.4f}")
print(f"At this threshold:")
print(f"  Precision: {precisions[best_idx]:.4f}")
print(f"  Recall:    {recalls[best_idx]:.4f}")
print(f"  F1:        {f_beta_scores[best_idx]:.4f}")

#  tuning threshold
y_val_pred_tuned = (y_val_probs >= best_threshold).astype(int)
print(f"\n--- 2024 VALIDATION (Tuned Threshold) ---")
print(classification_report(y_val, y_val_pred_tuned))
print(f"PR-AUC: {pr_auc_val:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))


# 2025 Test Results using tuned threshold
if len(X_test) > 0:
    y_test_probs = logreg_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_default = logreg_model.predict(X_test_scaled)
    y_test_pred_tuned = (y_test_probs >= best_threshold).astype(int)
    pr_auc_test = average_precision_score(y_test, y_test_probs)

    print(classification_report(y_test, y_test_pred_default))
    print(f"\n Tuned Threshold ({best_threshold:.4f},")
    print(classification_report(y_test, y_test_pred_tuned))
  

import lightgbm as lgb
import optuna

random_state = 42

X_train_lgb = train_df.drop(columns=drop_cols).copy()
X_val_lgb   = val_df.drop(columns=drop_cols).copy()
X_test_lgb  = test_df.drop(columns=drop_cols).copy()


X_train_lgb['pitch_hand'] = X_train_lgb['pitch_hand'].astype('category')
X_val_lgb['pitch_hand'] = X_val_lgb ['pitch_hand'].astype('category')
X_test_lgb ['pitch_hand'] = X_test_lgb ['pitch_hand'].astype('category')

categorical_features = [
    'pitch_hand',
    'is_starter_game',
    'is_bulk_game', 
    'is_starter_role',
    'is_first_appearance_season','pitch_hand'
]

for df_ in [X_train_lgb, X_val_lgb, X_test_lgb]:
    for flag in categorical_features:
        df_[flag] = df_[flag].astype('category')
feature_cols = X_train_lgb.columns.tolist()


neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw_baseline = neg_count / pos_count


train_ds = lgb.Dataset(
    X_train_lgb, label=y_train,
    feature_name=feature_cols,
    categorical_feature=categorical_features,
    free_raw_data=False
)
val_ds = lgb.Dataset(
    X_val_lgb, label=y_val,
    feature_name=feature_cols,
    categorical_feature=categorical_features,
    reference=train_ds,
    free_raw_data=False
)
cv_folds_def = [
    ([2021], 2022),           
    ([2021, 2022], 2023),     
]

fold_datasets = []
for train_seasons, val_season in cv_folds_def:
    fold_train_df = train_df[train_df['season'].isin(train_seasons)]
    fold_val_df   = train_df[train_df['season'] == val_season]
    
    X_fold_train = fold_train_df.drop(columns=drop_cols).copy()
    y_fold_train = fold_train_df['target_inj_14d']
    X_fold_val   = fold_val_df.drop(columns=drop_cols).copy()
    y_fold_val   = fold_val_df['target_inj_14d']
    
    
    for df_ in [X_fold_train, X_fold_val]:
        df_['pitch_hand'] = df_['pitch_hand'].astype('category')
        for flag in ['is_starter_game', 'is_bulk_game',
                     'is_starter_role', 'is_first_appearance_season']:
            df_[flag] = df_[flag].astype('category')
    
    fold_train_ds = lgb.Dataset(
        X_fold_train, label=y_fold_train,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        free_raw_data=False
    )
    fold_val_ds = lgb.Dataset(
        X_fold_val, label=y_fold_val,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        reference=fold_train_ds,
        free_raw_data=False
    )
    
    fold_datasets.append({
        'train_ds': fold_train_ds,
        'val_ds': fold_val_ds,
        'X_val': X_fold_val,
        'y_val': y_fold_val,
        'train_seasons': train_seasons,
        'val_season': val_season,
        'n_train': len(fold_train_df),
        'n_val': len(fold_val_df),
        'pos_train': int(y_fold_train.sum()),
        'pos_val': int(y_fold_val.sum()),
    })

def cv_objective(trial):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "seed": random_state,
        "n_jobs": 4,
        "max_bin": 255,
        "subsample_freq": 1,
        "feature_pre_filter": False,
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 8, 31),
        "max_depth":         trial.suggest_int("max_depth", 3, 5),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 500),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "subsample":         trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
    }
    
    fold_scores = []
    for fold in fold_datasets:
        model = lgb.train(
            params,
            fold['train_ds'],
            num_boost_round=2000,
            valid_sets=[fold['val_ds']],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_probs = model.predict(fold['X_val'], num_iteration=model.best_iteration)
        score = average_precision_score(fold['y_val'], val_probs)
        fold_scores.append(score)
    
    return float(np.mean(fold_scores))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True),
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(cv_objective, n_trials=100, show_progress_bar=True)

print(f"\nBest CV mean PR-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_params_full = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "seed": random_state,
    "n_jobs": 4,
    "max_bin": 255,
    "subsample_freq": 1,
    "feature_pre_filter": False,
    **study.best_params,
}

print("\nPer-fold performance with best params:")
fold_scores_final = []
for fold in fold_datasets:
    fold_model = lgb.train(
        best_params_full,
        fold['train_ds'],
        num_boost_round=2000,
        valid_sets=[fold['val_ds']],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_probs = fold_model.predict(fold['X_val'], num_iteration=fold_model.best_iteration)
    score = average_precision_score(fold['y_val'], val_probs)
    fold_scores_final.append(score)
    print(f"  train={fold['train_seasons']} → val={fold['val_season']}: "
          f"PR-AUC={score:.4f}  (best_iter={fold_model.best_iteration})")

print(f"  Mean: {np.mean(fold_scores_final):.4f}  "
      f"Std: {np.std(fold_scores_final):.4f}")


final_model = lgb.train(
    best_params_full,
    train_ds,                
    num_boost_round=2000,
    valid_sets=[val_ds],     
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
print(f"\nFinal model best iteration: {final_model.best_iteration}")


y_val_probs_lgb = final_model.predict(X_val_lgb, num_iteration=final_model.best_iteration)
pr_auc_val_lgb = average_precision_score(y_val, y_val_probs_lgb)

# Tune threshold on validation 
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs_lgb)
beta = 0.5
f_beta_scores = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-9)
best_idx_lgb = np.argmax(f_beta_scores)
best_threshold_lgb = thresholds[best_idx_lgb]

print(f"PR-AUC: {pr_auc_val_lgb:.4f}  "
      f"(LogReg baseline: {pr_auc_val:.4f})")
print(f"Best threshold: {best_threshold_lgb:.4f}")
print(f"  Precision: {precisions[best_idx_lgb]:.4f}")
print(f"  Recall:    {recalls[best_idx_lgb]:.4f}")
print(f"  F1:        {f_beta_scores[best_idx_lgb]:.4f}")
y_val_pred_lgb_tuned = (y_val_probs_lgb >= best_threshold_lgb).astype(int)
print(classification_report(y_val, y_val_pred_lgb_tuned))

#evaluate on test set
y_test_probs_lgb = final_model.predict(X_test_lgb, num_iteration=final_model.best_iteration)
pr_auc_test_lgb = average_precision_score(y_test, y_test_probs_lgb)
y_test_pred_lgb_tuned = (y_test_probs_lgb >= best_threshold_lgb).astype(int)


print(f"PR-AUC: {pr_auc_test_lgb:.4f}  "
      f"(LogReg baseline: {pr_auc_test:.4f})")
print(f"Tuned Threshold ({best_threshold_lgb:.4f},")
print(classification_report(y_test, y_test_pred_lgb_tuned))


# use training set  as reference distribution for score
train_probs = final_model.predict(
    X_train_lgb, num_iteration=final_model.best_iteration
)
#convert raw probability to percentile rank 
def prob_to_risk_score(probs, reference_probs):
    ranks = np.searchsorted(np.sort(reference_probs), probs, side='right')
    return (ranks / len(reference_probs)) * 100

val_risk_scores  = prob_to_risk_score(y_val_probs_lgb, train_probs)
test_risk_scores = prob_to_risk_score(y_test_probs_lgb, train_probs)


percentile_cuts = {
    'HIGH':     95.0,   
    'ELEVATED': 85.0,   
    'MODERATE': 70.0,   
    # below 70th percentile = LOW
}

tier_thresholds = {
    tier: float(np.percentile(train_probs, pct))
    for tier, pct in percentile_cuts.items()
}
#assign each tier a threshold based on training distrubution
print("=== RANK-BASED TIER THRESHOLDS (from training distribution) ===")
for tier in [ 'HIGH', 'ELEVATED', 'MODERATE']:
    pct = percentile_cuts[tier]
    thr = tier_thresholds[tier]
    n_train = int(np.sum(train_probs >= thr))
    n_val   = int(np.sum(y_val_probs_lgb >= thr))
    n_test  = int(np.sum(y_test_probs_lgb >= thr))
    print(f"  {tier:10s}: pct={pct:.1f}, prob>={thr:.4f} | "
          f"n_train={n_train}, n_val={n_val}, n_test={n_test}")

def assign_tier(prob):
    if prob >= tier_thresholds['HIGH']:
        return "HIGH"
    elif prob >= tier_thresholds['ELEVATED']:
        return "ELEVATED"
    elif prob >= tier_thresholds['MODERATE']:
        return "MODERATE"
    else:
        return "LOW"

val_results = pd.DataFrame({
    'risk_score':     val_risk_scores,
    'tier':           [assign_tier(p) for p in y_val_probs_lgb],
    'actual_injury':  y_val.values,
    'probability':    y_val_probs_lgb,
}).reset_index(drop=True)


test_results = pd.DataFrame({
    'risk_score':     test_risk_scores,
    'tier':           [assign_tier(p) for p in y_test_probs_lgb],
    'actual_injury':  y_test.values,
    'probability':    y_test_probs_lgb,
}).reset_index(drop=True)
# identify players that are archetype flags
test_meta = test_df[['mlbam_id', 'game_date']].reset_index(drop=True)
test_results['mlbam_id'] = test_meta['mlbam_id'].values
test_results['game_date'] = test_meta['game_date'].values
pitcher_flag_counts = (
    test_results[test_results['tier'].isin(['HIGH', 'CRITICAL'])]
    .groupby('mlbam_id').size().rename('n_top_tier_flags')
)
test_results = test_results.merge(
    pitcher_flag_counts, left_on='mlbam_id', right_index=True, how='left'
)
test_results['n_top_tier_flags'] = test_results['n_top_tier_flags'].fillna(0)

# Outings where the model flagged the same pitcher 4+ times = archetype, not acute
test_results['flag_type'] = np.where(
    test_results['n_top_tier_flags'] >= 4, 'ARCHETYPE', 'ACUTE'
)

tier_order = [ 'MODERATE', 'ELEVATED', 'HIGH']
base_rate_val  = y_val.mean()
base_rate_test = y_test.mean()

def build_tier_summary(results_df, base_rate):
    summary = results_df.groupby('tier').agg(
        n_outings      = ('actual_injury', 'count'),
        n_injuries     = ('actual_injury', 'sum'),
        injury_rate    = ('actual_injury', 'mean'),
        avg_prob       = ('probability',   'mean'),
        avg_risk_score = ('risk_score',    'mean'),
    )
    summary['lift_vs_baseline'] = summary['injury_rate'] / base_rate
    summary['false_alarms_per_injury'] = (
        (summary['n_outings'] - summary['n_injuries'])
        / summary['n_injuries'].replace(0, np.nan)
    )
    return summary.reindex(tier_order)

print("Val Risk Tier Summary")
tier_summary_val = build_tier_summary(val_results, base_rate_val)
print(tier_summary_val.round(3))
print(f"Base injury rate: {base_rate_val:.4f}")

print("Test Risk Tier Summar")
tier_summary_test = build_tier_summary(test_results, base_rate_test)
print(tier_summary_test.round(3))
print(f"Base injury rate: {base_rate_test:.4f}")


def build_acute_archetype_summary(results_df, base_rate):
    high_only = results_df[results_df['tier'] == 'HIGH']
    summary = high_only.groupby('flag_type').agg(
        n_outings      = ('actual_injury', 'count'),
        n_injuries     = ('actual_injury', 'sum'),
        injury_rate    = ('actual_injury', 'mean'),
        avg_prob       = ('probability',   'mean'),
    )
    summary['lift_vs_baseline'] = summary['injury_rate'] / base_rate
    summary['false_alarms_per_injury'] = (
        (summary['n_outings'] - summary['n_injuries'])
        / summary['n_injuries'].replace(0, np.nan)
    )
    return summary.reindex(['ACUTE', 'ARCHETYPE'])

acute_arch_test = build_acute_archetype_summary(test_results, base_rate_test)
print(acute_arch_test.round(3))

arch_pitchers = (
    test_results[(test_results['tier'] == 'HIGH') & (test_results['flag_type'] == 'ARCHETYPE')]
    .groupby('mlbam_id').agg(
        n_flagged=('actual_injury', 'count'),
        n_injured=('actual_injury', 'sum'),
    )
    .sort_values('n_flagged', ascending=False)
)
print(f"\n Archetype-flagged pitchers ({len(arch_pitchers)} total):")
print(arch_pitchers.head(10))

print("How many healthy pitchers rested per real injury caught?")
for tier in [ 'HIGH', 'ELEVATED', 'MODERATE']:
    row = tier_summary_test.loc[tier] if tier in tier_summary_test.index else None
    if row is None or np.isnan(row['n_injuries']) or row['n_injuries'] == 0:
        print(f"  {tier:10s}: no injuries captured at this tier")
        continue
    n_flagged     = int(row['n_outings'])
    n_caught      = int(row['n_injuries'])
    n_false       = n_flagged - n_caught
    cost_ratio    = n_false / n_caught
    total_inj     = y_test.sum()
    pct_caught    = n_caught / total_inj
    print(f"  {tier:10s}: flagged={n_flagged:4d} | injuries caught={n_caught} "
          f"({pct_caught:.1%} of all 2025 injuries) | "
          f"false alarms={n_false} | "
          f"cost ratio={cost_ratio:.1f} healthy rest days per injury caught")
    

import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(final_model)
shap_values_val = explainer.shap_values(X_val_lgb)
shap_values_test = explainer.shap_values(X_test_lgb)

if isinstance(shap_values_val, list):
    shap_values_val = shap_values_val[1]   
if isinstance(shap_values_test, list):
    shap_values_test = shap_values_test[1]


plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_val,
    X_val_lgb,
    max_display=20,
    show=False,
    plot_size=(10, 8)
)
plt.title("SHAP Summary: Feature Importance for Injury Prediction",
          fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")


y_test_probs_ensemble = 0.5 * y_test_probs + 0.5 * y_test_probs_lgb
pr_auc_ensemble = average_precision_score(y_test, y_test_probs_ensemble)
print(f"Ensemble PR-AUC: {pr_auc_ensemble:.4f}")


def get_pitcher_shap(pitcher_id, top_n=10):
    mask = test_meta['mlbam_id'] == pitcher_id
    pitcher_rows = test_meta[mask].index.tolist()
    
    if not pitcher_rows:
        print(f"No 2025 outings found for pitcher {pitcher_id}")
        return None
    
    print(f"\n{'='*70}")
    print(f"PITCHER {pitcher_id}: {len(pitcher_rows)} outings in 2025 test set")
    print(f"{'='*70}")
    

import shap
import matplotlib.pyplot as plt

target_pitcher = 694973         
target_date    = '2025-07-01'        

target_date_ts = pd.Timestamp(target_date)
mask = (test_meta['mlbam_id'] == target_pitcher) & (test_meta['game_date'] == target_date_ts)
matches = test_meta[mask].index.tolist()

if len(matches) == 0:
    pitcher_outings = test_meta[test_meta['mlbam_id'] == target_pitcher].copy()
    if len(pitcher_outings) == 0:
        raise ValueError(
            f"Pitcher {target_pitcher} has no outings in the 2025 test set."
        )
    print(f"No outing found for pitcher {target_pitcher} on {target_date_ts.date()}.")
    print(f"Available 2025 dates for this pitcher:")
    print(pitcher_outings['game_date'].dt.date.tolist())
    raise ValueError("Update target_date to one of the dates listed above.")

if len(matches) > 1:
    print(f"Warning: {len(matches)} outings match. Using the first one.")

idx = matches[0]

pitcher_id = test_meta.iloc[idx]['mlbam_id']
game_date  = test_meta.iloc[idx]['game_date']
prob       = y_test_probs_lgb[idx]
risk_score = test_risk_scores[idx]
tier       = assign_tier(prob)
actual     = bool(y_test.iloc[idx])
flag_type  = test_results.iloc[idx]['flag_type'] if 'flag_type' in test_results.columns else 'n/a'

print(f"\nSelected outing:")
print(f"  Pitcher {pitcher_id} on {game_date.date()}")
print(f"  Predicted prob: {prob:.4f}")
print(f"  Risk score: {risk_score:.0f}/100")
print(f"  Tier: {tier}  |  Flag type: {flag_type}")
print(f"  Actually injured within 14 days: {'YES' if actual else 'no'}")


expected_value = explainer.expected_value
ev_arr = np.atleast_1d(expected_value)
expected_value = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])

print(f"\nModel baseline (expected log-odds output): {expected_value:.4f}")


row_features = X_test_lgb.iloc[idx].copy()
for col in row_features.index:
    val = row_features[col]
    if hasattr(val, 'item'):
        try:
            row_features[col] = val.item()
        except (AttributeError, ValueError):
            row_features[col] = str(val)
row_data = row_features.values

single_explanation = shap.Explanation(
    values=shap_values_test[idx],
    base_values=expected_value,
    data=row_data,
    feature_names=X_test_lgb.columns.tolist(),
)


plt.figure(figsize=(11, 7))
shap.plots.waterfall(single_explanation, max_display=12, show=False)

injury_str = "ACTUALLY INJURED ✓" if actual else "Not injured within 14 days"
plt.title(
    f"SHAP Waterfall — Pitcher {pitcher_id} on {game_date.date()}\n"
    f"Tier: {tier}  |  Flag: {flag_type}  |  Risk: {risk_score:.0f}/100  |  "
    f"Prob: {prob:.4f}  |  {injury_str}",
    fontsize=11, pad=15
)
plt.tight_layout()

filename = f"shap_waterfall_{pitcher_id}_{game_date.date()}.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {filename}")