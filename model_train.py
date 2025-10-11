from sklearn.ensemble import RandomForestClassifier

def train_random_forest(x_train_s, y_train):

    # Обучаем Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(x_train_s, y_train)

    return rf