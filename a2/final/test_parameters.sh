# Example running
python test_parameters.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_cmd "clf=MultinomialNB(alpha=0.01)" "clf=MultinomialNB(alpha=0.05)" "clf=MultinomialNB(alpha=0.1)" "clf=MultinomialNB(alpha=0.15)" --metrics_data f1_score
python test_parameters.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_cmd "clf=RidgeClassifier(solver='auto')" --metrics_data f1_score
