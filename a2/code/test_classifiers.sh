declare -a clfs=("Nearest_Neighbors"  "Nearest_Centroid" "SVM" "Linear_SVM" "RBF_SVM" "Decision_Tree" "Random_Forest" "Naive_Bayes_Gaussian" "Naive_Bayes_Multinomial" "Naive_Bayes_Bernoulli" "LDA" "Ridge" "Gradient_Boosting" "Perceptron" "Passive_Aggressive" "SGD" "Nearest_Centroid" "Elastic_Net" "Logistic_Regression" "Dummy");
for clf in "${clfs[@]}"
do
       echo $clf
       python classify.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name $clf --metrics_data classification_report
done
