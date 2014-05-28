# Randomized PCA
declare -a method=("RandomizedPCA") # "Percentile" "SelectFpr" "SparsePCA")
for m in "${method[@]}"
do
	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Linear_SVM --metrics_data f1_score --feature_reduction $m --k  500 1000 1500 2000 2500
	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Ridge --metrics_data f1_score --feature_reduction $m --k 500 1000 1500 2000 2500
	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Naive_Bayes_Multinomial --metrics_data f1_score --feature_reduction $m --k 500 1000 1500 2000 2500
done
# I think this is useless --Van
# # Select top tf-idf columns
# declare -a method=("tfidf") # "Percentile" "SelectFpr" "SparsePCA")
# for m in "${method[@]}"
# do
# 	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Linear_SVM --metrics_data f1_score --feature_reduction $m --k  3000 4000 5000 6000
# 	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Ridge --metrics_data f1_score --feature_reduction $m --k 3000 4000 5000 6000
# 	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Naive_Bayes_Multinomial --metrics_data f1_score --feature_reduction $m --k 3000 4000 5000 6000
# done

# SVM based feature reduction
# declare -a method=("L1LinearSVC")
# for m in "${method[@]}"
# do
# 	echo $m
# 	python test_reduction.py --v --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Linear_SVM --metrics_data f1_score --feature_reduction $m --k  0.01 0.05 0.1 1 1.5
# 	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Ridge --metrics_data f1_score --feature_reduction $m --k 0.01 0.05 0.1 1 1.5
# 	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Naive_Bayes_Multinomial --metrics_data f1_score --feature_reduction $m --k 0.01 0.05 0.1 1 1.5
# done

# Nonlinear SVM with Latent Dirichlet something
# declare -a method=("LatentDA")
# for m in "${method[@]}"
# do
# 	echo $m
# 	python test_reduction.py --train_data X_train.out --train_labels y_train.out --test_data X_test.out --test_labels y_test.out --clf_name Ridge --metrics_data f1_score --feature_reduction $m --k  100 200 300 400 500 600 700
# done
