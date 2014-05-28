X="data_train.txt"
y="data_train_labels.txt"
classifier="Nearest_Centroid"
feature_reduction="tfidf"
k=3500
X_train_out="X_train.out"
X_test_out="X_test.out"
y_train_out="y_train.out"
y_test_out="y_test.out"
clf_file="clf_file"
X_all="X_all.out"
y_all="y_all.out"

echo "Which option would you like to run?"
echo "1) Run once off, don't save the data for faster loading next time"
echo "2) Load the text files but also save the data for faster loading next time"
echo "3) Load from the files outputted in (2)"
echo "4) Load from the files outputted in (2) and also save the trained clf"
echo "5) Load from the filex saved in (2) and load the clf saved in (4)"
echo "6) Prepare the data for the lab demo"
echo "7) Run the test data for the lab"
echo "Option:"
read ans

case $ans in
# One off running (ie you don't want to save the data for faster loading next time)
1) python classify.py --train_data $X --train_labels $y --test_size 0.1 --feature_reduction $feature_reduction --k $k --clf_name $classifier --metrics_data classification_report
;;
# Loading the text files, but we want to save them somewhere too
2) python classify.py --train_data $X --train_labels $y --test_size 0.1 --test_data_save $X_train_out --test_labels_save $y_train_out --train_data_save $X_train_out --train_labels_save $y_train_out --feature_reduction $feature_reduction --k $k --clf_name $classifier --metrics_data classification_report
;;
# Load from the outputted files instead
3) python classify.py --train_data $X_train_out --train_labels $y_train_out --test_data $X_test_out --test_labels $y_test_out --feature_reduction $feature_reduction --k $k --clf_name $classifier --metrics_data classification_report;
;;
# Save the trained clf
4) python classify.py --train_data $X_train_out --train_labels $y_train_out --test_data $X_test_out --test_labels $y_test_out --feature_reduction $feature_reduction --k $k --clf_name $classifier --save_clf $clf_file --metrics_data classification_report
;;
# Load the trained clf
5) python classify.py --train_data $X_train_out --train_labels $y_train_out --test_data $X_test_out --test_labels $y_test_out --feature_reduction $feature_reduction --k $k --clf_file $clf_file --metrics_data classification_report
;;
# Prepare the data for the lab
6) python classify.py --train_data $X --train_data_save $X_all --train_labels $y --train_labels_save $y_all --clf_name $classifier --save_clf final_classifier
;;
# Run the data for the lab
7) python classify.py --train_data $X --train_data_save $X_all --train_labels $y --train_labels_save $y_all --test_data $X_test --clf_file final_classifier --metrics_data classification_report
;;
*) python classify.py -h
;;
esac
