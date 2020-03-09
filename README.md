# LT2212 V20 Assignment 2

# Assignment 1

### Part 1
In the first part we list of documents as an input and consitute a word frequency of each document and converts it into a numpy.ndarray where the rows are documenst and the columns are word frequencies. But before doing that, first we do some preprocessing and clean the text.
- I have used string.punctuation to remove the punctuation as I found that if the punctutation is not removed the number of columns are increased and there are a lot of garbage words that dont make sense.
- All the words are converted into lower case to avoid duplicated values.
- The words are split on the basic of spaces.
- I have also removoed stop words from the text to lower the frequnecy of common words. The list was complied from the link https://gist.github.com/sebleier/554280
- For further lowering the amount of unique words, another check was used that ignored the words which had the length of less than 3. I found that out that it further resulted in the decrease of garbage words.
- All the nans were replaced with zero
- We can also ignore the words that have word frequency in the entire corpus less than 50(The reason for choosing that it doesnt fit into the memory when the word frequency in the entire corpus was less than 50)  .

### Part 2

The dimensionality reduction algorithm selected was TruncatedSVD.

### Part 3
Model 1 is : Logistic Regression
Model 2 is : DecisionTreeClassifier

### Part 4
#### Results of LogisticRegression Classifier without reduced dimensionality


              precision    recall  f1-score   support

           0       0.78      0.88      0.82       137
           1       0.77      0.75      0.76       218
           2       0.78      0.77      0.78       213
           3       0.74      0.75      0.74       219
           4       0.84      0.84      0.84       193
           5       0.84      0.79      0.81       206
           6       0.84      0.76      0.80       191
           7       0.89      0.86      0.87       211
           8       0.90      0.91      0.91       163
           9       0.93      0.92      0.92       220
          10       0.92      0.95      0.94       188
          11       0.89      0.96      0.93       200
          12       0.83      0.76      0.79       207
          13       0.90      0.88      0.89       181
          14       0.91      0.96      0.93       178
          15       0.89      0.83      0.86       219
          16       0.87      0.90      0.89       185
          17       0.91      0.97      0.94       178
          18       0.80      0.86      0.83       136
          19       0.81      0.83      0.82       127

    accuracy                           0.85      3770
    macro avg      0.85      0.86      0.85      3770
    weighted avg   0.85      0.85      0.85      3770




#### Results of DecisionTreeClassifier without reduced dimensionality

              precision    recall  f1-score   support

           0       0.53      0.59      0.56       138
           1       0.55      0.57      0.56       208
           2       0.70      0.61      0.65       240
           3       0.50      0.54      0.52       207
           4       0.64      0.68      0.66       179
           5       0.63      0.58      0.60       208
           6       0.60      0.51      0.55       203
           7       0.69      0.65      0.67       219
           8       0.75      0.70      0.72       177
           9       0.70      0.68      0.69       225
          10       0.75      0.76      0.76       191
          11       0.74      0.84      0.79       190
          12       0.55      0.50      0.52       208
          13       0.64      0.63      0.63       179
          14       0.76      0.76      0.76       189
          15       0.69      0.63      0.66       222
          16       0.64      0.73      0.68       167
          17       0.76      0.83      0.80       174
          18       0.57      0.53      0.55       156
          19       0.41      0.59      0.48        90

    accuracy                           0.65      3770
    macro avg       0.64      0.65     0.64      3770
    weighted avg    0.65      0.65     0.65      3770

Logistic regression performed alot better the decision tree classifier without the reduced dimensionality data.
Next we will see the result of the same classifiers on the dimensionality reduction with 50 percent , 25 percent, 10 percent and 5 percent of the original dimensions.
Lets start with the 50 percent.
#### Results of LogisticRegression with 50 precent of the original dimensions

              precision    recall  f1-score   support

           0       0.78      0.88      0.83       136
           1       0.76      0.74      0.75       219
           2       0.78      0.77      0.77       212
           3       0.73      0.74      0.74       219
           4       0.81      0.83      0.82       189
           5       0.83      0.77      0.80       208
           6       0.86      0.76      0.80       195
           7       0.89      0.87      0.88       210
           8       0.89      0.90      0.90       164
           9       0.93      0.94      0.94       216
          10       0.92      0.95      0.93       189
          11       0.90      0.97      0.94       201
          12       0.83      0.75      0.79       208
          13       0.89      0.88      0.89       180
          14       0.89      0.95      0.92       176
          15       0.89      0.82      0.85       222
          16       0.89      0.90      0.90       187
          17       0.91      0.97      0.94       177
          18       0.79      0.85      0.82       136
          19       0.77      0.79      0.78       126

    accuracy                           0.85      3770
    macro avg       0.85      0.85     0.85      3770
    weighted avg    0.85      0.85     0.85      3770

We can see that with 50 percent of the original features, TruncatedSVD was able to the capture much of the information with 50 percent of the original features which in turned helped the classifier performed the classification which yielded nearly the same result as the original dataset with all the features.

#### Results of DecisionTree with 50 precent of the original dimensions

                precision    recall  f1-score   support

           0       0.23      0.19      0.21       189
           1       0.19      0.23      0.21       175
           2       0.42      0.40      0.41       220
           3       0.20      0.27      0.23       166
           4       0.30      0.27      0.28       216
           5       0.44      0.45      0.44       188
           6       0.25      0.23      0.24       195
           7       0.46      0.44      0.45       217
           8       0.43      0.32      0.37       219
           9       0.37      0.42      0.39       192
          10       0.60      0.64      0.62       183
          11       0.45      0.53      0.49       184
          12       0.28      0.24      0.26       216
          13       0.36      0.36      0.36       174
          14       0.58      0.52      0.55       209
          15       0.45      0.46      0.46       200
          16       0.39      0.43      0.41       174
          17       0.56      0.64      0.60       166
          18       0.28      0.28      0.28       144
          19       0.25      0.22      0.23       143

    accuracy                           0.38      3770
    macro avg       0.37      0.38     0.37      3770
    weighted avg    0.38      0.38     0.38      3770
    
The decisionTree Classifier wasnt able to get the same result as the logistic regression and perfomed badly with respect to the original features with no dimensionaity reduction.

#### Results of LogisticRegression with 25 precent of the original dimensions

              precision    recall  f1-score   support

           0       0.77      0.82      0.80       145
           1       0.73      0.73      0.73       214
           2       0.75      0.73      0.74       214
           3       0.72      0.73      0.72       218
           4       0.82      0.83      0.82       190
           5       0.81      0.78      0.79       200
           6       0.86      0.73      0.79       204
           7       0.89      0.86      0.87       214
           8       0.87      0.89      0.88       162
           9       0.92      0.94      0.93       214
          10       0.93      0.94      0.93       192
          11       0.90      0.95      0.92       204
          12       0.76      0.73      0.75       197
          13       0.87      0.86      0.86       180
          14       0.91      0.94      0.93       182
          15       0.86      0.83      0.84       213
          16       0.86      0.90      0.88       183
          17       0.92      0.98      0.95       178
          18       0.80      0.84      0.82       139
          19       0.75      0.76      0.75       127

    accuracy                            0.84      3770
    macro avg       0.83      0.84      0.84      3770
    weighted avg    0.84      0.84      0.84      3770

With 25 percent of the original features it seems that the accuracy dropped a little from 0.85 to 0.84 which is not a drastic change but points that all the variannce wasnt captured with the 25 percent of the original data.

#### Results of DecisionTree with 25 precent of the original dimensions

              precision    recall  f1-score   support

           0       0.21      0.21      0.21       158
           1       0.20      0.24      0.21       178
           2       0.41      0.41      0.41       207
           3       0.23      0.27      0.25       184
           4       0.28      0.26      0.27       209
           5       0.42      0.39      0.40       210
           6       0.25      0.20      0.22       211
           7       0.43      0.41      0.42       217
           8       0.42      0.34      0.38       206
           9       0.36      0.48      0.41       164
          10       0.64      0.64      0.64       193
          11       0.48      0.57      0.52       183
          12       0.24      0.21      0.23       219
          13       0.43      0.44      0.44       171
          14       0.59      0.54      0.56       207
          15       0.44      0.44      0.44       206
          16       0.42      0.40      0.41       199
          17       0.56      0.65      0.60       164
          18       0.28      0.27      0.28       151
          19       0.20      0.20      0.20       133

    accuracy                           0.38      3770
    macro avg       0.37      0.38     0.37      3770
    weighted avg    0.38      0.38     0.38      3770

Interestingly reducing the number of features to 25 percent of the original data doesnt have an impact on the decisiontree. Rather it has the same ccuracy it had with the 50 percent of the original features. It seems that that the decision tree with the default parameter is doing the splits that require a few features and doesnt take much help from when the features is 50 percent of the original features.

#### Results of LogisticRegression with 10 precent of the original dimensions

               precision    recall  f1-score   support

           0       0.72      0.76      0.74       147
           1       0.67      0.70      0.68       204
           2       0.75      0.69      0.72       226
           3       0.68      0.72      0.70       212
           4       0.80      0.82      0.81       187
           5       0.75      0.75      0.75       193
           6       0.86      0.68      0.76       220
           7       0.87      0.86      0.86       207
           8       0.84      0.90      0.87       155
           9       0.89      0.93      0.91       209
          10       0.93      0.96      0.95       189
          11       0.86      0.94      0.90       196
          12       0.79      0.72      0.76       207
          13       0.86      0.81      0.84       187
          14       0.89      0.87      0.88       192
          15       0.83      0.80      0.82       210
          16       0.85      0.86      0.85       187
          17       0.91      0.97      0.94       178
          18       0.73      0.78      0.76       137
          19       0.69      0.71      0.70       127

    accuracy                           0.81      3770
    macro avg       0.81      0.81     0.81      3770
    weighted avg    0.81      0.81     0.81      3770

We are seeing a pattern with the logistic regression classifier that as the feature percentage is lowered the accuracy is decreasing. This points to the fact that the with less features less variance is captured which in turns makes it difficult for the classifier to make accurate predictions.

#### Results of DecisionTreeClassifier with 10 precent of the original dimensions

              precision    recall  f1-score   support

           0       0.32      0.30      0.31       164
           1       0.26      0.25      0.25       220
           2       0.43      0.44      0.44       204
           3       0.27      0.34      0.30       175
           4       0.26      0.24      0.25       208
           5       0.41      0.42      0.42       189
           6       0.29      0.23      0.26       223
           7       0.44      0.41      0.42       220
           8       0.43      0.39      0.41       185
           9       0.41      0.49      0.45       183
          10       0.64      0.66      0.65       188
          11       0.49      0.62      0.55       172
          12       0.26      0.23      0.24       216
          13       0.35      0.33      0.34       188
          14       0.59      0.55      0.57       200
          15       0.44      0.44      0.44       205
          16       0.41      0.44      0.42       179
          17       0.60      0.66      0.63       172
          18       0.31      0.31      0.31       146
          19       0.19      0.19      0.19       133

    accuracy                           0.40      3770
    macro avg       0.39      0.40     0.39      3770
    weighted avg    0.39      0.40     0.39      3770

The decisiontree classifier is not following the conventional pattern in a sense that with 10 percent features the accuracy is increased with less amount of features. This can be due to the fact that usually decisiontree works well with lower dimensions and maybe suffers from the curse of dimensionality.

#### Results of LogisticRegression with 5 precent of the original dimensions

              precision    recall  f1-score   support

           0       0.64      0.76      0.70       130
           1       0.68      0.67      0.67       217
           2       0.71      0.73      0.72       203
           3       0.67      0.70      0.69       212
           4       0.79      0.80      0.79       188
           5       0.74      0.72      0.73       199
           6       0.80      0.65      0.72       213
           7       0.80      0.79      0.80       208
           8       0.81      0.83      0.82       162
           9       0.87      0.92      0.89       206
          10       0.92      0.97      0.94       185
          11       0.86      0.92      0.89       201
          12       0.72      0.64      0.68       213
          13       0.86      0.78      0.82       196
          14       0.88      0.86      0.87       191
          15       0.79      0.76      0.77       213
          16       0.78      0.80      0.79       186
          17       0.87      0.93      0.90       177
          18       0.66      0.72      0.69       133
          19       0.65      0.62      0.64       137

    accuracy                           0.78      3770
    macro avg       0.78      0.78     0.78      3770
    weighted avg    0.78      0.78     0.78      3770

The accuracy is decreasing as the dimensionlity is decreasing.

#### Results of DecisionTreeClassifier with 5 precent of the original dimensions

              precision    recall  f1-score   support

           0       0.30      0.30      0.30       153
           1       0.21      0.24      0.23       190
           2       0.37      0.40      0.38       194
           3       0.27      0.30      0.28       194
           4       0.30      0.28      0.29       207
           5       0.43      0.46      0.44       182
           6       0.27      0.21      0.24       217
           7       0.42      0.42      0.42       207
           8       0.45      0.33      0.38       224
           9       0.38      0.48      0.43       176
          10       0.69      0.66      0.67       203
          11       0.52      0.53      0.52       215
          12       0.29      0.27      0.28       197
          13       0.41      0.38      0.39       188
          14       0.61      0.57      0.59       200
          15       0.44      0.48      0.46       186
          16       0.44      0.46      0.45       182
          17       0.59      0.66      0.63       171
          18       0.34      0.31      0.32       156
          19       0.23      0.23      0.23       128

    accuracy                           0.40      3770
    macro avg       0.40      0.40     0.40      3770
    weighted avg    0.40      0.40     0.40      3770

With only 5 percent of the original feature the performance of the decisionTree classifier actually improved further with better accuracy, f1 score and precision than 50 percent of the dimensional.

As pointed out above that the logistic regression performs well with more features and as the dimensionality is decreased, the accuracy is decreased. But when the features were 50 percent of the original dimension the classifier was able to perform the same as the classifier with 100 percent of the features. The decisionTree classifier performed okay with the 100 percent of the dimensions but the accuracy decreased a lot when 50 percent features were selected. As the features are further reduced the accuracy actually increased a little bit which can be random too. 




    




