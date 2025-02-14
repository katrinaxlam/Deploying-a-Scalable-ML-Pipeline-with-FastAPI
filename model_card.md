# Model Card

## Model Details
**Model Type**: Random Forest Classifier  
**Model Version**: 1.0  
**Trained On**: Census dataset (with features such as age, workclass, education, occupation, etc.)  
**Training Framework**: Scikit-learn  
**License**: MIT  
**Created By**: Katrina Lam

## Intended Use
This model is designed to predict whether an individual earns more than $50K annually based on their demographic features (e.g., age, 
education, workclass, occupation, etc.). The model is intended for use in income classification tasks where the dataset is structured 
similarly to the **Census dataset**.

**Primary Use Case**: 
- Predict income classification (above or below $50K) based on demographic data.

---

## Training Data
The model was trained on the **Census dataset** (also known as the "Adult dataset"). The dataset includes demographic features of adults, 
such as:
- **Features**: age, workclass, education, marital status, occupation, relationship, race, sex, hours worked per week, etc.
- **Number of Instances**: 32,561
- **Total Columns in Dataset**: 15
- **Number of Features**: 14 (including both categorical and numerical features)

The dataset is publicly available and is often used in machine learning tasks for classification.

---

## Evaluation Data
The model was evaluated using a **test split** (20% of the total data). The test data also comes from the **Census dataset**, so it reflects 
real-world distributions similar to the training data.

**Evaluation Metrics**:
- Precision: 0.7391
- Recall: 0.6384
- F1-Score: 0.6851

These metrics indicate that the model performs reasonably well in identifying individuals who earn more than $50K.

---

## Metrics
**Metrics Used**:
- **Precision**: The percentage of predicted positive cases that were actually positive.  
- **Recall**: The percentage of actual positive cases that were correctly identified.  
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

**Model Performance**:
- **Precision**: 0.7391  
- **Recall**: 0.6384  
- **F1-Score**: 0.6851

These performance metrics show that the model has good **precision**, but there’s room for improvement in **recall**. It identifies the 
majority of positive cases correctly, but there are some false negatives.

---

## Ethical Considerations
The **Census dataset** includes sensitive demographic information, and it is important to ensure privacy when using this data in real-world 
applications. This model could potentially have biases based on the distribution of categories in the dataset, especially regarding 
underrepresented demographic groups. 

- **Bias**: The model's performance could vary significantly across different groups (e.g., gender, race, work class). Bias mitigation 
techniques should be considered if the model is used in sensitive applications.

- **Data Privacy**: Care should be taken to protect the privacy of individuals' data, especially when using this model in a production 
environment.

---

## Caveats and Recommendations
- **Performance Limitations**: This model performs well with typical census data but may struggle with out-of-sample data or data that 
differs significantly from the training set.
  
- **Bias and Fairness**: The model may have biases due to overrepresentation of certain demographic groups in the training data. It's 
recommended to evaluate the model on specific demographic groups before deploying it in sensitive applications.
  
- **Model Improvements**: The model could benefit from hyperparameter tuning and possibly using different machine learning models (e.g., 
gradient boosting, neural networks) to improve performance, particularly in terms of recall.

