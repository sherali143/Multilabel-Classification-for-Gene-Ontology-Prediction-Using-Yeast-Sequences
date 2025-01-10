# Multilabel Classification for Gene Ontology Prediction Using Yeast Sequences

## Overview
This project aims to develop a machine learning model capable of predicting Gene Ontology (GO) terms for yeast sequences using a multilabel classification approach. The dataset includes yeast gene sequences in FASTA format and their associated GO terms. Our goal is to preprocess the data, extract features from the sequences, and train a classification model to predict multiple GO terms for each yeast sequence.

## Group Members:
- **Sher Ali** (P21-8024)

## Objective
The objective is to enhance the functional annotation of yeast gene sequences by predicting multiple GO terms using machine learning techniques, such as Random Forest for multilabel classification.

---

## Steps and Approach

### 1. **Data Exploration**
   - **FASTA File**: We will assess the sequences in terms of length and variety.
   - **GO File**: We will explore the distribution and frequency of GO terms across the sequences.
   - **Objective Mapping**: Understand the multilabel nature of the problem by analyzing how GO terms are associated with each sequence.

### 2. **Data Preprocessing**
   - **Sequence Preprocessing**: Parse the FASTA file to extract and standardize protein sequences.
   - **Label Transformation**: Convert GO terms into a binary matrix format using `MultiLabelBinarizer`. Each sequence will be associated with a binary vector representing the presence or absence of each GO term.

### 3. **Feature Extraction**
   - **Sequence Encoding**: The sequences will be encoded into a numeric format suitable for machine learning.
     - **One-hot Encoding** for shorter sequences.
     - **K-mer Frequency Analysis** (optional) to capture sequence patterns.
   - **Dimensionality Reduction** (optional): Apply techniques like PCA for high-dimensional feature spaces.

### 4. **Model Selection**
   - **Traditional Models**: We use a **Random Forest Classifier** adapted for multilabel data through `MultiOutputClassifier`.
   - **Deep Learning Models**: We may explore CNNs or RNNs for sequence-based modeling, depending on the data's nature.

### 5. **Model Training and Cross-Validation**
   - **Train/Test Split**: Split the dataset into training and test sets.
   - **Cross-Validation**: Implement K-fold cross-validation to evaluate model stability and prevent overfitting.

### 6. **Hyperparameter Tuning**
   - **Grid Search/Random Search**: Used to tune hyperparameters for optimal performance.
   - **Regularization**: Implement regularization techniques, especially for deep learning models.

### 7. **Model Evaluation**
   - **Evaluation Metrics**:
     - **Hamming Loss**: Measures incorrect label predictions.
     - **F1 Score** (Macro/Micro): Evaluates precision and recall across labels.

### 8. **Generalization and Testing**
   - Test the model on unseen sequences to evaluate its ability to generalize.

### 9. **Documentation**
   - Maintain detailed documentation at every step for a replicable and clear workflow.

### 10. **Expected Outcome**
   - A trained model capable of predicting multiple GO terms for each yeast sequence. The model's success will be evaluated based on its accuracy in assigning GO terms to sequences and its robustness on unseen data.

---

## Installation Requirements

This project requires the following Python packages:

- **BioPython** for handling FASTA files:
  ```bash
  pip install biopython
  ```
- **Pandas** for data manipulation:
  ```bash
  pip install pandas
  ```
- **Scikit-learn** for machine learning models and metrics:
  ```bash
  pip install scikit-learn
  ```
- **NumPy** for array manipulation:
  ```bash
  pip install numpy
  ```

---

## Code Walkthrough

### 1. **Parsing the FASTA File**

We use `SeqIO` from the **BioPython** library to read the FASTA file and extract sequences.

```python
from Bio import SeqIO
fasta_file = r"C:\Users\Sher Ali\Downloads\YEAST.fasta"
fasta_dict = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    protein_id = record.id.split('|')[1]  # Extract protein ID
    fasta_dict[protein_id] = str(record.seq)  # Store sequence
```

### 2. **Parsing the Gene Ontology (GO) File**

The GO file is parsed to map each protein to its associated GO terms.

```python
go_file = r"C:\Users\Sher Ali\Downloads\AllProteinswithFunctions-Bakers Yeast.txt"
go_dict = {}
with open(go_file, "r") as file:
    for line in file:
        parts = line.strip().split(';')
        protein_id = parts[0]
        go_terms = parts[1:]
        go_dict[protein_id] = go_terms
```

### 3. **Data Combination**

We combine the sequences with their corresponding GO terms into a DataFrame.

```python
data = []
for protein_id, sequence in fasta_dict.items():
    if protein_id in go_dict:
        data.append({
            "Protein ID": protein_id,
            "Sequence": sequence,
            "GO Terms": go_dict[protein_id]
        })

df = pd.DataFrame(data)
```

### 4. **Label Transformation**

The GO terms are transformed into a binary matrix where each column corresponds to a GO term.

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
go_labels = mlb.fit_transform(df["GO Terms"])
go_labels_df = pd.DataFrame(go_labels, columns=mlb.classes_)
df = pd.concat([df, go_labels_df], axis=1).drop("GO Terms", axis=1)
```

### 5. **One-Hot Encoding of Sequences**

Each sequence is converted into a one-hot encoded vector.

```python
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories=[sorted(set("".join(df["Sequence"]))), sparse_output=False, dtype=int)
onehot_encoder.fit(np.array(sorted(set("".join(df["Sequence"])))).reshape(-1, 1))
```

### 6. **Train/Test Split**

We split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7. **Training the Model**

A Random Forest classifier, adapted for multilabel classification, is trained on the data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
```

### 8. **Evaluation**

The model's performance is evaluated using Hamming Loss and F1 Scores (Macro and Micro).

```python
from sklearn.metrics import hamming_loss, f1_score
y_pred = model.predict(X_test)
hamming = hamming_loss(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average="micro")
f1_macro = f1_score(y_test, y_pred, average="macro")
```

---

## Expected Outcome

After training, the model will output the following metrics for evaluation:

- **Hamming Loss**: Fraction of labels predicted incorrectly.
- **F1 Score (Micro)**: Evaluates precision and recall across all labels.
- **F1 Score (Macro)**: Averages precision and recall across each label.

---

## Conclusion

The trained model will predict multiple GO terms for yeast sequences, aiding in the functional annotation of genes. Further improvements can be made by exploring deep learning models (e.g., CNN or RNN) for sequence-based analysis.

--- 

## Future Improvements

1. **K-mer Frequency Analysis**: To capture repeating patterns in longer sequences.
2. **Deep Learning Models**: Investigate the potential of CNNs and RNNs for more accurate predictions.
3. **Model Optimization**: Further hyperparameter tuning using grid search or random search.

---

