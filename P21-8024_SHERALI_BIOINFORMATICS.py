# NAME : SHERALI
# P21-8024

from Bio import SeqIO
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, f1_score
import numpy as np

# File paths
fasta_file = r"C:\Users\Sher Ali\Downloads\YEAST.fasta"
go_file = r"C:\Users\Sher Ali\Downloads\AllProteinswithFunctions-Bakers Yeast.txt"

# Step 1: Parse the FASTA file to extract protein sequences
fasta_dict = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    protein_id = record.id.split('|')[1]  # Extract protein ID
    fasta_dict[protein_id] = str(record.seq)  # Store sequence

print("FASTA file parsed. Number of sequences:", len(fasta_dict))

# Step 2: Parse the gene ontology file
go_dict = {}
with open(go_file, "r") as file:
    for line in file:
        parts = line.strip().split(';')
        protein_id = parts[0]
        go_terms = parts[1:]
        go_dict[protein_id] = go_terms

print("GO terms file parsed. Number of entries:", len(go_dict))

# Step 3: Combine sequence data and GO terms
data = []
for protein_id, sequence in fasta_dict.items():
    if protein_id in go_dict:
        data.append({
            "Protein ID": protein_id,
            "Sequence": sequence,
            "GO Terms": go_dict[protein_id]
        })

print(f"Total data combined: {len(data)} entries")

df = pd.DataFrame(data)

# Step 4: Convert GO terms to binary matrix format
mlb = MultiLabelBinarizer()
go_labels = mlb.fit_transform(df["GO Terms"])
go_labels_df = pd.DataFrame(go_labels, columns=mlb.classes_)
df = pd.concat([df, go_labels_df], axis=1).drop("GO Terms", axis=1)

print(f"GO terms binary matrix created. Shape: {go_labels_df.shape}")

# Step 5: One-hot encoding for sequences

amino_acids = sorted(set("".join(df["Sequence"])))
print(f"Amino acids found: {amino_acids}")

# Initialize and fit the OneHotEncoder with the amino acids
onehot_encoder = OneHotEncoder(categories=[amino_acids], sparse_output=False, dtype=int)
onehot_encoder.fit(np.array(amino_acids).reshape(-1, 1))  


print("OneHotEncoder fitted with amino acids.")

# Define a function to one-hot encode each sequence
def one_hot_encode_sequence(sequence, encoder):
    sequence_array = np.array(list(sequence)).reshape(-1, 1)
    return encoder.transform(sequence_array).flatten()

# Apply one-hot encoding to all sequences and store them in a list
one_hot_encoded_sequences = []
for seq in df["Sequence"]:
    encoded_seq = one_hot_encode_sequence(seq, onehot_encoder)
    one_hot_encoded_sequences.append(encoded_seq)

# Now check the shape of the first few one-hot encoded sequences
for i in range(5):  # Check the first 5 sequences
    print(f"Sequence {i+1} encoded length: {len(one_hot_encoded_sequences[i])}")

# Convert the list of one-hot encoded sequences into a DataFrame
one_hot_encoded_df = pd.DataFrame(one_hot_encoded_sequences)

# Step 6: Combine the one-hot encoded sequences with the GO term labels
X = one_hot_encoded_df
y = go_labels_df

# Print the shapes of X and y
print(f"Feature matrix X shape: {X.shape}")
print(f"Target matrix y shape: {y.shape}")

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data split into training and test sets. Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Step 8: Train Random Forest classifier
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
print("Training model...")
model.fit(X_train, y_train)

# Step 9: Model evaluation
y_pred = model.predict(X_test)
hamming = hamming_loss(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average="micro")
f1_macro = f1_score(y_test, y_pred, average="macro")

print(f"Hamming Loss: {hamming}")
print(f"F1 Score (Micro): {f1_micro}")
print(f"F1 Score (Macro): {f1_macro}")
