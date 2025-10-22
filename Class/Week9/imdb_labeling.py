from datasets import load_dataset
import pandas as pd
import re

# For Snorkel
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

# For ML model and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load and Shuffle Data
imdb = load_dataset("imdb")
# Shuffle the datasets first to get a random mix of positive and negative reviews
train_shuffled = imdb["train"].shuffle(seed=42)
test_shuffled = imdb["test"].shuffle(seed=42)

# Now select from the shuffled data
train = pd.DataFrame(train_shuffled.select(range(2000)))
test = pd.DataFrame(test_shuffled.select(range(500)))
print("Train size:", len(train), "Test size:", len(test))

# 2. Define Labeling Functions (LFs)
ABSTAIN, NEG, POS = -1, 0, 1
positive_words = {"great", "excellent", "amazing", "wonderful", "best", "fantastic"}
negative_words = {"bad", "terrible", "awful", "worst", "boring", "poor"}

@labeling_function()
def lf_positive(x):
    return POS if any(w in x.text.lower().split() for w in positive_words) else ABSTAIN

@labeling_function()
def lf_negative(x):
    return NEG if any(w in x.text.lower().split() for w in negative_words) else ABSTAIN

@labeling_function()
def lf_exclaim(x):
    return POS if x.text.count("!") > 2 else ABSTAIN

lfs = [lf_positive, lf_negative, lf_exclaim]

# 3. Apply LFs and Train Label Model
applier = PandasLFApplier(lfs)
L_train = applier.apply(train)

print("LF Analysis Summary:")
print(LFAnalysis(L_train, lfs).lf_summary())

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)
train_preds = label_model.predict(L_train)

# 4. Clean Text for Final Classifier
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    return text.lower()

train["text"] = train["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)

# 5. Train a Final Classifier on Snorkel's Labels
vectorizer = TfidfVectorizer(max_features=5_000)
vectorizer.fit(train["text"])

train_filtered_mask = train_preds != ABSTAIN
X_train_filtered = vectorizer.transform(train["text"][train_filtered_mask])
y_train_filtered = train_preds[train_filtered_mask]

print("\nTraining classifier on Snorkel's labels...")
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_filtered, y_train_filtered)

X_test = vectorizer.transform(test["text"])
y_test = test["label"]
preds = clf.predict(X_test)

# Get the classification report as a string
snorkel_report = classification_report(y_test, preds, target_names=["neg", "pos"])
print("\nPerformance of classifier trained with Snorkel labels:")
print(snorkel_report) # Also print to console

# 6. (For Comparison) Train a Fully Supervised Classifier
print("\nTraining a fully supervised classifier for comparison...")
X_train_full = vectorizer.transform(train["text"])

clf_fs = LogisticRegression(max_iter=200)
clf_fs.fit(X_train_full, train["label"])
fs_preds = clf_fs.predict(X_test)

# Get the second classification report as a string
supervised_report = classification_report(y_test, fs_preds, target_names=["neg", "pos"])
print("\nFully supervised performance:")
print(supervised_report) # Also print to console

# 7. <<-- NEW SECTION TO SAVE TO FILE -->>
# Save the performance metrics to a text file
with open("performance_report.txt", "w") as f:
    f.write("--- Performance of Classifier Trained with Snorkel Labels ---\n")
    f.write(snorkel_report)
    f.write("\n\n")
    f.write("--- Performance of Fully Supervised Classifier ---\n")
    f.write(supervised_report)

print("\n✅ Performance metrics saved to 'performance_report.txt'")