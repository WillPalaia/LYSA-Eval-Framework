"""
All credit for help in this project will go to the following resources:

https://www.youtube.com/watch?v=QpzMWQvxXWk&t=2064
https://www.youtube.com/watch?v=szczpgOEdXs&t=193s
https://www.analyticsvidhya.com/blog/2021/05/how-to-build-word-cloud-in-python/
https://www.youtube.com/watch?v=DkzbCJtFvqM

These resources provided the necessary foundation for learning how to optimize the pre-trained BERT model for my dataset.

"""
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from wordcloud import WordCloud

RANDOM_SEED = 4
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

file_path = 'reviews7.csv'
df = pd.read_csv(file_path)
df.columns = ["PlayerName", "Technical", "Tactical", "Effort", "Overall", "Comments"]
df = df.iloc[4:].reset_index(drop=True)

# Convert ratings to sentiment labels
def to_sentiment(rating):
    rating = float(rating)
    if rating < 6:
        return 0
    else:
        return 1

df['Sentiment'] = df['Overall'].apply(to_sentiment)
ax = sns.countplot(x='Sentiment', data=df)
plt.xlabel('Sentiment')
countLabels = ['mediocre', 'positive']
ax.set_xticklabels(countLabels)
plt.show()

# Data preprocessing
df['Overall'] = pd.to_numeric(df['Overall'], errors='coerce')
df = df.dropna(subset=['Overall'])
df['Overall'] = np.floor(df['Overall'])

# Graph to visualize distribution of overall ratings
sns.countplot(x='Overall', data=df)
plt.xlabel('Overall Rating')
plt.show()

#Splitting training for different sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

# Dataset class for loading data
class PlayerReviewDataset(Dataset):
    def __init__(self, comments, targets, tokenizer, max_len):
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# Tokenizer and data loaders
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 180
BATCH_SIZE = 8
EPOCHS = 10

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PlayerReviewDataset(
        comments=df.Comments.to_numpy(),
        targets=df.Sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0  # Set to 0 if running on systems where multi-threading might cause issues
    )

train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# Load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')


class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


model = SentimentClassifier(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)



# Evaluation function
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)




history = defaultdict(lambda: defaultdict(list))
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} Train accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val loss {val_loss} Val accuracy {val_acc}')
    print()

    history[epoch]['train_acc'] = train_acc
    history[epoch]['train_loss'] = train_loss
    history[epoch]['val_acc'] = val_acc
    history[epoch]['val_loss'] = val_loss

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc


# Print classification report
y_pred = []
y_true = []

model = model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        y_pred.extend(preds)
        y_true.extend(targets)

y_pred = torch.stack(y_pred).cpu().numpy()
y_true = torch.stack(y_true).cpu().numpy()

unique_labels = np.unique(np.concatenate((y_true, y_pred)))

print(classification_report(y_true, y_pred, labels=unique_labels, target_names=[countLabels[i] for i in unique_labels]))

# Displaying confusion matrix
def show_confusion_matrix(conf_matrix):
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.show()

cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
show_confusion_matrix(cm)

# Example prediction
encoded_review = tokenizer.encode_plus(
    example_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
example_text = "This player is bad! They're a terrible player! They don't do a bad job on the field! They are always very lazy on the field"
print(f'Review text: {example_text}')
print(f'Sentiment  : {countLabels[prediction]}')

# Word cloud generation for positive reviews with n-grams
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

positive_reviews = df[df['Sentiment'] >= 1]['Comments']

# Create the CountVectorizer with n-grams (e.g., unigrams, bigrams, trigrams)
vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
X = vectorizer.fit_transform(positive_reviews)

# Sum up the counts of each vocabulary word
word_counts = X.sum(axis=0)
word_counts = word_counts.tolist()[0]

# Get the words themselves
words = vectorizer.get_feature_names_out()

# Create a dictionary of words and their counts
word_freq = {word: count for word, count in zip(words, word_counts)}

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
