import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # kelime frekanslarini hesaplamak icin
from itertools import product # grid search icin kombinasyon olusturmak

text = """Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi.
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.
Fiyatına göre performansı harika.
Kesinlikle tavsiye ederim ve öneririm!"""

words = text.replace(".", "").replace("!","").lower().split()

word_count = Counter(words)
vocab = sorted(word_count, key=word_count.get, reverse=True) # type: ignore
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

data = [(words[i], words[i+1]) for i in range(len(words)-1)]

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
            input -> embedding -> lstm -> fc -> output
        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output = self.fc(lstm_out.view(1,-1))
        return output

model = LSTM(len(vocab), embedding_dim=8, hidden_dim=32)

def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)

embedding_sizes = [8,16]
hidden_sizes = [32,64]
learning_rates = [0.01, 0.005]

best_loss = float("inf")
best_params = {}

for emd_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    print(f"Deneme: Embedding: {emd_size}, Hidden: {hidden_size}, learning_rate: {lr}")

    model = LSTM(len(vocab), emd_size, hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epochs_loss = 0
        for word, next_word in data:
            model.zero_grad()
            input_tensor = prepare_sequence([word], word_to_ix)
            target_tensor = prepare_sequence([next_word], word_to_ix)
            output = model(input_tensor)
            loss = loss_function(output, target_tensor)
            loss.backward()
            optimizer.step()
            epochs_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {epochs_loss:.5f}")
        total_loss = epochs_loss
    
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_dim":emd_size, "hidden_dim":hidden_size, "learning_rate":lr}
    print()
print(f"Best Params: {best_params}")

final_model = LSTM(len(vocab), best_params["embedding_dim"],best_params["hidden_dim"])
optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
loss_function = nn.CrossEntropyLoss()

print("Final model Training")
epochs = 100
for epoch in range(epochs):
    epochs_loss = 0
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_sequence([word],word_to_ix)
        target_tensor = prepare_sequence([next_word], word_to_ix)
        output = final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epochs_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Final Model Epoch: {epoch}, Loss: {epochs_loss:.5f}")

def predict_sequence(start_word, num_word):
    current_word = start_word
    output_sequence = [current_word]

    for _ in range(num_word):
        with torch.no_grad():
            input_tensor = prepare_sequence([current_word], word_to_ix)
            output = final_model(input_tensor)
            predicted_idx = torch.argmax(output).item()
            predicted_word = ix_to_word[predicted_idx]#type:ignore
            output_sequence.append(predicted_word)
            current_word = predicted_word
    return output_sequence

"""
Bu ürün beklentimi fazlasıyla karşıladı.  
Malzeme kalitesi gerçekten çok iyi.  
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.  
Fiyatına göre performansı harika.  
Kesinlikle tavsiye ederim ve öneririm!
"""


start_word = "ve"
num_predictions = 7
predicted_sequence = predict_sequence(start_word, num_predictions)
print(" ".join(predicted_sequence))