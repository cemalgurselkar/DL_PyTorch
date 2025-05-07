import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import product

texts = """Bu kulaklığı yaklaşık üç haftadır kullanıyorum ve genel anlamda oldukça memnun kaldım.
Kutusundan çıkar çıkmaz şarjı neredeyse tam doluydu ve ilk bağlantıyı saniyeler içinde gerçekleştirdim. 
Özellikle telefon görüşmeleri sırasında karşı tarafın sesini net bir şekilde duyabiliyor olmak benim için büyük bir artı. 
Ses kalitesi, fiyatına göre gerçekten etkileyici.
Baslar güçlü, tizler net ve dengeli.
Ayrıca dış mekânda kullanırken de ortam seslerini bir miktar filtreliyor olması işlevsel olmuş."""

words = texts.replace(".","").replace("!","").lower().split()

word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)#type:ignore

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

#eğitim verisi hazırlama
data = [(words[i], words[i+1]) for i in range(len(words)-1)]

class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self,x):
        x = self.embedding(x) #input --> embedding
        lst_out,_ = self.lstm(x.view(1,1,-1))
        output = self.fc(lst_out.view(1,-1))
        return output

#kelime listesi -> tensor
def prepare_seq(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)

#hyperparameters tuning
embedding_size = [2,4,8,16]
hidden_size = [16,32,64,128]
learning_rate = [0.01, 0.005,0.003]

best_loss = float("inf") #en düşük kayip saklamak için
best_params = {}

print("Hyperparameter turing is starting....")

for emb_size, hid_size, lr in product(embedding_size, hidden_size, learning_rate):
    model = LSTM(vocab_size=len(vocab),embedding_dim=emb_size, hidden_dim=hid_size)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for word, next_word in data:
            model.zero_grad()
            input_tensor = prepare_seq([word], word_to_ix)
            target_tensor = prepare_seq([next_word], word_to_ix)
            output = model(input_tensor)
            loss = loss_func(output, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{epochs}, Loss: {epoch_loss:.5f}")

    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_dim":emb_size, "hidden_size":hid_size, "learning_rate":lr}
    print()

print(f"Best params: {best_params}")

final_model = LSTM(len(vocab), best_params["embedding_dim"], best_params["hidden_size"])
optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
loss_function = nn.CrossEntropyLoss()

print("Final model Training")
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_seq([word], word_to_ix)
        target_tensor = prepare_seq([next_word], word_to_ix)
        output_tensor = final_model(input_tensor)
        loss = loss_function(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Final Model Epoch: {epoch}/{epochs}, Loss: {epoch_loss}")

def predicted_seq(start_word, num_words):
    current_words = start_word
    output_seq = [current_words]

    for _ in range(num_words):
        with torch.no_grad():
            input_tensor = prepare_seq([current_words], word_to_ix) 
            output = final_model(input_tensor)
            predicted_idx = torch.argmax(output).item() #en yüksek olasılıga sahip kelimenin indexi
            predicted_words = ix_to_word[predicted_idx] # type: ignore
            output_seq.append(predicted_words)
            current_words = predicted_words #bir sonraki tahmin için mevcut kelimeleri güncelle.
    return output_seq

start_word = "tam"
num_prediction = 5
predicted = predicted_seq(start_word, num_prediction)
print(predicted)