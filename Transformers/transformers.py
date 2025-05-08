import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter

positive_sentences = [
    "This is amazing!",
    "I love it so much!",
    "Perfect in every way.",
    "Absolutely fantastic!",
    "Couldn't be happier with this.",
    "Highly recommended!",
    "Exceeded my expectations.",
    "Wonderful experience.",
    "Great quality and value.",
    "I'm thrilled with the results.",
    "Best purchase ever!",
    "So glad I found this.",
    "Incredible performance.",
    "Top-notch service.",
    "Very impressive!",
    "Exactly what I needed.",
    "Works like a charm.",
    "Super satisfied!",
    "Brilliant product!",
    "Outstanding customer support.",
    "Pleasantly surprised!",
    "Worth every penny.",
    "A dream come true.",
    "Simply the best!",
    "Five stars!",
    "Beyond my wildest dreams.",
    "Flawless execution.",
    "I'm in love with this!",
    "A game-changer!",
    "So easy to use!",
    "Delighted with my choice.",
    "Exceptional quality.",
    "Better than expected.",
    "A must-have!",
    "Pure perfection.",
    "Fast and efficient.",
    "Truly outstanding!",
    "I can't get enough!",
    "A fantastic deal.",
    "Very user-friendly.",
    "Extremely happy!",
    "Just wow!",
    "A great investment.",
    "Smooth and seamless.",
    "I'm blown away!",
    "Reliable and effective.",
    "A joy to use.",
    "Excellent craftsmanship.",
    "Highly efficient.",
    "Perfect for my needs.",
    "Superb performance.",
    "I'm obsessed!",
    "Unbelievable value.",
    "A total win!",
    "So worth it!",
    "Beautifully designed.",
    "Very responsive.",
    "A breath of fresh air.",
    "Keeps getting better.",
    "No complaints at all.",
    "I'm a huge fan!",
    "Does the job perfectly.",
    "A stellar product.",
    "Lightning-fast!",
    "Exquisite attention to detail.",
    "A cut above the rest.",
    "Extremely versatile.",
    "I'm beyond satisfied.",
    "A real gem.",
    "Phenomenal service.",
    "Just perfect!",
    "Makes life easier.",
    "A top-tier product.",
    "Very intuitive.",
    "I'm so impressed!",
    "A masterpiece!",
    "Unmatched quality.",
    "An absolute delight.",
    "Works flawlessly.",
    "A huge improvement.",
    "The best out there.",
    "Incredibly efficient.",
    "A solid choice.",
    "Very durable.",
    "I'm speechless!",
    "A top performer.",
    "So sleek and modern.",
    "A fantastic find.",
    "Highly durable.",
    "Exceeds all standards.",
    "A real bargain.",
    "Super reliable.",
    "A pleasure to own.",
    "Very stylish.",
    "A top-quality item.",
    "I'm ecstatic!",
    "A top recommendation.",
    "So innovative!",
    "A top-seller for a reason.",
    "A winner in my book.",
    "A+ all the way!"
]
negative_sentences = [
    "I do not like it.",
    "This is terrible!",
    "Worst experience ever.",
    "Absolutely horrible.",
    "Not worth the money.",
    "I regret buying this.",
    "Poor quality product.",
    "Very disappointed.",
    "Does not work as advertised.",
    "Complete waste of time.",
    "Unbelievably bad.",
    "I would not recommend this.",
    "Fell apart immediately.",
    "Extremely frustrating.",
    "A total letdown.",
    "Not what I expected.",
    "Cheaply made.",
    "Awful customer service.",
    "Defective on arrival.",
    "Overpriced junk.",
    "Broke after one use.",
    "Misleading description.",
    "I'm so disappointed.",
    "Useless and ineffective.",
    "A huge mistake.",
    "Stay away from this.",
    "Not functional at all.",
    "Very poor performance.",
    "Waste of my money.",
    "Extremely slow.",
    "Unreliable and buggy.",
    "A disaster!",
    "Not worth it.",
    "I demand a refund.",
    "Pathetic quality.",
    "Doesn't meet standards.",
    "I'm furious!",
    "Inferior materials.",
    "Never buying again.",
    "Extremely noisy.",
    "A complete rip-off.",
    "Unusable product.",
    "Highly dissatisfied.",
    "False advertising.",
    "A nightmare to deal with.",
    "Shoddy workmanship.",
    "Terrible experience.",
    "I feel scammed.",
    "Ridiculously overpriced.",
    "Not durable at all.",
    "Broken upon delivery.",
    "Very uncomfortable.",
    "A big disappointment.",
    "Not user-friendly.",
    "Extremely difficult to use.",
    "I'm outraged!",
    "Does not last.",
    "Poorly designed.",
    "A major flaw.",
    "I want my money back.",
    "Not as described.",
    "Completely useless.",
    "Failed expectations.",
    "Very flimsy.",
    "A letdown in every way.",
    "Unacceptable quality.",
    "I'm very unhappy.",
    "Not up to par.",
    "Extremely unreliable.",
    "A frustrating purchase.",
    "Not worth the hassle.",
    "Definitely returning this.",
    "A terrible investment.",
    "Hard to assemble.",
    "Very cheaply built.",
    "I'm extremely annoyed.",
    "Does not function properly.",
    "A waste of space.",
    "Not recommended at all.",
    "Extremely overrated.",
    "I'm highly dissatisfied.",
    "A bad decision.",
    "Very slow performance.",
    "Not efficient.",
    "A faulty product.",
    "I'm very frustrated.",
    "Does not deliver.",
    "Very misleading.",
    "A huge waste.",
    "Not suitable for use.",
    "Extremely poor design.",
    "I'm thoroughly disappointed.",
    "A complete failure.",
    "Not worth the price.",
    "Very subpar.",
    "I'm deeply unhappy.",
    "A regrettable purchase.",
    "Not up to standard.",
    "Extremely defective.",
    "I'm really upset.",
    "A terrible product.",
    "Not what was promised.",
    "Very low quality.",
    "I'm extremely let down.",
    "A big mistake."
]

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

data = positive_sentences + negative_sentences
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

data = [preprocess(sentence) for sentence in data]

all_words = " ".join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0

max_len = 15
def sentences_to_tensor(sentence, vocab, max_len=15):
    tokens = sentence.split()
    indices = [vocab.get(word,0) for word in tokens]
    indices = indices[:max_len]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices)

X = torch.stack([sentences_to_tensor(sentence, vocab, max_len=max_len) for sentence in data])
y = torch.tensor(labels)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embedding_dim,num_heads,num_layers,hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1,max_len,embedding_dim))
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_layers,dim_feedforward=hidden_dim)
        self.fc = nn.Linear(embedding_dim*max_len, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        emedded = self.emb(x) + self.positional_encoding
        output = self.transformer(emedded,emedded)
        output = output.view(output.size(0), -1)
        output = torch.relu(self.fc(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output
    
vocab_size = len(vocab)
embedding_dim = 32
num_heads = 4
num_layers = 4
hidden_dim = 64
num_classes = 1
num_epoch = 10
model = TransformerClass(vocab_size, embedding_dim,num_heads, num_layers, hidden_dim, num_classes)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

model.train()
for epoch in range(num_epoch):
    optimizer.zero_grad()
    output = model(x_train.long()).squeeze()
    loss = criterion(output, y_train.float())
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {loss}")