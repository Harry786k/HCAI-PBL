import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# TASK 1: LOAD DATA AND BASELINE MODEL
# ==========================================
print("Loading AG News dataset (this might take a few seconds)...")
dataset = load_dataset("fancyzhx/ag_news")

train_data = dataset['train']
test_data = dataset['test']

X_train_raw = train_data['text']
y_train = train_data['label']

X_test_raw = test_data['text']
y_test = test_data['label'] # <--- This is the variable your script was missing!

print("Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print("Training Baseline AI Model...")
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)
print(f"\n--- Baseline AI Test Accuracy: {baseline_acc * 100:.2f}% ---")

# ==========================================
# TASK 2: SIMULATE THE HUMAN EXPERT
# ==========================================
def simulate_expert(true_labels, random_seed=42):
    np.random.seed(random_seed)
    expert_predictions = []
    
    for label in true_labels:
        chance = np.random.rand() 
        if label == 1:    # Sports: 95% accurate
            pred = label if chance < 0.95 else np.random.choice([0, 2, 3])
        elif label == 2:  # Business: 90% accurate
            pred = label if chance < 0.90 else np.random.choice([0, 1, 3])
        elif label == 0:  # World: 60% accurate
            pred = label if chance < 0.60 else np.random.choice([1, 2, 3])
        else:             # Sci/Tech: 40% accurate
            pred = label if chance < 0.40 else np.random.choice([0, 1, 2])
        expert_predictions.append(pred)
        
    return np.array(expert_predictions)

print("\nSimulating expert predictions on the test set...")
expert_preds_test = simulate_expert(y_test)
expert_accuracy = accuracy_score(y_test, expert_preds_test)

print(f"\n--- Simulated Expert Test Accuracy: {expert_accuracy * 100:.2f}% ---")
print("\nExpert's Detailed Report:")
print(classification_report(y_test, expert_preds_test, target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# ==========================================
# TASK 3: LEARNING TO DEFER (L2D)
# ==========================================
print("\n==========================================")
print("TASK 3: LEARNING TO DEFER")
print("==========================================")
print("Simulating expert on training data to learn when to defer...")
expert_preds_train = simulate_expert(y_train)

# 1. Create the target for the Allocator (Learn WHEN to defer)
# Rule: We only want to defer if the AI is WRONG and the Expert is CORRECT.
ai_preds_train = baseline_model.predict(X_train)
should_defer_train = ((ai_preds_train != y_train) & (expert_preds_train == y_train)).astype(int)

# 2. Train the Deferral Classifier (The Allocator)
print("Training the Allocator Model...")
allocator_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
allocator_model.fit(X_train, should_defer_train)

# 3. Let the Team take the Test
# The Allocator decides 1 (Defer to Expert) or 0 (Let AI handle it)
print("Evaluating the Human-AI Team...")
defer_decisions = allocator_model.predict(X_test)

team_predictions = []
for i in range(len(y_test)):
    if defer_decisions[i] == 1:
        team_predictions.append(expert_preds_test[i]) # Defer to expert
    else:
        team_predictions.append(y_pred[i]) # Use Baseline AI (y_pred from Task 1)

# 4. Grade the Team
team_accuracy = accuracy_score(y_test, team_predictions)
deferred_count = sum(defer_decisions)
total_test = len(y_test)

print(f"\n--- Final Results ---")
print(f"1. Baseline AI Alone: {baseline_acc * 100:.2f}%")
print(f"2. Human Expert Alone: {expert_accuracy * 100:.2f}%")
print(f"3. Human-AI Team:     {team_accuracy * 100:.2f}%")
print(f"\nDeferral Stats:")
print(f"The AI deferred {deferred_count} out of {total_test} articles to the expert ({(deferred_count/total_test)*100:.2f}%).")
# Deferral quality metrics
y_test_np = np.array(y_test)
defer_mask = defer_decisions == 1
keep_mask = defer_decisions == 0

expert_acc_deferred = accuracy_score(
    y_test_np[defer_mask],
    expert_preds_test[defer_mask]
)

ai_acc_deferred = accuracy_score(
    y_test_np[defer_mask],
    y_pred[defer_mask]
)

ai_acc_not_deferred = accuracy_score(
    y_test_np[keep_mask],
    y_pred[keep_mask]
)

print("\n--- Deferral Quality ---")
print(f"Deferral Rate: {(deferred_count / total_test) * 100:.2f}%")
print(f"Expert Accuracy on Deferred Examples: {expert_acc_deferred * 100:.2f}%")
print(f"AI Accuracy on Deferred Examples: {ai_acc_deferred * 100:.2f}%")
print(f"AI Accuracy on Non-Deferred Examples: {ai_acc_not_deferred * 100:.2f}%")
# ==========================================
# TASK 4: ACTIVE LEARNING (UNCERTAINTY SAMPLING)
# ==========================================
print("\n==========================================")
print("TASK 4: ACTIVE LEARNING")
print("==========================================")
print("Finding the articles the AI is most confused about...")

# 1. Get the AI's confidence scores for all training articles
ai_probs_train = baseline_model.predict_proba(X_train)

# 2. Calculate "Uncertainty" (1.0 minus the probability of its top guess)
# High uncertainty means the AI's top guess was very weak (e.g., only 30% sure)
uncertainty_scores = 1.0 - np.max(ai_probs_train, axis=1)

# 3. Sort the articles from most confused to least confused
most_confused_indices = np.argsort(uncertainty_scores)[::-1]

# 4. We only have budget to ask the Expert about 2000 articles
BUDGET = 2000
query_indices = most_confused_indices[:BUDGET]
print(f"Querying the expert for the top {BUDGET} most uncertain articles...")

# 5. Get the expert's predictions ONLY for those 2000 articles
expert_queried_preds = simulate_expert(y_train[query_indices])
ai_queried_preds = baseline_model.predict(X_train[query_indices])
true_queried_labels = np.array(y_train)[query_indices] # Convert to numpy array to ensure index works

# 6. Create the limited training data for our new Active Learning Allocator
should_defer_active = ((ai_queried_preds != true_queried_labels) & 
                       (expert_queried_preds == true_queried_labels)).astype(int)

print("Training the New Allocator on the limited Active Learning data...")
active_allocator = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
active_allocator.fit(X_train[query_indices], should_defer_active)

# 7. Evaluate the New Active Learning Team
print("Evaluating the New Team...")
active_defer_decisions = active_allocator.predict(X_test)

active_team_predictions = []
for i in range(len(y_test)):
    if active_defer_decisions[i] == 1:
        active_team_predictions.append(expert_preds_test[i])
    else:
        active_team_predictions.append(y_pred[i])

active_team_acc = accuracy_score(y_test, active_team_predictions)
active_deferred_count = sum(active_defer_decisions)

print(f"\n--- Active Learning Results ---")
print(f"Active Learning Team Accuracy: {active_team_acc * 100:.2f}%")
print(f"The new AI deferred {active_deferred_count} out of {total_test} articles to the expert ({(active_deferred_count/total_test)*100:.2f}%).")