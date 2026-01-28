"""
Check train/test distribution - try different dataset names.
"""

from datasets import load_dataset
import random

# Try different possible dataset names
possible_names = [
    'agentlens/sports-milestone-classifier',
    'agentlens/sport-milestone-classifier',
    'agentlens/sports-milestone',
    'agentlens/sport-milestone',
]

dataset = None
dataset_name = None

for name in possible_names:
    try:
        print(f"Trying: {name}...")
        dataset = load_dataset(name, split='train')
        dataset_name = name
        print(f"✅ Found! Using: {name}\n")
        break
    except Exception as e:
        print(f"❌ Not found: {name}")
        continue

if dataset is None:
    print("\n❌ Could not find any valid dataset name!")
    print("Please check the correct dataset name on HuggingFace Hub.")
    exit(1)

print(f"📥 Loaded dataset: {dataset_name}")
print(f"Total dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Shuffle with same seed as test_optimization.py
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)

# Split same way
num_train = 50
num_test = 20
train_data = dataset.select(indices[:num_train])
test_data = dataset.select(indices[num_train:num_train + num_test])

# Find the label column
label_col = None
for col in ['milestone', 'label', 'is_milestone']:
    if col in dataset.column_names:
        label_col = col
        break

if label_col is None:
    print(f"❌ Could not find label column! Available: {dataset.column_names}")
    exit(1)

print(f"Label column: {label_col}\n")

# Check distribution
train_true = sum(1 for x in train_data if x[label_col])
train_false = num_train - train_true
test_true = sum(1 for x in test_data if x[label_col])
test_false = num_test - test_true

print(f'{"="*80}')
print(f'📊 TRAIN SET ({num_train} samples):')
print(f'{"="*80}')
print(f'  ✅ True:  {train_true:2d} ({train_true/num_train*100:5.1f}%)')
print(f'  ❌ False: {train_false:2d} ({train_false/num_train*100:5.1f}%)')

print(f'\n{"="*80}')
print(f'📊 TEST SET ({num_test} samples):')
print(f'{"="*80}')
print(f'  ✅ True:  {test_true:2d} ({test_true/num_test*100:5.1f}%)')
print(f'  ❌ False: {test_false:2d} ({test_false/num_test*100:5.1f}%)')

print(f'\n{"="*80}')
print(f'📊 OVERALL BALANCE:')
print(f'{"="*80}')
print(f'  Train ratio (True/Total): {train_true/num_train:.3f}')
print(f'  Test ratio (True/Total):  {test_true/num_test:.3f}')
print(f'  Difference: {abs(train_true/num_train - test_true/num_test):.3f}')

# Check if balanced
train_balanced = 0.3 <= train_true/num_train <= 0.7
test_balanced = 0.3 <= test_true/num_test <= 0.7
similar = abs(train_true/num_train - test_true/num_test) < 0.15

print(f'\n{"="*80}')
print(f'✅ BALANCE CHECK:')
print(f'{"="*80}')
print(f'  Train set balanced (30-70%): {"✅ YES" if train_balanced else "❌ NO"}')
print(f'  Test set balanced (30-70%):  {"✅ YES" if test_balanced else "❌ NO"}')
print(f'  Train/Test similar (<15%):   {"✅ YES" if similar else "❌ NO"}')

# Show first 10 samples from each set
print(f'\n{"="*80}')
print(f'🔍 First 10 TRAIN samples (labels):')
print(f'{"="*80}')
for i in range(min(10, num_train)):
    label = "✅ True " if train_data[i][label_col] else "❌ False"
    print(f'  Sample {i:2d}: {label}')

print(f'\n{"="*80}')
print(f'🔍 First 10 TEST samples (labels):')
print(f'{"="*80}')
for i in range(min(10, num_test)):
    label = "✅ True " if test_data[i][label_col] else "❌ False"
    print(f'  Sample {i:2d}: {label}')
