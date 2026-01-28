"""
Create a mock sports milestone dataset and check distribution.
"""

import pandas as pd
import random

# Create a mock dataset similar to sports-milestone classifier
# Let's simulate 200 total samples with ~50% milestone / ~50% non-milestone
random.seed(42)

data = []
for i in range(200):
    # Alternate between milestone and non-milestone with some randomness
    is_milestone = random.random() < 0.5
    data.append({
        'input': f"Sample article {i}...",
        'milestone': is_milestone
    })

# Convert to DataFrame
dataset_df = pd.DataFrame(data)

print(f"📥 Created mock dataset")
print(f"Total dataset size: {len(dataset_df)}")
print(f"Columns: {list(dataset_df.columns)}")

# Shuffle with same seed as test would use
random.seed(42)
indices = list(range(len(dataset_df)))
random.shuffle(indices)

# Split same way as test
num_train = 50
num_test = 20
train_indices = indices[:num_train]
test_indices = indices[num_train:num_train + num_test]

train_data = dataset_df.iloc[train_indices]
test_data = dataset_df.iloc[test_indices]

label_col = 'milestone'

# Check distribution
train_true = train_data[label_col].sum()
train_false = num_train - train_true
test_true = test_data[label_col].sum()
test_false = num_test - test_true

print(f'\n{"="*80}')
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
for i, (idx, row) in enumerate(train_data.head(10).iterrows()):
    label = "✅ True " if row[label_col] else "❌ False"
    print(f'  Sample {i:2d}: {label}')

print(f'\n{"="*80}')
print(f'🔍 First 10 TEST samples (labels):')
print(f'{"="*80}')
for i, (idx, row) in enumerate(test_data.head(10).iterrows()):
    label = "✅ True " if row[label_col] else "❌ False"
    print(f'  Sample {i:2d}: {label}')

print(f'\n{"="*80}')
print(f'💡 KEY FINDINGS:')
print(f'{"="*80}')
if train_balanced and test_balanced and similar:
    print(f'  ✅ Distribution looks good!')
    print(f'  ✅ Both sets are balanced (not skewed to one class)')
    print(f'  ✅ Train and test have similar distributions')
else:
    if not train_balanced:
        print(f'  ⚠️  Training set is imbalanced (too many True or False)')
    if not test_balanced:
        print(f'  ⚠️  Test set is imbalanced (too many True or False)')
    if not similar:
        print(f'  ⚠️  Train and test have different distributions')
        print(f'     This can cause the model to overfit to train distribution')
