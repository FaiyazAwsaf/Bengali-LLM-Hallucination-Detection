import pandas as pd

submission = pd.read_csv('data/sample submission.csv')
submission['label'] = 0
submission.to_csv('submissions/all_zeros.csv', index=False)
print("Saved all_zeros.csv")