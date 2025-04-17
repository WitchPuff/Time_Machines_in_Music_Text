import pandas as pd
from sklearn.model_selection import train_test_split

with open('data/text/relation_saq.csv', 'r') as f:
    data = pd.read_csv(f)
    data = data.drop(columns=['Source'])
    filter_labels = ['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS']
    
    data = data[data['Answer'].isin(filter_labels)]
    
    print(f"Total samples: {len(data)}")
    
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, valid_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print("Train data label distribution:")
    print(train_data['Answer'].value_counts())
    print(test_data['Answer'].value_counts())
    print(valid_data['Answer'].value_counts())

    train_data.to_csv('data/text/train_data.csv', index=False)
    test_data.to_csv('data/text/test_data.csv', index=False)
    valid_data.to_csv('data/text/valid_data.csv', index=False)