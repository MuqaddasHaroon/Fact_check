import ast
import torch
def extract_original_from_list(entry):
    try:
        parsed = ast.literal_eval(entry)
        if isinstance(parsed, list) and parsed:
            return parsed[0][0].strip()
    except:
        pass
    return entry

def extract_original_from_tuple(entry):
    try:
        parsed = ast.literal_eval(entry)
        if isinstance(parsed, tuple) and parsed:
            return parsed[0].strip()
    except:
        pass
    return entry

def custom_collate_fn(tokenizer):
    def collate_fn(batch):
        texts1 = [example.texts[0] for example in batch]
        texts2 = [example.texts[1] for example in batch]
        labels = [example.label for example in batch]
        features1 = tokenizer(texts1, padding=True, truncation=True, return_tensors="pt")
        features2 = tokenizer(texts2, padding=True, truncation=True, return_tensors="pt")
        return [features1, features2], torch.tensor(labels, dtype=torch.float)
    return collate_fn
