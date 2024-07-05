import torch

def get_verdict(pred):
    assert len(pred) == 3
    return torch.argmax(pred)

def max_label(pred):
    # B 75
    
    result = torch.zeros(25, 3)
    max_verdict = torch.zeros(25)
    for one_pred in pred:
        for i in range(25):
            verdict = get_verdict(one_pred[i * 3 : (i + 1) * 3])
            if verdict > max_verdict[i]:
                max_verdict[i] = verdict
                result[i] = one_pred[i * 3 : (i + 1) * 3]
            elif verdict == max_verdict[i] and \
                 result[i, verdict] < one_pred[i * 3 + verdict]:
                result[i] = one_pred[i * 3 : (i + 1) * 3]
    result = result.view(75)
    return result