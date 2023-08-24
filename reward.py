import json
def base_reward(label, prediction, partial_credit=False):
    label = json.loads(label)
    n = len(label)
    try: 
        prediction = json.loads(prediction)
    except:
        return 0

    total = 0
    for y, y_hat in zip(label, prediction):
        if partial_credit:
            y = y.split('/')
            y_hat = y_hat.split('/')
            tt = 0
            for i in range(len(y)):
                if i < len(y_hat):
                    if y_hat[i] == y[i]:
                        tt += 1
                else:
                    tt -= 1
            total += max(0, tt / len(y))  
        else:
            if 'category' in y_hat and y['category'] == y_hat['category']:
                total += 1
    return total / n