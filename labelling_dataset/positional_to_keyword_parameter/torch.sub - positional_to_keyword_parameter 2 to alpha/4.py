import torch


def biasSVD(data, k, steps, learning_rate, l, mark_mat):

    for t in range(steps):

        user_gradient = torch.add(user_gradient, torch.mul(l, user))
        item_gradient = torch.add(item_gradient, torch.mul(l, item))
        bu_gradient = torch.add(bu_gradient, torch.mul(l, bu))
        bi_gradient = torch.add(bi_gradient, torch.mul(l, bi))
        print('----------gradient has trained----------')
        user = torch.sub(user, learning_rate * user_gradient)
        item = torch.sub(item, learning_rate * item_gradient)
        bu = torch.sub(bu, learning_rate * bu_gradient)
        bi = torch.sub(bi, learning_rate * bi_gradient)
        error = 0
        count = len(mark_mat)
        for _ in mark_mat:
            i = _[0]
            j = _[1]
            error += torch.pow(data[i][j]-torch.dot(user[i], item[j].t())-miu-bi[j]-bu[i], 2)
        print('----------round', t, '----------rmse is: ', torch.sqrt((1/count)*error))
        print('----------round', t, '----------train end')
    # test = torch.load('//TODO', map_location=torch.device('cpu'))
    # _ = test.shape[0]
    # for __ in range(_):
    #   i = test[__][0]
    #   j = test[__][0]
    #   error += torch.pow(data[i][j]-torch.dot(user[i], item[j].t())-miu-bi[j]-bu[i], 2)
    #   print(torch.sqrt((1/count)*error))


