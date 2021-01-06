import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import json

#load emnist
def loaddata(images_train, labels_train, images_test, labels_test):
    A_train = []
    A_test = []
    B_train = []
    B_test = []
    C_train = []
    C_test = []
    D_train = []
    D_test = []
    E_train = []
    E_test = []
    F_train = []
    F_test = []
    G_train = []
    G_test = []
    H_train = []
    H_test = []
    I_train = []
    I_test = []
    J_train = []
    J_test = []
    K_train = []
    K_test = []
    L_train = []
    L_test = []
    M_train = []
    M_test = []
    N_train = []
    N_test = []
    O_train = []
    O_test = []
    P_train = []
    P_test = []
    Q_train = []
    Q_test = []
    R_train = []
    R_test = []
    S_train = []
    S_test = []
    T_train = []
    T_test = []
    U_train = []
    U_test = []
    V_train = []
    V_test = []
    W_train = []
    W_test = []
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Z_train = []
    Z_test = []

    for i in range(0, len(labels_train)):
        if labels_train[i] == 1:
            A_train.append(images_train[i])
        if labels_train[i] == 2:
            B_train.append(images_train[i])
        if labels_train[i] == 3:
            C_train.append(images_train[i])
        if labels_train[i] == 4:
            D_train.append(images_train[i])
        if labels_train[i] == 5:
            E_train.append(images_train[i])
        if labels_train[i] == 6:
            F_train.append(images_train[i])
        if labels_train[i] == 7:
            G_train.append(images_train[i])
        if labels_train[i] == 8:
            H_train.append(images_train[i])
        if labels_train[i] == 9:
            I_train.append(images_train[i])
        if labels_train[i] == 10:
            J_train.append(images_train[i])
        if labels_train[i] == 11:
            K_train.append(images_train[i])
        if labels_train[i] == 12:
            L_train.append(images_train[i])
        if labels_train[i] == 13:
            M_train.append(images_train[i])
        if labels_train[i] == 14:
            N_train.append(images_train[i])
        if labels_train[i] == 15:
            O_train.append(images_train[i])
        if labels_train[i] == 16:
            P_train.append(images_train[i])
        if labels_train[i] == 17:
            Q_train.append(images_train[i])
        if labels_train[i] == 18:
            R_train.append(images_train[i])
        if labels_train[i] == 19:
            S_train.append(images_train[i])
        if labels_train[i] == 20:
            T_train.append(images_train[i])
        if labels_train[i] == 21:
            U_train.append(images_train[i])
        if labels_train[i] == 22:
            V_train.append(images_train[i])
        if labels_train[i] == 23:
            W_train.append(images_train[i])
        if labels_train[i] == 24:
            X_train.append(images_train[i])
        if labels_train[i] == 25:
            Y_train.append(images_train[i])
        if labels_train[i] == 26:
            Z_train.append(images_train[i])
            
    for i in range(0, len(labels_test)):
        if labels_test[i] == 1:
            A_test.append(images_test[i])
        if labels_test[i] == 2:
            B_test.append(images_test[i])
        if labels_test[i] == 3:
            C_test.append(images_test[i])
        if labels_test[i] == 4:
            D_test.append(images_test[i])
        if labels_test[i] == 5:
            E_test.append(images_test[i])
        if labels_test[i] == 6:
            F_test.append(images_test[i])
        if labels_test[i] == 7:
            G_test.append(images_test[i])
        if labels_test[i] == 8:
            H_test.append(images_test[i])
        if labels_test[i] == 9:
            I_test.append(images_test[i])
        if labels_test[i] == 10:
            J_test.append(images_test[i])
        if labels_test[i] == 11:
            K_test.append(images_test[i])
        if labels_test[i] == 12:
            L_test.append(images_test[i])
        if labels_test[i] == 13:
            M_test.append(images_test[i])
        if labels_test[i] == 14:
            N_test.append(images_test[i])
        if labels_test[i] == 15:
            O_test.append(images_test[i])
        if labels_test[i] == 16:
            P_test.append(images_test[i])
        if labels_test[i] == 17:
            Q_test.append(images_test[i])
        if labels_test[i] == 18:
            R_test.append(images_test[i])
        if labels_test[i] == 19:
            S_test.append(images_test[i])
        if labels_test[i] == 20:
            T_test.append(images_test[i])
        if labels_test[i] == 21:
            U_test.append(images_test[i])
        if labels_test[i] == 22:
            V_test.append(images_test[i])
        if labels_test[i] == 23:
            W_test.append(images_test[i])
        if labels_test[i] == 24:
            X_test.append(images_test[i])
        if labels_test[i] == 25:
            Y_test.append(images_test[i])
        if labels_test[i] == 26:
            Z_test.append(images_test[i])

    list_train = [A_train, B_train, C_train, D_train, E_train, F_train, G_train, H_train, I_train, J_train, K_train, L_train, M_train, N_train, O_train, P_train, Q_train, R_train, S_train, T_train, U_train, V_train, W_train, X_train, Y_train, Z_train]
    list_test = [A_test, B_test, C_test, D_test, E_test, F_test, G_test, H_test, I_test, J_test, K_test, L_test, M_test, N_test, O_test, P_test, Q_test, R_test, S_test, T_test, U_test, V_test, W_test, X_test, Y_test, Z_test]

    return list_train, list_test
#main
#train_image, test_image = loaddata(images_train, images_test, labels_train, labels_test)
#print(len(train_image))
#print(len(test_image))

#with open("Inputdata/train_A.txt", 'w') as f:
#    f.write(json.dumps(A_train))


#read data
#with open("Inputdata/train_A.txt", 'r') as f:
#    A_train = json.loads(f.read())