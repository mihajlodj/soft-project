from matplotlib import pyplot as plt

def read_output():
    loss = []
    accuracy = []

    file = open('training_output.txt', 'r')
    line = file.readline()
    while line != '':
        rowdata = line.split('-')
        if len(rowdata) > 1:
            acctxt = rowdata[-1]
            losstxt = rowdata[-2]
            accuracy.append(float(acctxt.split(': ')[1]))
            loss.append(float(losstxt.split(': ')[1]))

        line = file.readline()
    return loss, accuracy


if __name__ == '__main__':
    loss, accuracy = read_output()
    fig, ax = plt.subplots(ncols=2, figsize=(20, 5))

    ax[0].plot(loss, color='teal', label='loss')
    ax[0].title.set_text('Training loss')
    ax[0].legend()

    ax[1].plot(accuracy, color='teal', label='accuracy')
    ax[1].title.set_text('Training accuracy')
    ax[1].legend()

    plt.show()
