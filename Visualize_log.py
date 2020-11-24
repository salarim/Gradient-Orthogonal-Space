import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    viz_step_size = 10.0
    file_path = 'Orthogonal-56082350.out'
    output_path = 'step_size_{}.png'.format(viz_step_size)

    accs = {}

    with open(file_path, 'r') as f:
        step_size = 0.0
        index = 0
        acc = 0.0
        for line in f:
            splited_line = line.split(' ')
            if line.startswith('###'):
                index = int(splited_line[-1])
            elif line.startswith('Step'):
                step_size = float(splited_line[-1])
            elif line.startswith('Test'):
                acc = float(splited_line[-1][1:-2])
                if step_size not in accs:
                    accs[step_size] = []
                accs[step_size].append(acc)


    data = np.array(accs[viz_step_size])

    bins = np.arange(0, 100, 1.0)

    plt.xlim([min(data)-5, max(data)+5])

    plt.hist(data, bins=bins, weights=np.ones(len(data)) / len(data))
    plt.xlabel('acc of models with distance of {}'.format(viz_step_size))
    plt.ylabel('percentage')

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(output_path)


if __name__ == '__main__':
    main()
