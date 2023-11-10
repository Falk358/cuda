import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt




def main():
    objects = ('No optimizations', 'Compressed matrix', 'Compressed matrix + \n more threads and blocks', 'Compressed matrix + \n more threads and blocks + \n swap pointers')
    y_pos = np.arange(len(objects))
    performance = [9.28633, 0.0353552, 0.0267005, 0.0236252]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation = 30)
    plt.ylabel('Measured compute time in seconds (log scale)')
    plt.title('Exercise_efficiency performance measurements')
    plt.yscale("log")
    
    plt.tight_layout()
    plt.savefig("exercise_3.jpeg")


if __name__ == "__main__":
    main()
