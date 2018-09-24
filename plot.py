# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plot:
    def __init__(self, vanilla, double, dualing, double_dualing):
        self.df = pd.DataFrame({'x': range(1,len(vanilla) + 1), 'vanilla': vanilla,
                           'double': double,
                           'dualing': dualing,
                           'double_dualing': double_dualing})


    def make_plot(self):
        # style
        plt.style.use('seaborn-darkgrid')

        # create a color palette
        palette = plt.get_cmap('Set1')

        # multiple line plot
        num = 0
        for column in self.df.drop('x', axis=1):
            num += 1
            plt.plot(self.df['x'], self.df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

        # Add legend
        plt.legend(loc=2, ncol=2)

        # Add titles
        plt.title("Plot of the training scores vs episodes for several q learning types", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("# Episode")
        plt.ylabel("Score")

        plt.savefig('./results_graph.png')



class PlotSingle:
    def __init__(self, scores, model_type):
        self.df = pd.DataFrame({'x': range(1,len(scores) + 1), 'scores': scores})
        self.model_type = model_type


    def make_plot(self):
        # style
        plt.style.use('seaborn-darkgrid')

        # create a color palette
        palette = plt.get_cmap('Set1')

        # multiple line plot
        num = 0
        for column in self.df.drop('x', axis=1):
            num += 1
            plt.plot(self.df['x'], self.df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

        # Add legend
        plt.legend(loc=2, ncol=2)

        # Add titles
        plt.title("Plot of the training scores vs episodes for " + self.model_type, loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("# Episode")
        plt.ylabel("Score")

        plt.savefig('./results_graph.png')