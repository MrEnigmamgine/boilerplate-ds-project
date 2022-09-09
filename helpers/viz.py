import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_wordcloud_from_freq(freq , figsize=[12,9]):
    
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    cloud = WordCloud()
    cloud.generate_from_frequencies(freq)
    
    plt.figure(figsize=figsize)
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()

def plot_wordcloud(input , figsize=[12,9], title=None, save_path=None):
    
    from wordcloud import WordCloud
    cloud = WordCloud()
    input_type = type(input)
    if input_type == str:
        cloud.generate(input)
    
    if input_type == list:
        input = pd.Series(input)
        input_type = type(input)

    if input_type == pd.core.series.Series:
        if input.dtype != 'int64':
            s = ' '.join(input)
            s = pd.Series(s.split(' '))
            input = s.value_counts()
        input = input.to_dict()
        input_type = type(input)
    
    if input_type == dict:
        output = {}
        for k, v in input.items():
            if type(k) != str:
                if type(k) == tuple:
                    output[' '.join(k)] = v
                else:
                    output[str(k)] = v
            else:
                output[k] = v
        cloud.generate_from_frequencies(output)
    
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.imshow(cloud)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()

def get_n_colors(n: int, palette='bright') -> list:
    """Gets a list consisting of n colors."""
    import seaborn as sns
    colors = []
    for i in range(n):
        colors.append(sns.color_palette(palette=palette)[i])
    return colors

def cat_to_colors(series):
    """Given a series, return a series of equal length consisting of colors."""
    cats = series.value_counts().index.to_list()
    colors = get_n_colors(len(cats))
    x = series.apply(lambda x: colors[cats.index(x)])
    return x