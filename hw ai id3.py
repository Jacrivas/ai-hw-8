import pandas as pd
import numpy as np
from collections import Counter

data = {
    "Toothed": ["Toothed", "Toothed", "Toothed", "Toothed", "Not Toothed",
                "Toothed", "Toothed", "Toothed", "Toothed", "Not Toothed"],
    "Hair": ["Hair", "Hair", "Hair", "Not Hair", "Hair",
             "Hair", "Hair", "Not Hair", "Not Hair", "Not Hair"],
    "Breathes": ["Breathes", "Breathes", "Breathes", "Breathes", "Breathes",
                 "Breathes", "Breathes", "Not Breathes", "Breathes", "Breathes"],
    "Legs": ["Legs", "Legs", "Legs", "Not Legs", "Legs",
             "Legs", "Legs", "Not Legs", "Not Legs", "Legs"],
    "Species": ["Mammal", "Mammal", "Mammal", "Reptile", "Mammal",
                "Mammal", "Mammal", "Reptile", "Reptile", "Reptile"]
}

df = pd.DataFrame(data)


def entropy(column):
    elements, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    return -sum(probabilities * np.log2(probabilities))


def information_gain(df, attribute, target="Species"):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[attribute], return_counts=True)

    weighted_entropy = sum(
        (counts[i] / counts.sum()) * entropy(df[df[attribute] == values[i]][target]) for i in range(len(values)))

    return total_entropy - weighted_entropy


def id3(df, target="Species", attributes=None, parent_class=None):
    if attributes is None:
        attributes = df.columns[:-1]

    if len(np.unique(df[target])) == 1:
        return np.unique(df[target])[0]

    elif len(attributes) == 0:
        return parent_class

    else:
        parent_class = Counter(df[target]).most_common(1)[0][0]

        gains = [information_gain(df, attr, target) for attr in attributes]
        best_attr = attributes[np.argmax(gains)]

        tree = {best_attr: {}}

        remaining_attrs = [attr for attr in attributes if attr != best_attr]

        for value in np.unique(df[best_attr]):
            subset = df[df[best_attr] == value]

            subtree = id3(subset, target, remaining_attrs, parent_class)
            tree[best_attr][value] = subtree

        return tree


decision_tree = id3(df)

print(decision_tree)
