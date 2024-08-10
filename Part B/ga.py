import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("../../iphi2802.csv", delimiter="\t", encoding="utf-8")
df = df[df["region_main_id"] == 1683]
inscriptions = df["text"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(inscriptions)
vocabulary = vectorizer.get_feature_names_out()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def create_individual(icls):
    return icls([random.randint(0, len(vocabulary) - 1) for _ in range(2)])


toolbox.register("individual", create_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness
def evaluate(individual):
    words = [vocabulary[i] for i in individual]
    completed_inscription = f"{words[0]} αλεξανδρε ουδις {words[1]}"

    completed_vector = vectorizer.transform([completed_inscription])
    similarities = cosine_similarity(
        completed_vector, tfidf_matrix[1:]
    )  # Exclude the target

    return (np.mean(similarities[0][:10]),)  # Return average similarity with top 10


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register(
    "mutate", tools.mutUniformInt, low=0, up=len(vocabulary) - 1, indpb=0.1
)
toolbox.register("select", tools.selTournament, tournsize=3)


def main(POP, CXPB, MUTPB):
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=500,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best = hof[0]
    best_words = [vocabulary[i] for i in best]
    best_inscr = f"{best_words[0]} αλεξανδρε ουδις {best_words[1]}"
    best_fitness = best.fitness.values[0]
    print(f"\t\tBest solution: {best_inscr}")
    print(f"\t\tFitness: {best_fitness}")

    return pop, log, hof, best_fitness, best_inscr, [POP, CXPB, MUTPB]


POP = [20] * 5 + [200] * 5
CXPB = [0.6, 0.6, 0.6, 0.9, 0.1] * 2
MUTPB = [0.00, 0.01, 0.10, 0.01, 0.01] * 2

if __name__ == "__main__":
    FITNS = []
    for iter in range(10):
        print(f"Iteration: {iter}")
        for i in range(10):
            _, _, _, best, inscr, stats = main(POP[i], CXPB[i], MUTPB[i])
            print(f"\tBest: {best}, Inscription: {inscr}")
            FITNS.append([best, inscr, stats])
    print(FITNS)
