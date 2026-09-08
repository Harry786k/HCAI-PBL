import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from django.shortcuts import render
from django.conf import settings


def index(request):
    context = {}

    if request.method == "POST" and "csv_file" in request.FILES:
        try:
            # Load CSV
            df = pd.read_csv(request.FILES["csv_file"]).dropna()
            X, y = df.iloc[:, :-1], df.iloc[:, -1]

            # Task 3: Visualization
            plt.figure(figsize=(8, 5))
            plt.scatter(
                X.iloc[:, 0],
                X.iloc[:, 1],
                c=pd.factorize(y)[0],
                alpha=0.7
            )
            plt.xlabel(X.columns[0])
            plt.ylabel(X.columns[1])
            plt.title(f"Scatter Plot: {X.columns[0]} vs {X.columns[1]}")

            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            filename = "project1_scatter.png"
            plt.savefig(os.path.join(settings.MEDIA_ROOT, filename))
            plt.close()

            # Task 4: Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Test different k values
            results = []

            for k in [1, 3, 5, 7, 9]:
                if k <= len(X_train):
                    model = KNeighborsClassifier(n_neighbors=k)
                    model.fit(X_train, y_train)
                    acc = accuracy_score(y_test, model.predict(X_test))
                    results.append((acc, k))

            best_accuracy, best_k = max(results)

            context.update({
                "plot_url": settings.MEDIA_URL + filename,
                "algorithm": "K-Nearest Neighbors (KNN)",
                "best_k": best_k,
                "accuracy": round(best_accuracy * 100, 2),
                "training_percentage": 80,
                "testing_percentage": 20
            })

        except Exception as e:
            context["error"] = str(e)

    return render(request, "project1/index.html", context)