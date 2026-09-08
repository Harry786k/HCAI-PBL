import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from palmerpenguins import load_penguins

def index(request):
    penguins = load_penguins().dropna()
    X_raw = penguins.drop(columns=['species'])
    y = penguins['species']
    X = pd.get_dummies(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    media_dir = settings.MEDIA_ROOT
    if not os.path.exists(media_dir):
        os.makedirs(media_dir)

    # 1. DECISION TREE MODELS
    max_leaf_options = [2, 3, 4, 5, 7, 10, None]
    tree_data = []
    for opt in max_leaf_options:
        model = DecisionTreeClassifier(max_leaf_nodes=opt, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        leaves = model.get_n_leaves()
        filename = f'tree_leaves_{leaves}.png'
        plt.figure(figsize=(12, 8))
        plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
        plt.savefig(os.path.join(media_dir, filename))
        plt.close()
        
        tree_data.append({
            'param_val': 'None' if opt is None else opt,
            'complexity_score': int(leaves),
            'accuracy': float(acc),
            'error_rate': float(1.0 - acc), 
            'image_url': settings.MEDIA_URL + filename
        })

    # 2. LOGISTIC REGRESSION MODELS
    c_options = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    logreg_data = []
    for c_val in c_options:
        model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        complexity = float(np.sum(np.square(model.coef_)))
        filename = f'logreg_c_{c_val}.png'
        
        plt.figure(figsize=(10, 6))
        plt.imshow(model.coef_, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Coefficient Value')
        plt.yticks(ticks=range(len(model.classes_)), labels=model.classes_)
        plt.xticks(ticks=range(len(X.columns)), labels=X.columns, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(media_dir, filename))
        plt.close()
        
        logreg_data.append({
            'param_val': c_val,
            'complexity_score': float(round(complexity, 2)),
            'accuracy': float(acc),
            'error_rate': float(1.0 - acc), 
            'image_url': settings.MEDIA_URL + filename
        })
        
    context = {
        'trees_json': json.dumps(tree_data),
        'logreg_json': json.dumps(logreg_data),
        # Pass a list of penguins to the frontend for the dropdown menu
        'penguins_list': json.dumps([f"Penguin {i} ({species})" for i, species in enumerate(y)])
    }
    return render(request, "project2/index.html", context)

# ==========================================
# TASK 4: COUNTERFACTUAL ALGORITHM
# ==========================================
def generate_counterfactuals(request):
    if request.method != "POST":
        return JsonResponse({
            "status": "error",
            "message": "POST request required."
        }, status=405)

    try:
        data = json.loads(request.body)

        # ==========================================
        # 1. LOAD AND PREPARE DATA
        # ==========================================
        penguins = load_penguins().dropna().reset_index(drop=True)

        X_raw = penguins.drop(columns=["species"])
        y = penguins["species"]

        # Convert categorical variables to dummy variables
        X = pd.get_dummies(X_raw, dtype=float)

        # Use the same type of split as the main model
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        penguin_idx = int(data["penguin_idx"])
        target_class = data["target_class"]

        # Check selected penguin
        if penguin_idx < 0 or penguin_idx >= len(X):
            return JsonResponse({
                "status": "error",
                "message": "Invalid penguin selected."
            })

        # Check target species
        if target_class not in y.unique():
            return JsonResponse({
                "status": "error",
                "message": "Invalid target species."
            })

        current_species = y.iloc[penguin_idx]

        # A counterfactual should have a different target
        if target_class == current_species:
            return JsonResponse({
                "status": "error",
                "message": (
                    f"This penguin is already {current_species}. "
                    "Please choose a different desired species."
                )
            })

        # Original penguin
        x_orig_raw = X_raw.iloc[penguin_idx].copy()
        x_orig = X.iloc[penguin_idx].astype(float).copy()

        # ==========================================
        # 2. RECREATE THE CURRENTLY SELECTED MODEL
        # ==========================================
        if data["model_type"] == "tree":

            opt = (
                None
                if str(data["param_val"]) == "None"
                else int(data["param_val"])
            )

            model = DecisionTreeClassifier(
                max_leaf_nodes=opt,
                random_state=42
            )

        else:

            model = LogisticRegression(
                C=float(data["param_val"]),
                max_iter=1000,
                random_state=42
            )

        # Train using training data
        model.fit(X_train, y_train)

        # ==========================================
        # 3. GENERATE LOCAL SAMPLES
        # ==========================================
        N = 5000

        numeric_features = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]

        categorical_features = [
            "island",
            "sex"
        ]

        # Try different noise levels if necessary
        noise_scales = [0.25, 0.5, 0.75, 1.0]

        valid_samples = None
        valid_raw_samples = None

        for noise_scale in noise_scales:

            # Make N copies of the selected penguin
            sampled_raw = pd.DataFrame([
                x_orig_raw.to_dict()
                for _ in range(N)
            ])

            # ------------------------------------------
            # Numerical features
            # ------------------------------------------
            for col in numeric_features:

                std_value = X_raw[col].std()

                sampled_values = np.random.normal(
                    loc=float(x_orig_raw[col]),
                    scale=float(std_value * noise_scale),
                    size=N
                )

                # Keep values inside realistic dataset range
                sampled_values = np.clip(
                    sampled_values,
                    X_raw[col].min(),
                    X_raw[col].max()
                )

                sampled_raw[col] = sampled_values

            # ------------------------------------------
            # Categorical features
            # ------------------------------------------
            for col in categorical_features:

                possible_values = X_raw[col].dropna().unique()

                # Change category for around 10% of samples
                change_mask = np.random.rand(N) < 0.10

                if change_mask.any():
                    sampled_raw.loc[
                        change_mask,
                        col
                    ] = np.random.choice(
                        possible_values,
                        size=int(change_mask.sum())
                    )

            # ------------------------------------------
            # Year
            # ------------------------------------------
            possible_years = X_raw["year"].dropna().unique()

            year_change_mask = np.random.rand(N) < 0.10

            if year_change_mask.any():
                sampled_raw.loc[
                    year_change_mask,
                    "year"
                ] = np.random.choice(
                    possible_years,
                    size=int(year_change_mask.sum())
                )

            # Convert samples to dummy representation
            sampled_X = pd.get_dummies(
                sampled_raw,
                dtype=float
            )

            # Make sure columns exactly match model input
            sampled_X = sampled_X.reindex(
                columns=X.columns,
                fill_value=0.0
            ).astype(float)

            # ==========================================
            # 4. FIND SAMPLES WITH DESIRED CLASS
            # ==========================================
            predictions = model.predict(sampled_X)

            valid_idx = np.where(
                predictions == target_class
            )[0]

            if len(valid_idx) > 0:

                valid_samples = (
                    sampled_X
                    .iloc[valid_idx]
                    .copy()
                    .reset_index(drop=True)
                )

                valid_raw_samples = (
                    sampled_raw
                    .iloc[valid_idx]
                    .copy()
                    .reset_index(drop=True)
                )

                break

        # No counterfactual found
        if valid_samples is None or len(valid_samples) == 0:

            return JsonResponse({
                "status": "error",
                "message": (
                    "No counterfactuals found for this target. "
                    "Please try another penguin or target species."
                )
            })

        # ==========================================
        # 5. MAD-WEIGHTED L1 DISTANCE
        # ==========================================
        mad = X_train.apply(
            lambda col: np.median(
                np.abs(
                    col - np.median(col)
                )
            )
        )

        # Prevent division by zero
        mad = mad.replace(0, 1.0)

        distances = (
            valid_samples - x_orig
        ).abs() / mad

        valid_samples["distance"] = distances.sum(axis=1)

        # ==========================================
        # 6. SELECT BEST 3 COUNTERFACTUALS
        # ==========================================
        best_indices = (
            valid_samples["distance"]
            .sort_values()
            .head(3)
            .index
        )

        results = []

        for idx in best_indices:

            encoded_row = valid_samples.iloc[idx]
            raw_row = valid_raw_samples.iloc[idx]

            results.append({
                "bill_length": round(
                    float(raw_row["bill_length_mm"]),
                    1
                ),

                "bill_depth": round(
                    float(raw_row["bill_depth_mm"]),
                    1
                ),

                "flipper": round(
                    float(raw_row["flipper_length_mm"]),
                    1
                ),

                "mass": round(
                    float(raw_row["body_mass_g"]),
                    1
                ),

                "island": str(
                    raw_row["island"]
                ),

                "sex": str(
                    raw_row["sex"]
                ),

                "year": int(
                    raw_row["year"]
                ),

                "distance": round(
                    float(encoded_row["distance"]),
                    2
                )
            })

        # ==========================================
        # 7. RETURN RESULTS TO HTML
        # ==========================================
        return JsonResponse({
            "status": "success",
            "data": results
        })

    except Exception as error:

        print("COUNTERFACTUAL ERROR:", error)

        return JsonResponse({
            "status": "error",
            "message": str(error)
        })
    # ==========================================
# TASK 5: PDP and ALE ALGORITHMS
# ==========================================
def feature_effects(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        feature = data['feature']
        
        # 1. Load Data and Rebuild Model
        penguins = load_penguins().dropna()
        X = pd.get_dummies(penguins.drop(columns=['species']))
        y = penguins['species']
        
        if data['model_type'] == 'tree':
            opt = None if data['param_val'] == 'None' else int(data['param_val'])
            model = DecisionTreeClassifier(max_leaf_nodes=opt, random_state=42)
        else:
            model = LogisticRegression(C=float(data['param_val']), max_iter=1000, random_state=42)
        model.fit(X, y)
        
        media_dir = settings.MEDIA_ROOT
        
        # ==========================================
        # CALCULATE PDP (From Scratch)
        # ==========================================
        # Create a grid of 50 points across the feature's min and max
        grid_values = np.linspace(X[feature].min(), X[feature].max(), 50)
        pdp_results = {cls: [] for cls in model.classes_}
        
        for val in grid_values:
            X_temp = X.copy()
            X_temp[feature] = val # Force the feature to the grid value for ALL instances
            
            # Predict probabilities
            probs = model.predict_proba(X_temp)
            mean_probs = probs.mean(axis=0) # Average the probabilities
            
            for i, cls in enumerate(model.classes_):
                pdp_results[cls].append(mean_probs[i])
                
        # Draw PDP Plot
        plt.figure(figsize=(8, 5))
        for cls in model.classes_:
            plt.plot(grid_values, pdp_results[cls], label=cls, linewidth=2)
        plt.title(f'Partial Dependence Plot (PDP): {feature}')
        plt.xlabel(feature)
        plt.ylabel('Average Predicted Probability')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        pdp_filename = f'pdp_{feature}.png'
        plt.savefig(os.path.join(media_dir, pdp_filename))
        plt.close()

        # ==========================================
        # CALCULATE ALE (From Scratch via Discretization)
        # ==========================================
        # Create 20 quantile-based bins (intervals)
        K = 20
        quantiles = np.linspace(0, 1, K + 1)
        z = np.unique(np.quantile(X[feature], quantiles)) # Bin edges
        
        ale_effects = {cls: np.zeros(len(z)-1) for cls in model.classes_}
        bin_centers = (z[:-1] + z[1:]) / 2
        
        for k in range(len(z)-1):
            # Find instances that fall into this bin
            lower, upper = z[k], z[k+1]
            indices = (X[feature] >= lower) & (X[feature] <= upper)
            
            if not indices.any(): continue
                
            X_bin = X[indices].copy()
            
            # Replace feature with upper bound and get probabilities
            X_bin[feature] = upper
            prob_upper = model.predict_proba(X_bin)
            
            # Replace feature with lower bound and get probabilities
            X_bin[feature] = lower
            prob_lower = model.predict_proba(X_bin)
            
            # Calculate the local effect (discretized partial derivative)
            diffs = prob_upper - prob_lower
            mean_diffs = diffs.mean(axis=0)
            
            for i, cls in enumerate(model.classes_):
                ale_effects[cls][k] = mean_diffs[i]
                
        # Accumulate the effects and center them
        ale_results = {cls: np.cumsum(ale_effects[cls]) for cls in model.classes_}
        for cls in model.classes_:
            ale_results[cls] -= ale_results[cls].mean()
            
        # Draw ALE Plot
        plt.figure(figsize=(8, 5))
        for cls in model.classes_:
            plt.plot(bin_centers, ale_results[cls], label=cls, linewidth=2, marker='o', markersize=4)
        plt.title(f'Accumulated Local Effects (ALE): {feature}')
        plt.xlabel(feature)
        plt.ylabel('Centered ALE Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        ale_filename = f'ale_{feature}.png'
        plt.savefig(os.path.join(media_dir, ale_filename))
        plt.close()

        return JsonResponse({
            'status': 'success',
            'pdp_url': settings.MEDIA_URL + pdp_filename,
            'ale_url': settings.MEDIA_URL + ale_filename
        })