import random
from functools import lru_cache

from django.shortcuts import render
from datasets import load_dataset


@lru_cache(maxsize=1)
def get_articles():
    dataset = load_dataset(
        "fancyzhx/ag_news",
        split="train[:500]"
    )
    return dataset["text"]


def expert_interface(request):
    message = ""

    if request.method == "POST":
        label = request.POST.get("label")
        message = f"Success! You labeled the previous article as: {label}."

    try:
        article = random.choice(get_articles())
    except Exception:
        article = "The AG News dataset could not be loaded."
        message = "Please check the dataset connection and try again."

    return render(
        request,
        "project3/interface.html",
        {
            "article": article,
            "message": message
        }
    )