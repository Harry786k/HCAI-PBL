from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("design1/", views.design1_view, name="design1"),
    path("design2/", views.design2_view, name="design2"),
    path("complete/", views.complete_view, name="complete"),
]